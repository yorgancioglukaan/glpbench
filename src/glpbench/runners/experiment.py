import torch
from glpbench.datasets import load_dataset
import argparse
import copy
import json
import random
import time
from dataclasses import dataclass

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from glpbench.logging import (
    CSVResultLogger,
    ExperimentResult,
    make_configuration_id,
)
from glpbench.preprocessing import available_reduction_methods
from glpbench.training import (
    available_lr_schedulers,
    build_lr_scheduler,
    current_learning_rate,
    step_lr_scheduler,
    validate_scheduler_config,
)
from glpbench.evaluation import prepare_candidate_data
from glpbench.models import (
    LinkPrediction,
    available_agae_regularizers,
    available_model_types,
    count_trainable_parameters,
    get_model_spec,
    sample_training_negative_edges,
)
from glpbench.protocols import (
    generate_comparable_splits,
    select_query_node_sets,
)


def calculate_metrics(labels, probabilities):
    """Calculate binary link-prediction metrics."""
    labels = labels.detach().cpu().numpy()
    probabilities = probabilities.detach().cpu().numpy()
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(labels, probabilities),
        "average_precision": average_precision_score(labels, probabilities),
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


def _json_object(value):
    """Parse a JSON object for an argparse option."""
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError("must be valid JSON") from error
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError("must decode to a JSON object")
    return parsed


def _synchronize_device(device):
    """Wait for queued accelerator work before recording a timestamp."""
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "synchronize")
    ):
        torch.mps.synchronize()


def _resolve_seeds(args):
    """Return validated seeds from explicit or generated CLI settings."""
    supplied_seeds = getattr(args, "seed", None)
    num_seeds = getattr(args, "num_seeds", None)
    if num_seeds is not None:
        if supplied_seeds not in (None, []):
            raise ValueError("seed and num_seeds cannot be combined")
        if (
            not isinstance(num_seeds, int)
            or isinstance(num_seeds, bool)
            or num_seeds <= 0
        ):
            raise ValueError("num_seeds must be a positive integer")
        return tuple(range(num_seeds))

    if supplied_seeds is None:
        seeds = (42,)
    elif isinstance(supplied_seeds, int) and not isinstance(supplied_seeds, bool):
        seeds = (supplied_seeds,)
    else:
        seeds = tuple(supplied_seeds)
    if not seeds:
        raise ValueError("at least one seed is required")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ValueError("seeds must be non-negative integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


@dataclass(frozen=True)
class TrainingSummary:
    """Store fixed-budget training and checkpoint-selection outcomes."""

    selected_loss: torch.Tensor | None
    training_time_seconds: float | None
    epochs_trained: int
    best_epoch: int | None
    selected_epoch: int | None
    checkpoint_strategy: str | None
    selection_metric: str | None
    best_validation_score: float | None
    early_stopping_patience: int | None
    early_stopping_min_delta: float
    early_stopping_minimum_epochs: int
    stopped_early: bool
    lr_scheduler: str
    lr_scheduler_kwargs: dict
    final_learning_rate: float | None


def train_with_validation_selection(
    predictor, train_data, validation_data, *, epochs, seed,
    selection_metric, device, early_stopping_patience=None,
    early_stopping_min_delta=0.0, early_stopping_minimum_epochs=0,
    checkpoint_strategy="best-validation", verbose=False,
    progress_label="model", progress_position=1, compact_progress=False,
    epoch_callback=None, lr_scheduler="none", lr_scheduler_kwargs=None,
):
    """Train with validation checkpointing and optional early stopping."""
    if selection_metric not in {"roc_auc", "average_precision"}:
        raise ValueError("unsupported checkpoint selection metric")
    if checkpoint_strategy not in {"best-validation", "final"}:
        raise ValueError("unsupported checkpoint strategy")
    if epoch_callback is not None and not callable(epoch_callback):
        raise ValueError("epoch_callback must be callable")
    if early_stopping_patience is not None and (
        not isinstance(early_stopping_patience, int)
        or isinstance(early_stopping_patience, bool)
        or early_stopping_patience <= 0
    ):
        raise ValueError("early_stopping_patience must be a positive integer")
    if (
        not isinstance(early_stopping_min_delta, (int, float))
        or isinstance(early_stopping_min_delta, bool)
        or early_stopping_min_delta < 0
    ):
        raise ValueError("early_stopping_min_delta must be non-negative")
    if (
        not isinstance(early_stopping_minimum_epochs, int)
        or isinstance(early_stopping_minimum_epochs, bool)
        or not 0 <= early_stopping_minimum_epochs <= epochs
    ):
        raise ValueError(
            "early_stopping_minimum_epochs must be between zero and epochs"
        )
    scheduler_parameters = validate_scheduler_config(
        lr_scheduler, lr_scheduler_kwargs, epochs=epochs
    )
    if not predictor.trainable:
        if lr_scheduler != "none":
            raise ValueError("learning-rate schedulers require a trainable model")
        return TrainingSummary(
            selected_loss=None,
            training_time_seconds=None,
            epochs_trained=0,
            best_epoch=None,
            selected_epoch=None,
            checkpoint_strategy=None,
            selection_metric=None,
            best_validation_score=None,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=float(early_stopping_min_delta),
            early_stopping_minimum_epochs=early_stopping_minimum_epochs,
            stopped_early=False,
            lr_scheduler="none",
            lr_scheduler_kwargs={},
            final_learning_rate=None,
        )

    scheduler = build_lr_scheduler(
        getattr(predictor, "optimizer", None),
        lr_scheduler,
        scheduler_parameters,
        epochs=epochs,
    )

    epoch_progress = tqdm(
        range(epochs),
        desc=f"{progress_label} epochs",
        unit="epoch",
        disable=not verbose,
        leave=False,
        position=progress_position,
    )
    best_state = None
    best_epoch = None
    best_score = None
    stopping_reference_score = None
    selected_loss = None
    final_loss = None
    epochs_without_improvement = 0
    epochs_trained = 0
    stopped_early = False

    _synchronize_device(device)
    training_start = time.perf_counter()
    for epoch in epoch_progress:
        negative_edges = sample_training_negative_edges(
            train_data, seed=seed + epoch
        )
        loss = predictor.train(negative_edge_index=negative_edges)
        final_loss = loss.detach().clone()
        validation_probabilities = predictor.test(validation_data)
        validation_metrics = calculate_metrics(
            validation_data.edge_label, validation_probabilities
        )
        validation_score = float(validation_metrics[selection_metric])
        learning_rate = current_learning_rate(getattr(predictor, "optimizer", None))
        if epoch_callback is not None:
            epoch_callback({
                "epoch": epoch + 1,
                "training_loss": float(loss.detach().item()),
                "learning_rate": learning_rate,
                **{
                    f"validation_{name}": float(value)
                    for name, value in validation_metrics.items()
                },
            })
        if best_score is None or validation_score > best_score:
            best_score = validation_score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(predictor.model.state_dict())
            selected_loss = loss.detach().clone()

        step_lr_scheduler(scheduler, lr_scheduler, validation_score)

        epochs_trained = epoch + 1
        stopping_active = (
            early_stopping_patience is not None
            and epochs_trained >= max(1, early_stopping_minimum_epochs)
        )
        if stopping_active:
            significant_improvement = (
                stopping_reference_score is None
                or validation_score
                > stopping_reference_score + float(early_stopping_min_delta)
            )
            if significant_improvement:
                stopping_reference_score = validation_score
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

        if verbose:
            component_values = {
                "loss": f"{loss.detach().item():.4f}",
                "lr": f"{learning_rate:.2g}" if learning_rate is not None else "n/a",
                f"val_{selection_metric}": f"{validation_score:.4f}",
            }
            if compact_progress:
                component_values[f"best_{selection_metric}"] = f"{best_score:.4f}"
                if early_stopping_patience is not None:
                    component_values["patience"] = (
                        f"{epochs_without_improvement}/"
                        f"{early_stopping_patience}"
                    )
            else:
                component_values = {
                    name: f"{value.detach().item():.4f}"
                    for name, value in predictor.last_loss_components.items()
                } | component_values
            epoch_progress.set_postfix(component_values)

        if (
            stopping_active
            and epochs_without_improvement >= early_stopping_patience
        ):
            stopped_early = epochs_trained < epochs
            break

    if best_state is None:
        raise RuntimeError("training did not produce a validation checkpoint")
    if checkpoint_strategy == "best-validation":
        predictor.model.load_state_dict(best_state)
        selected_epoch = best_epoch
    else:
        selected_loss = final_loss
        selected_epoch = epochs_trained
    _synchronize_device(device)
    training_time_seconds = time.perf_counter() - training_start
    return TrainingSummary(
        selected_loss=selected_loss,
        training_time_seconds=training_time_seconds,
        epochs_trained=epochs_trained,
        best_epoch=best_epoch,
        selected_epoch=selected_epoch,
        checkpoint_strategy=checkpoint_strategy,
        selection_metric=selection_metric,
        best_validation_score=best_score,
        early_stopping_patience=early_stopping_patience,
        early_stopping_min_delta=float(early_stopping_min_delta),
        early_stopping_minimum_epochs=early_stopping_minimum_epochs,
        stopped_early=stopped_early,
        lr_scheduler=lr_scheduler,
        lr_scheduler_kwargs=scheduler_parameters,
        final_learning_rate=current_learning_rate(
            getattr(predictor, "optimizer", None)
        ),
    )


# Compatibility alias for existing API users and tests.
_train_with_checkpoint_selection = train_with_validation_selection


def parse_args():
    """Parse command-line experiment parameters."""
    parser = argparse.ArgumentParser(description="Run a link-prediction experiment")
    parser.add_argument("dataset", help="benchmark dataset name")
    parser.add_argument(
        "out_filepath", nargs="?", default="out/results.csv",
        help="path to the result CSV",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--model-type",
        choices=available_model_types(),
        action="append",
        default=None,
        help="architecture to run; repeat to select multiple, or omit to run all",
    )
    parser.add_argument(
        "--conv-layer-type",
        choices=("GCNConv", "GraphConv", "SAGEConv", "GATv2Conv"),
        default="GCNConv",
    )
    parser.add_argument(
        "--gatv2-kwargs",
        type=_json_object,
        default=None,
        help="JSON object of GATv2Conv options",
    )
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--out-channels", type=int, default=64)
    parser.add_argument(
        "--reduction-method",
        choices=available_reduction_methods(),
        default="pca",
        help="node-feature reduction method used by asymmetric models",
    )
    parser.add_argument("--decoder-hidden-channels", type=int, default=None)
    parser.add_argument("--decoder-dropout", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=1.0)
    parser.add_argument(
        "--agae-regularization",
        choices=available_agae_regularizers(),
        action="append",
        default=None,
        help="AGAE regularizer; repeat to compare several",
    )
    parser.add_argument("--regularization-strength", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--deal-theta-ss", type=float, default=1.0)
    parser.add_argument("--deal-theta-aa", type=float, default=1.0)
    parser.add_argument("--deal-theta-sa", type=float, default=1.0)
    parser.add_argument("--deal-lambda-ss", type=float, default=1.0)
    parser.add_argument("--deal-lambda-aa", type=float, default=1.0)
    parser.add_argument("--deal-lambda-sa", type=float, default=1.0)
    parser.add_argument("--deal-similarity-scale", type=float, default=1.0)
    parser.add_argument("--deal-similarity-margin", type=float, default=0.0)
    query_group = parser.add_mutually_exclusive_group()
    query_group.add_argument("--query-size", type=int, default=None)
    query_group.add_argument("--query-fraction", type=float, default=None)
    validation_query_group = parser.add_mutually_exclusive_group()
    validation_query_group.add_argument(
        "--validation-query-size", type=int, default=None
    )
    validation_query_group.add_argument(
        "--validation-query-fraction", type=float, default=None
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--lr-scheduler",
        choices=available_lr_schedulers(),
        default="none",
    )
    parser.add_argument(
        "--lr-scheduler-kwargs",
        type=_json_object,
        default=None,
        help="JSON object of learning-rate scheduler parameters",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("roc_auc", "average_precision"),
        default="roc_auc",
        help="semi-inductive validation metric used to select the checkpoint",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="stop after this many non-improving epochs; disabled when omitted",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="minimum validation improvement when early stopping is enabled",
    )
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        "--seed",
        type=int,
        action="append",
        default=None,
        help="random seed; repeat to run multiple seeds (default: 42)",
    )
    seed_group.add_argument(
        "--num-seeds",
        type=int,
        default=None,
        help="run generated seeds 0 through N-1",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--verbose", action="store_true", help="show experiment progress"
    )
    parser.add_argument(
        "--normalize-features", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def run_experiment(
    args, run_id=None, completed_results=None, *, progress_callback=None,
    compact_progress=False,
):
    """Run one training experiment and log both test settings."""
    if completed_results is None:
        completed_results = set()
    elif not isinstance(completed_results, set):
        raise TypeError("completed_results must be a set")
    query_size = getattr(args, "query_size", None)
    query_fraction = getattr(args, "query_fraction", None)
    validation_query_size = getattr(args, "validation_query_size", None)
    validation_query_fraction = getattr(
        args, "validation_query_fraction", None
    )
    if query_size is None and query_fraction is None:
        query_size = 10
    if args.epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    regularization_strength = getattr(args, "regularization_strength", 0.0)
    learning_rate = getattr(args, "learning_rate", 0.01)
    weight_decay = getattr(args, "weight_decay", 0.0)
    if regularization_strength < 0:
        raise ValueError("regularization_strength must be non-negative")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")

    seeds = _resolve_seeds(args)
    device = torch.device(args.device)
    verbose = getattr(args, "verbose", False)
    detailed_verbose = verbose and not compact_progress

    def emit(event, **values):
        if progress_callback is not None:
            progress_callback(event, **values)

    def report(message):
        if detailed_verbose:
            tqdm.write(message)

    if args.model_type is None:
        model_types = available_model_types()
    elif isinstance(args.model_type, str):
        model_types = (args.model_type,)
    else:
        model_types = tuple(dict.fromkeys(args.model_type))

    selected_regularizers = getattr(args, "agae_regularization", None)
    if selected_regularizers is None:
        selected_regularizers = ("none",)
    elif isinstance(selected_regularizers, str):
        selected_regularizers = (selected_regularizers,)
    else:
        selected_regularizers = tuple(dict.fromkeys(selected_regularizers))
    if not selected_regularizers:
        raise ValueError("at least one AGAE regularization method is required")
    unknown_regularizers = set(selected_regularizers).difference(
        available_agae_regularizers()
    )
    if unknown_regularizers:
        raise ValueError(
            f"Unknown AGAE regularization: {sorted(unknown_regularizers)[0]}"
        )
    model_specs = {
        model_type: get_model_spec(model_type) for model_type in model_types
    }
    if (
        not any(spec.supports_agae_regularization for spec in model_specs.values())
        and selected_regularizers != ("none",)
    ):
        raise ValueError(
            "AGAE regularization requires an asymmetric model type"
        )
    experiment_configs = [
        (model_type, regularization_type)
        for model_type in model_types
        for regularization_type in (
            selected_regularizers
            if model_specs[model_type].supports_agae_regularization
            else ("none",)
        )
    ]

    # Load once, then clone before seed-specific preprocessing.
    emit("stage", dataset=args.dataset, stage="loading")
    report(f"Loading {args.dataset}...")
    source_data = load_dataset(
        args.dataset,
        root=args.data_root,
        normalize_features=False,
        device="cpu",
    )
    reduction_method = getattr(args, "reduction_method", "pca")
    gatv2_kwargs = getattr(args, "gatv2_kwargs", None) or {}
    if not isinstance(gatv2_kwargs, dict):
        raise ValueError("gatv2_kwargs must be a dictionary")
    logger = CSVResultLogger(args.out_filepath, run_id=run_id)
    decoder_hidden_channels = getattr(args, "decoder_hidden_channels", None)
    decoder_dropout = getattr(args, "decoder_dropout", 0.0)
    kl_weight = getattr(args, "kl_weight", 1.0)
    selection_metric = getattr(args, "selection_metric", "roc_auc")
    early_stopping_patience = getattr(
        args, "early_stopping_patience", None
    )
    early_stopping_min_delta = getattr(
        args, "early_stopping_min_delta", 0.0
    )
    early_stopping_minimum_epochs = getattr(
        args, "early_stopping_minimum_epochs", 0
    )
    checkpoint_strategy = getattr(
        args, "checkpoint_strategy", "best-validation"
    )
    training_policy = getattr(args, "training_policy", "default")
    epoch_callback = getattr(args, "epoch_callback", None)
    lr_scheduler = getattr(args, "lr_scheduler", "none")
    lr_scheduler_kwargs = getattr(args, "lr_scheduler_kwargs", None)
    if not isinstance(training_policy, str) or not training_policy.strip():
        raise ValueError("training_policy must be a non-empty string")
    if early_stopping_patience is not None and (
        not isinstance(early_stopping_patience, int)
        or isinstance(early_stopping_patience, bool)
        or early_stopping_patience <= 0
    ):
        raise ValueError("early_stopping_patience must be a positive integer")
    if (
        not isinstance(early_stopping_min_delta, (int, float))
        or isinstance(early_stopping_min_delta, bool)
        or early_stopping_min_delta < 0
    ):
        raise ValueError("early_stopping_min_delta must be non-negative")
    deal_config = {
        "theta_ss": getattr(args, "deal_theta_ss", 1.0),
        "theta_aa": getattr(args, "deal_theta_aa", 1.0),
        "theta_sa": getattr(args, "deal_theta_sa", 1.0),
        "lambda_ss": getattr(args, "deal_lambda_ss", 1.0),
        "lambda_aa": getattr(args, "deal_lambda_aa", 1.0),
        "lambda_sa": getattr(args, "deal_lambda_sa", 1.0),
        "similarity_scale": getattr(args, "deal_similarity_scale", 1.0),
        "similarity_margin": getattr(args, "deal_similarity_margin", 0.0),
    }
    seed_progress = tqdm(
        seeds,
        desc="Seeds",
        unit="seed",
        disable=not detailed_verbose or len(seeds) == 1,
        position=0,
    )
    for seed in seed_progress:
        emit("stage", dataset=args.dataset, seed=seed, stage="splitting")
        random.seed(seed)
        torch.manual_seed(seed)
        data = source_data.clone()
        query_nodes = select_query_node_sets(
            data,
            query_size=query_size,
            query_fraction=query_fraction,
            validation_query_size=validation_query_size,
            validation_query_fraction=validation_query_fraction,
            seed=seed,
        )

        if any(spec.uses_reduced_features for spec in model_specs.values()):
            fit_mask = torch.ones(data.num_nodes, dtype=torch.bool)
            fit_mask[list(query_nodes.validation)] = False
            fit_mask[list(query_nodes.test)] = False
            num_fit_nodes = int(fit_mask.sum())
            if args.out_channels > min(num_fit_nodes, data.num_features):
                raise ValueError(
                    "out_channels cannot exceed the smaller of the number of "
                    "non-query nodes and num_features"
                )

        report(f"Generating comparable graph splits for seed {seed}...")
        splits = generate_comparable_splits(
            data,
            query=query_nodes.test,
            validation_query=query_nodes.validation,
            train_ratio=args.train_ratio,
            seed=seed,
        )
        if any(spec.uses_reduced_features for spec in model_specs.values()):
            report(f"Calculating reduced node features for seed {seed}...")
            preparation_model_type = "asymmetric"
        else:
            preparation_model_type = model_types[0]
        emit("stage", dataset=args.dataset, seed=seed, stage="preprocessing")
        candidate_data = prepare_candidate_data(
            data,
            splits,
            model_type=preparation_model_type,
            normalize_features=args.normalize_features,
            out_channels=args.out_channels,
            reduction_method=reduction_method,
            random_state=seed,
        )
        reducer_fit_time_seconds = candidate_data.reducer_fit_time_seconds
        train = candidate_data.train.to(device)
        validation = candidate_data.validation.to(device)
        test_transductive = candidate_data.test_transductive.to(device)
        test_inductive = candidate_data.test_inductive.to(device)

        model_progress = tqdm(
            experiment_configs,
            desc=f"Seed {seed} architectures",
            unit="model",
            disable=not detailed_verbose,
            leave=len(seeds) == 1,
            position=1 if len(seeds) > 1 else 0,
        )
        for model_type, regularization_type in model_progress:
            model_spec = model_specs[model_type]
            progress_label = model_type
            if model_spec.supports_agae_regularization:
                progress_label += f"/{regularization_type}"
            model_progress.set_postfix_str(progress_label)
            emit(
                "run_started",
                dataset=args.dataset,
                model_type=model_type,
                seed=seed,
                stage="initializing",
            )
            torch.manual_seed(seed)
            effective_weight_decay = (
                weight_decay if model_spec.trainable else 0.0
            )
            predictor_kwargs = dict(
                model_type=model_type,
                conv_layer_type=args.conv_layer_type,
                hidden_channels=args.hidden_channels,
                out_channels=args.out_channels,
                decoder_hidden_channels=decoder_hidden_channels,
                decoder_dropout=decoder_dropout,
                kl_weight=kl_weight,
                learning_rate=learning_rate,
                weight_decay=effective_weight_decay,
                gatv2_kwargs=gatv2_kwargs,
                **deal_config,
            )
            if model_spec.supports_agae_regularization:
                predictor_kwargs.update({
                    "regularization_type": regularization_type,
                    "regularization_strength": regularization_strength,
                })
            predictor = LinkPrediction(train, **predictor_kwargs)
            predictor.model = predictor.model.to(device)
            trainable_parameter_count = count_trainable_parameters(
                predictor.model
            )
            effective_conv_layer = (
                args.conv_layer_type
                if model_spec.uses_conv_layer else "None"
            )
            extra_fields = {
                "reduction_method": reduction_method,
                "validation_query_size": len(query_nodes.validation),
                "learning_rate": learning_rate,
                "device": str(device),
                "query_selection": {
                    "mode": (
                        "fraction" if query_fraction is not None else "size"
                    ),
                    "value": (
                        query_fraction if query_fraction is not None else query_size
                    ),
                    "resolved_size": len(query_nodes.test),
                },
                "validation_query_selection": {
                    "mode": (
                        "fraction"
                        if validation_query_fraction is not None
                        else "size"
                        if validation_query_size is not None
                        else "match-test"
                    ),
                    "value": (
                        validation_query_fraction
                        if validation_query_fraction is not None
                        else validation_query_size
                    ),
                    "resolved_size": len(query_nodes.validation),
                },
            }
            if (
                model_spec.uses_conv_layer
                and args.conv_layer_type == "GATv2Conv"
            ):
                extra_fields["gatv2_kwargs"] = gatv2_kwargs
            if model_spec.uses_mlp_decoder:
                extra_fields.update({
                    "decoder_hidden_channels": decoder_hidden_channels,
                    "decoder_dropout": decoder_dropout,
                })
            elif model_type == "variational":
                extra_fields["kl_weight"] = kl_weight
            elif model_type == "deal":
                extra_fields.update({
                    f"deal_{name}": value for name, value in deal_config.items()
                })

            configuration_ids = {
                split_type: make_configuration_id({
                    "dataset": data.dataset_name,
                    "model_type": model_type,
                    "split_type": split_type,
                    "conv_layer_type": effective_conv_layer,
                    "hidden_channels": args.hidden_channels,
                    "out_channels": args.out_channels,
                    "query_size": len(query_nodes.test),
                    "validation_query_size": len(query_nodes.validation),
                    "train_ratio": args.train_ratio,
                    "epochs": args.epochs,
                    "normalize_features": args.normalize_features,
                    "regularization_type": regularization_type,
                    "regularization_strength": (
                        regularization_strength
                        if model_spec.supports_agae_regularization else 0.0
                    ),
                    "weight_decay": effective_weight_decay,
                    "selection_metric": selection_metric,
                    "training_policy": training_policy,
                    "checkpoint_strategy": checkpoint_strategy,
                    "lr_scheduler": lr_scheduler,
                    "lr_scheduler_kwargs": lr_scheduler_kwargs or {},
                    "early_stopping_patience": early_stopping_patience,
                    "early_stopping_min_delta": early_stopping_min_delta,
                    "early_stopping_minimum_epochs": (
                        early_stopping_minimum_epochs
                    ),
                    "extra_fields": extra_fields,
                })
                for split_type in ("transductive", "inductive")
            }
            missing_split_types = {
                split_type for split_type, configuration_id
                in configuration_ids.items()
                if (configuration_id, seed) not in completed_results
            }
            if not missing_split_types:
                report(
                    f"Skipping completed {progress_label} model for seed {seed}."
                )
                emit(
                    "run_skipped",
                    dataset=args.dataset,
                    model_type=model_type,
                    seed=seed,
                )
                continue
            if predictor.trainable:
                report(f"Training {progress_label} model for seed {seed}...")
            else:
                report(
                    f"Evaluating {model_type} baseline for seed {seed} "
                    "without training..."
                )
            emit(
                "stage",
                dataset=args.dataset,
                model_type=model_type,
                seed=seed,
                stage="training" if predictor.trainable else "evaluating",
            )
            training_epoch_callback = None
            if epoch_callback is not None:
                def training_epoch_callback(record):
                    epoch_callback({
                        "dataset": data.dataset_name,
                        "model_type": model_type,
                        "random_seed": seed,
                        "training_policy": training_policy,
                        **record,
                    })
            training_summary = _train_with_checkpoint_selection(
                predictor,
                train,
                validation,
                epochs=args.epochs,
                seed=seed,
                selection_metric=selection_metric,
                device=device,
                early_stopping_patience=early_stopping_patience,
                early_stopping_min_delta=early_stopping_min_delta,
                early_stopping_minimum_epochs=early_stopping_minimum_epochs,
                checkpoint_strategy=checkpoint_strategy,
                lr_scheduler=lr_scheduler,
                lr_scheduler_kwargs=lr_scheduler_kwargs,
                verbose=verbose,
                progress_label=(
                    f"{args.dataset}/{progress_label}/seed={seed}"
                    if compact_progress
                    else f"seed {seed} {progress_label}"
                ),
                progress_position=(
                    1 if compact_progress
                    else 2 if len(seeds) > 1 else 1
                ),
                compact_progress=compact_progress,
                epoch_callback=training_epoch_callback,
            )

            emit(
                "stage",
                dataset=args.dataset,
                model_type=model_type,
                seed=seed,
                stage="evaluating",
            )
            report(f"Evaluating {progress_label} model for seed {seed}...")
            for split_type, test_data in (
                ("transductive", test_transductive),
                ("inductive", test_inductive),
            ):
                if split_type not in missing_split_types:
                    continue
                probabilities = predictor.test(test_data)
                metrics = calculate_metrics(test_data.edge_label, probabilities)
                configuration_id = configuration_ids[split_type]
                result = ExperimentResult(
                    configuration_id=configuration_id,
                    dataset=data.dataset_name,
                    model_type=model_type,
                    split_type=split_type,
                    conv_layer_type=effective_conv_layer,
                    hidden_channels=args.hidden_channels,
                    out_channels=args.out_channels,
                    query_size=len(query_nodes.test),
                    train_ratio=args.train_ratio,
                    epochs=args.epochs,
                    random_seed=seed,
                    normalize_features=args.normalize_features,
                    training_time_seconds=training_summary.training_time_seconds,
                    trainable_parameter_count=trainable_parameter_count,
                    reducer_fit_time_seconds=(
                        reducer_fit_time_seconds
                        if model_spec.uses_reduced_features else None
                    ),
                    epochs_trained=training_summary.epochs_trained,
                    best_epoch=training_summary.best_epoch,
                    selected_epoch=training_summary.selected_epoch,
                    training_policy=training_policy,
                    checkpoint_strategy=training_summary.checkpoint_strategy,
                    lr_scheduler=training_summary.lr_scheduler,
                    lr_scheduler_kwargs=training_summary.lr_scheduler_kwargs,
                    final_learning_rate=training_summary.final_learning_rate,
                    selection_metric=training_summary.selection_metric,
                    best_validation_score=(
                        training_summary.best_validation_score
                    ),
                    early_stopping_patience=(
                        training_summary.early_stopping_patience
                    ),
                    early_stopping_min_delta=(
                        training_summary.early_stopping_min_delta
                    ),
                    early_stopping_minimum_epochs=(
                        training_summary.early_stopping_minimum_epochs
                    ),
                    stopped_early=training_summary.stopped_early,
                    regularization_type=regularization_type,
                    regularization_strength=(
                        regularization_strength
                        if model_spec.supports_agae_regularization else 0.0
                    ),
                    weight_decay=effective_weight_decay,
                    loss=(
                        float(training_summary.selected_loss.detach().cpu())
                        if training_summary.selected_loss is not None else None
                    ),
                    extra_fields=extra_fields,
                    **metrics,
                )
                logger.log(result)
                completed_results.add((configuration_id, seed))
            emit(
                "run_completed",
                dataset=args.dataset,
                model_type=model_type,
                seed=seed,
            )
    report(f"Results written to {args.out_filepath}")
    return logger.run_id


if __name__ == "__main__":
    run_id = run_experiment(parse_args())
    print(f"Completed experiment run {run_id}")
