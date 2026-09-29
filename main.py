import torch
from datasets import load_dataset
import argparse
import random
import time

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm.auto import tqdm

from experiment_logging import CSVResultLogger, ExperimentResult
from feature_reduction import available_reduction_methods
from link_prediction import (
    LinkPrediction,
    available_agae_regularizers,
    available_model_types,
    sample_training_negative_edges,
)
from utils import (
    calculate_reduced_features,
    generate_comparable_splits,
    select_query_nodes,
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
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--verbose", action="store_true", help="show experiment progress"
    )
    parser.add_argument(
        "--normalize-features", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def run_experiment(args):
    """Run one training experiment and log both test settings."""
    query_size = getattr(args, "query_size", None)
    query_fraction = getattr(args, "query_fraction", None)
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    verbose = getattr(args, "verbose", False)

    def report(message):
        if verbose:
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
    if "asymmetric" not in model_types and selected_regularizers != ("none",):
        raise ValueError(
            "AGAE regularization requires the asymmetric model type"
        )
    experiment_configs = [
        (model_type, regularization_type)
        for model_type in model_types
        for regularization_type in (
            selected_regularizers if model_type == "asymmetric" else ("none",)
        )
    ]

    # Load and prepare the source graph on CPU for splitting.
    report(f"Loading {args.dataset}...")
    data = load_dataset(
        args.dataset,
        root=args.data_root,
        normalize_features=args.normalize_features,
        device="cpu",
    )
    query = select_query_nodes(
        data,
        query_size=query_size,
        query_fraction=query_fraction,
        seed=args.seed,
    )

    reduction_method = getattr(args, "reduction_method", "pca")
    if "asymmetric" in model_types:
        report("Calculating reduced node features...")
        fit_mask = torch.ones(data.num_nodes, dtype=torch.bool)
        fit_mask[query] = False
        num_fit_nodes = int(fit_mask.sum())
        if args.out_channels > min(num_fit_nodes, data.num_features):
            raise ValueError(
                "out_channels cannot exceed the smaller of the number of "
                "non-query nodes and num_features"
            )
        calculate_reduced_features(
            data,
            dim=args.out_channels,
            reduction_method=reduction_method,
            fit_nodes=fit_mask,
            random_state=args.seed,
        )

    report("Generating comparable graph splits...")
    splits = generate_comparable_splits(
        data, query=query, train_ratio=args.train_ratio, seed=args.seed
    )
    train = splits.train.to(device)
    test_transductive = splits.test_transductive.to(device)
    test_inductive = splits.test_inductive.to(device)

    logger = CSVResultLogger(args.out_filepath)
    decoder_hidden_channels = getattr(args, "decoder_hidden_channels", None)
    decoder_dropout = getattr(args, "decoder_dropout", 0.0)
    kl_weight = getattr(args, "kl_weight", 1.0)
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
    model_progress = tqdm(
        experiment_configs,
        desc="Architectures",
        unit="model",
        disable=not verbose,
        position=0,
    )
    for model_type, regularization_type in model_progress:
        progress_label = model_type
        if model_type == "asymmetric":
            progress_label += f"/{regularization_type}"
        model_progress.set_postfix_str(progress_label)
        torch.manual_seed(args.seed)
        predictor_kwargs = dict(
            model_type=model_type,
            conv_layer_type=args.conv_layer_type,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            decoder_hidden_channels=decoder_hidden_channels,
            decoder_dropout=decoder_dropout,
            kl_weight=kl_weight,
            learning_rate=learning_rate,
            weight_decay=(weight_decay if model_type == "asymmetric" else 0.0),
            **deal_config,
        )
        if model_type == "asymmetric":
            predictor_kwargs.update({
                "regularization_type": regularization_type,
                "regularization_strength": regularization_strength,
            })
        predictor = LinkPrediction(train, **predictor_kwargs)
        predictor.model = predictor.model.to(device)
        last_loss = None
        training_time_seconds = None
        if predictor.trainable:
            report(f"Training {progress_label} model...")
            epoch_progress = tqdm(
                range(args.epochs),
                desc=f"{progress_label} epochs",
                unit="epoch",
                disable=not verbose,
                leave=False,
                position=1,
            )
            _synchronize_device(device)
            training_start = time.perf_counter()
            for epoch in epoch_progress:
                negative_edges = sample_training_negative_edges(
                    train, seed=args.seed + epoch
                )
                last_loss = predictor.train(
                    negative_edge_index=negative_edges
                )
                if verbose:
                    component_values = {
                        name: f"{value.detach().item():.4f}"
                        for name, value in predictor.last_loss_components.items()
                    }
                    component_values["loss"] = f"{last_loss.detach().item():.4f}"
                    epoch_progress.set_postfix(component_values)
            _synchronize_device(device)
            training_time_seconds = time.perf_counter() - training_start
        else:
            report(f"Evaluating {model_type} baseline without training...")

        report(f"Evaluating {progress_label} model...")
        for split_type, test_data in (
            ("transductive", test_transductive),
            ("inductive", test_inductive),
        ):
            probabilities = predictor.test(test_data)
            metrics = calculate_metrics(test_data.edge_label, probabilities)
            extra_fields = {"reduction_method": reduction_method}
            if model_type == "mlp":
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
            result = ExperimentResult(
                dataset=data.dataset_name,
                model_type=model_type,
                split_type=split_type,
                conv_layer_type=(
                    args.conv_layer_type if predictor.model_spec.uses_conv_layer
                    else "None"
                ),
                hidden_channels=args.hidden_channels,
                out_channels=args.out_channels,
                query_size=len(query),
                train_ratio=args.train_ratio,
                epochs=args.epochs,
                random_seed=args.seed,
                training_time_seconds=training_time_seconds,
                regularization_type=regularization_type,
                regularization_strength=(
                    regularization_strength if model_type == "asymmetric" else 0.0
                ),
                weight_decay=(weight_decay if model_type == "asymmetric" else 0.0),
                loss=(
                    float(last_loss.detach().cpu()) if last_loss is not None else None
                ),
                extra_fields=extra_fields,
                **metrics,
            )
            logger.log(result)
    report(f"Results written to {args.out_filepath}")
    return logger.run_id


if __name__ == "__main__":
    run_id = run_experiment(parse_args())
    print(f"Completed experiment run {run_id}")
