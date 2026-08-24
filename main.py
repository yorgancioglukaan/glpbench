import torch
from datasets import load_dataset
import argparse
import random

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
from link_prediction import LinkPrediction, available_model_types
from utils import calculate_reduced_features, generate_comparable_splits


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
    parser.add_argument("--query-size", type=int, default=10)
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
    if args.query_size <= 0:
        raise ValueError("query_size must be a positive integer")
    if args.epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if not 0 < args.train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")

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

    # Load and prepare the source graph on CPU for splitting.
    report(f"Loading {args.dataset}...")
    data = load_dataset(
        args.dataset,
        root=args.data_root,
        normalize_features=args.normalize_features,
        device="cpu",
    )
    # Select seeded, non-isolated query nodes.
    candidate_queries = torch.unique(data.edge_index).tolist()
    if args.query_size > len(candidate_queries):
        raise ValueError("query_size exceeds the number of non-isolated nodes")
    query = random.sample(candidate_queries, args.query_size)

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
    train, _, test_transductive, test_inductive = generate_comparable_splits(
        data, query=query, train_ratio=args.train_ratio
    )
    train = train.to(device)
    test_transductive = test_transductive.to(device)
    test_inductive = test_inductive.to(device)

    logger = CSVResultLogger(args.out_filepath)
    decoder_hidden_channels = getattr(args, "decoder_hidden_channels", None)
    decoder_dropout = getattr(args, "decoder_dropout", 0.0)
    kl_weight = getattr(args, "kl_weight", 1.0)
    model_progress = tqdm(
        model_types,
        desc="Architectures",
        unit="model",
        disable=not verbose,
        position=0,
    )
    for model_type in model_progress:
        model_progress.set_postfix_str(model_type)
        torch.manual_seed(args.seed)
        predictor = LinkPrediction(
            train,
            model_type=model_type,
            conv_layer_type=args.conv_layer_type,
            hidden_channels=args.hidden_channels,
            out_channels=args.out_channels,
            decoder_hidden_channels=decoder_hidden_channels,
            decoder_dropout=decoder_dropout,
            kl_weight=kl_weight,
        )
        predictor.model = predictor.model.to(device)
        last_loss = None
        if predictor.trainable:
            report(f"Training {model_type} model...")
            epoch_progress = tqdm(
                range(args.epochs),
                desc=f"{model_type} epochs",
                unit="epoch",
                disable=not verbose,
                leave=False,
                position=1,
            )
            for _ in epoch_progress:
                last_loss = predictor.train()
                if verbose:
                    epoch_progress.set_postfix(
                        loss=f"{last_loss.detach().item():.4f}",
                        reconstruction=(
                            f"{predictor.last_reconstruction_loss.detach().item():.4f}"
                        ),
                        regularization=(
                            f"{predictor.last_regularization_loss.detach().item():.4f}"
                        ),
                    )
        else:
            report(f"Evaluating {model_type} baseline without training...")

        report(f"Evaluating {model_type} model...")
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
                query_size=args.query_size,
                train_ratio=args.train_ratio,
                epochs=args.epochs,
                random_seed=args.seed,
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
