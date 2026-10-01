"""Run validation-only convergence audits for selected model configurations."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import random
from uuid import uuid4

import torch
from tqdm.auto import tqdm
import yaml

import benchmark
from datasets import load_dataset
from experiment_logging import CSVEpochHistoryLogger, EpochHistoryRecord
from experiment_pipeline import prepare_validation_data
from link_prediction import LinkPrediction, count_trainable_parameters, get_model_spec
from main import calculate_metrics, train_with_validation_selection
from utils import generate_validation_splits


AUDIT_FIELDNAMES = [
    "run_id", "configuration_id", "timestamp_utc", "dataset", "model_type",
    "conv_layer_type", "reduction_method", "regularization_type",
    "regularization_strength", "random_seed", "query_size",
    "validation_query_size", "train_ratio", "epochs", "epochs_trained",
    "best_epoch", "selected_epoch", "selection_metric",
    "best_validation_score", "validation_roc_auc",
    "validation_average_precision", "training_time_seconds",
    "trainable_parameter_count", "reducer_fit_time_seconds",
    "learning_rate", "weight_decay", "lr_scheduler",
    "lr_scheduler_kwargs", "final_learning_rate", "normalize_features",
]


@dataclass(frozen=True)
class AuditConfig:
    """Describe a validation-only convergence audit."""

    benchmark_config: benchmark.BenchmarkConfig
    epoch_history_output: str
    failures_output: str
    fail_fast: bool

    @property
    def results_output(self):
        """Return the audit-summary CSV path."""
        return self.benchmark_config.output


@dataclass(frozen=True)
class AuditSummary:
    """Summarize one audit invocation."""

    run_id: str
    runs_total: int
    runs_succeeded: int
    runs_failed: int
    runs_skipped: int


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _unique_list(value, name):
    if not isinstance(value, list) or not value or len(set(value)) != len(value):
        raise ValueError(f"{name} must be a non-empty unique list")
    return value


def load_audit_config(path):
    """Load and validate a convergence-audit YAML specification."""
    with Path(path).open(encoding="utf-8") as stream:
        raw = _mapping(yaml.safe_load(stream), "audit specification")
    if raw.get("version") != 1 or raw.get("kind") != (
        "validation-convergence-audit"
    ):
        raise ValueError(
            "expected a version-1 validation-convergence-audit specification"
        )
    unknown = set(raw).difference({
        "version", "kind", "artifacts", "execution", "seeds",
        "selected_configurations", "datasets", "models", "splits", "training",
    })
    if unknown:
        raise ValueError(f"Unknown audit setting: {sorted(unknown)[0]}")

    artifacts = _mapping(raw.get("artifacts"), "artifacts")
    if set(artifacts) != {"results", "epoch_history", "failures"}:
        raise ValueError(
            "artifacts must define results, epoch_history, and failures"
        )
    if any(
        not isinstance(value, str) or not value.strip()
        for value in artifacts.values()
    ):
        raise ValueError("artifact paths must be non-empty strings")

    execution = _mapping(raw.get("execution"), "execution")
    if set(execution) != {"device", "data_root", "fail_fast"}:
        raise ValueError("execution must define device, data_root, and fail_fast")
    if not isinstance(execution["device"], str) or not execution["device"]:
        raise ValueError("execution.device must be a non-empty string")
    if not isinstance(execution["data_root"], str) or not execution["data_root"]:
        raise ValueError("execution.data_root must be a non-empty string")
    if not isinstance(execution["fail_fast"], bool):
        raise ValueError("execution.fail_fast must be boolean")

    seeds = _unique_list(raw.get("seeds"), "seeds")
    if any(
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0
        for seed in seeds
    ):
        raise ValueError("seeds must be non-negative integers")
    datasets = tuple(
        benchmark._resolve_dataset_name(name)
        for name in _unique_list(raw.get("datasets"), "datasets")
    )
    model_names = _unique_list(raw.get("models"), "models")
    if any(not get_model_spec(name).trainable for name in model_names):
        raise ValueError("convergence audits require trainable models")

    split = _mapping(raw.get("splits"), "splits")
    if set(split) != {
        "protocol", "query_fraction", "validation_query_fraction", "train_ratio"
    }:
        raise ValueError("splits must define the validation-only protocol")
    if split["protocol"] != "matched-semi-inductive-validation-only":
        raise ValueError("unsupported audit split protocol")

    training = _mapping(raw.get("training"), "training")
    if set(training) != {
        "max_epochs", "selection_metric", "checkpoint", "early_stopping",
        "scheduler",
    }:
        raise ValueError("training must define the convergence-audit policy")
    if training["checkpoint"] != "best-validation":
        raise ValueError("audit checkpoint must be best-validation")
    stopping = _mapping(training["early_stopping"], "early_stopping")
    if stopping != {"enabled": False}:
        raise ValueError("audit early stopping must be disabled")
    scheduler = _mapping(training["scheduler"], "scheduler")
    if scheduler != {"type": "none"}:
        raise ValueError("audit scheduler must be disabled")

    defaults = benchmark._validate_defaults({
        "query_size": None,
        "query_fraction": split["query_fraction"],
        "validation_query_size": None,
        "validation_query_fraction": split["validation_query_fraction"],
        "train_ratio": split["train_ratio"],
        "epochs": training["max_epochs"],
        "selection_metric": training["selection_metric"],
        "checkpoint_strategy": "best-validation",
        "training_policy": "convergence-audit",
        "early_stopping_patience": None,
        "lr_scheduler": "none",
        "lr_scheduler_kwargs": None,
        "device": execution["device"],
    })
    models = tuple(
        benchmark._parse_model_grid({"type": name}, defaults, index)
        for index, name in enumerate(model_names)
    )
    selected_path = benchmark._selected_configuration_path(
        raw.get("selected_configurations")
    )
    base = benchmark.BenchmarkConfig(
        output=artifacts["results"],
        data_root=execution["data_root"],
        seeds=tuple(seeds),
        defaults=defaults,
        datasets=datasets,
        models=models,
        error_output=artifacts["failures"],
        selected_configurations=selected_path,
        kind="validation-convergence-audit",
    )
    return AuditConfig(
        benchmark_config=base,
        epoch_history_output=artifacts["epoch_history"],
        failures_output=artifacts["failures"],
        fail_fast=execution["fail_fast"],
    )


def expand_audit_config(config):
    """Resolve selected hyperparameters into audit jobs."""
    if not isinstance(config, AuditConfig):
        raise TypeError("config must be an AuditConfig")
    return benchmark.expand_benchmark_config(config.benchmark_config)


def _predictor_kwargs(job):
    parameters = job.parameters
    kwargs = {
        "model_type": job.model_type,
        "conv_layer_type": job.conv_layer_type,
        "hidden_channels": parameters["hidden_channels"],
        "out_channels": parameters["out_channels"],
        "decoder_hidden_channels": parameters["decoder_hidden_channels"],
        "decoder_dropout": parameters["decoder_dropout"],
        "kl_weight": parameters["kl_weight"],
        "learning_rate": parameters["learning_rate"],
        "weight_decay": parameters["weight_decay"],
        "gatv2_kwargs": parameters["gatv2_kwargs"] or {},
        "theta_ss": parameters["deal_theta_ss"],
        "theta_aa": parameters["deal_theta_aa"],
        "theta_sa": parameters["deal_theta_sa"],
        "lambda_ss": parameters["deal_lambda_ss"],
        "lambda_aa": parameters["deal_lambda_aa"],
        "lambda_sa": parameters["deal_lambda_sa"],
        "similarity_scale": parameters["deal_similarity_scale"],
        "similarity_margin": parameters["deal_similarity_margin"],
    }
    if get_model_spec(job.model_type).supports_agae_regularization:
        kwargs.update({
            "regularization_type": job.regularization.type,
            "regularization_strength": job.regularization.strength,
        })
    return kwargs


def _load_completed(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != AUDIT_FIELDNAMES:
            raise ValueError("existing audit CSV schema does not match")
        return {
            (row["configuration_id"], int(row["random_seed"]))
            for row in reader
        }


def _append_row(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    if exists:
        with path.open(newline="", encoding="utf-8") as stream:
            if next(csv.reader(stream), None) != AUDIT_FIELDNAMES:
                raise ValueError("existing audit CSV schema does not match")
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=AUDIT_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_failure(path, run_id, job, seed, error):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": job.dataset,
        "model_type": job.model_type,
        "random_seed": seed,
        "training_policy": "convergence-audit",
        "error_type": type(error).__name__,
        "message": str(error),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def execute_audit(config, jobs, *, dry_run=False, resume=False, verbose=False,
                  run_id=None):
    """Execute selected validation-only audit jobs."""
    jobs = tuple(jobs)
    audit_run_id = run_id or str(uuid4())
    total = len(jobs) * len(config.benchmark_config.seeds)
    if dry_run:
        return AuditSummary(audit_run_id, total, 0, 0, 0)

    completed = _load_completed(config.results_output) if resume else set()
    epoch_logger = CSVEpochHistoryLogger(
        config.epoch_history_output, run_id=audit_run_id
    )
    completed_epochs = (
        benchmark.load_completed_epoch_history(config.epoch_history_output)
        if resume else set()
    )
    source_cache = {}
    split_cache = {}
    succeeded = failed = skipped = 0
    progress = tqdm(total=total, desc="Convergence audit", unit="run",
                    disable=not verbose)

    for job in jobs:
        configuration_id = benchmark._training_configuration_id(job)
        for seed in job.seeds:
            progress.set_postfix_str(f"{job.dataset}/{job.model_type}/{seed}")
            if (configuration_id, seed) in completed:
                skipped += 1
                progress.update(1)
                continue
            try:
                if job.dataset not in source_cache:
                    source_cache[job.dataset] = load_dataset(
                        job.dataset,
                        root=config.benchmark_config.data_root,
                        normalize_features=False,
                        device="cpu",
                    )
                source = source_cache[job.dataset]
                split_key = (job.dataset, seed)
                if split_key not in split_cache:
                    split_cache[split_key] = generate_validation_splits(
                        source,
                        query_fraction=job.parameters["query_fraction"],
                        validation_query_fraction=(
                            job.parameters["validation_query_fraction"]
                        ),
                        train_ratio=job.parameters["train_ratio"],
                        seed=seed,
                    )
                prepared = prepare_validation_data(
                    source,
                    split_cache[split_key],
                    model_type=job.model_type,
                    normalize_features=job.parameters["normalize_features"],
                    out_channels=job.parameters["out_channels"],
                    reduction_method=job.reduction_method,
                    random_state=seed,
                )
                device = torch.device(job.parameters["device"])
                train = prepared.train.to(device)
                validation = prepared.validation.to(device)
                random.seed(seed)
                torch.manual_seed(seed)
                predictor = LinkPrediction(train, **_predictor_kwargs(job))
                predictor.model = predictor.model.to(device)
                parameter_count = count_trainable_parameters(predictor.model)

                def record_epoch(record):
                    key = (configuration_id, seed, record["epoch"])
                    if key in completed_epochs:
                        return
                    epoch_logger.log(EpochHistoryRecord(
                        configuration_id=configuration_id,
                        dataset=job.dataset,
                        model_type=job.model_type,
                        random_seed=seed,
                        training_policy="convergence-audit",
                        epoch=record["epoch"],
                        training_loss=record["training_loss"],
                        learning_rate=record["learning_rate"],
                        validation_roc_auc=record["validation_roc_auc"],
                        validation_average_precision=(
                            record["validation_average_precision"]
                        ),
                    ))
                    completed_epochs.add(key)

                summary = train_with_validation_selection(
                    predictor,
                    train,
                    validation,
                    epochs=job.parameters["epochs"],
                    seed=seed,
                    selection_metric=job.parameters["selection_metric"],
                    device=device,
                    checkpoint_strategy="best-validation",
                    lr_scheduler="none",
                    epoch_callback=record_epoch,
                    verbose=verbose,
                    progress_label=f"{job.dataset}/{job.model_type}/{seed}",
                    progress_position=1,
                    compact_progress=True,
                )
                metrics = calculate_metrics(
                    validation.edge_label, predictor.test(validation)
                )
                _append_row(config.results_output, {
                    "run_id": audit_run_id,
                    "configuration_id": configuration_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset": job.dataset,
                    "model_type": job.model_type,
                    "conv_layer_type": job.conv_layer_type,
                    "reduction_method": job.reduction_method,
                    "regularization_type": job.regularization.type,
                    "regularization_strength": job.regularization.strength,
                    "random_seed": seed,
                    "query_size": int(train.test_query_mask.sum()),
                    "validation_query_size": int(
                        train.validation_query_mask.sum()
                    ),
                    "train_ratio": job.parameters["train_ratio"],
                    "epochs": job.parameters["epochs"],
                    "epochs_trained": summary.epochs_trained,
                    "best_epoch": summary.best_epoch,
                    "selected_epoch": summary.selected_epoch,
                    "selection_metric": summary.selection_metric,
                    "best_validation_score": summary.best_validation_score,
                    "validation_roc_auc": metrics["roc_auc"],
                    "validation_average_precision": metrics["average_precision"],
                    "training_time_seconds": summary.training_time_seconds,
                    "trainable_parameter_count": parameter_count,
                    "reducer_fit_time_seconds": (
                        prepared.reducer_fit_time_seconds
                        if prepared.reducer_fit_time_seconds is not None else ""
                    ),
                    "learning_rate": job.parameters["learning_rate"],
                    "weight_decay": job.parameters["weight_decay"],
                    "lr_scheduler": summary.lr_scheduler,
                    "lr_scheduler_kwargs": json.dumps(
                        summary.lr_scheduler_kwargs, sort_keys=True
                    ),
                    "final_learning_rate": summary.final_learning_rate,
                    "normalize_features": job.parameters["normalize_features"],
                })
                completed.add((configuration_id, seed))
                succeeded += 1
            except Exception as error:
                failed += 1
                _append_failure(
                    config.failures_output, audit_run_id, job, seed, error
                )
                if config.fail_fast:
                    progress.close()
                    raise
            finally:
                progress.update(1)
    progress.close()
    return AuditSummary(audit_run_id, total, succeeded, failed, skipped)


def parse_args():
    """Parse convergence-audit CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to an audit YAML file")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main(cli_args=None):
    """Load, expand, and execute a convergence audit."""
    args = parse_args() if cli_args is None else cli_args
    config = load_audit_config(args.config)
    jobs = expand_audit_config(config)
    total = len(jobs) * len(config.benchmark_config.seeds)
    print(json.dumps({
        "datasets": len(config.benchmark_config.datasets),
        "architectures": len(config.benchmark_config.models),
        "jobs": len(jobs),
        "seeds_per_job": len(config.benchmark_config.seeds),
        "training_runs": total,
        "maximum_epochs": total * config.benchmark_config.defaults["epochs"],
        "test_splits_evaluated": 0,
    }, indent=2, sort_keys=True))
    if args.dry_run:
        for index, job in enumerate(jobs, start=1):
            print(f"{index:4d}. {job.label}")
    summary = execute_audit(
        config,
        jobs,
        dry_run=args.dry_run,
        resume=args.resume,
        verbose=args.verbose,
    )
    print("Convergence audit complete")
    print(f"  Run ID: {summary.run_id}")
    print(
        f"  Runs: {summary.runs_succeeded} succeeded, "
        f"{summary.runs_failed} failed, {summary.runs_skipped} skipped, "
        f"{summary.runs_total} total"
    )
    print(f"  Results: {config.results_output}")
    print(f"  Epoch history: {config.epoch_history_output}")
    return summary


if __name__ == "__main__":
    main()
