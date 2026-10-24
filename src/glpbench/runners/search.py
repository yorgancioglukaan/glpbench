"""Run validation-only hyperparameter searches from a YAML specification."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from statistics import mean
from uuid import uuid4

import torch
from tqdm.auto import tqdm
import yaml

from glpbench.datasets import available_datasets
from glpbench.contracts import TrainingPolicy
from glpbench.exceptions import ConfigurationError
from glpbench.evaluation import (
    load_registered_dataset,
    prepare_validation_data,
    realize_registered_split,
    validate_registered_compatibility,
)
from glpbench.preprocessing import available_reduction_preprocessors
from glpbench.models import (
    LinkPrediction,
    available_model_types,
    count_trainable_parameters,
    get_model_adapter,
    resolve_model_parameters,
)
from glpbench.training import validate_scheduler_config
from glpbench.runners.experiment import (
    calculate_metrics,
    train_with_validation_selection,
)


_TRAINABLE_MODELS = tuple(
    model_type for model_type in available_model_types()
    if get_model_adapter(model_type).capabilities.trainable
)
_HISTORY_FIELDS = [
    "search_run_id", "candidate_id", "dataset", "architecture",
    "tuning_seed", "status", "hyperparameters", "validation_roc_auc",
    "validation_average_precision", "best_epoch", "epochs_trained",
    "stopped_early", "training_time_seconds", "reducer_fit_time_seconds",
    "trainable_parameter_count", "timestamp_utc",
]
_AGGREGATE_FIELDS = [
    "candidate_id", "dataset", "architecture", "num_seeds",
    "mean_validation_roc_auc", "mean_validation_average_precision",
    "mean_trainable_parameter_count", "selected", "hyperparameters",
]


@dataclass(frozen=True)
class SearchCandidate:
    """Store one resolved dataset–architecture configuration."""

    candidate_id: str
    dataset: str
    architecture: str
    hyperparameters: dict


@dataclass(frozen=True)
class SearchConfig:
    """Store a validated pilot-search specification."""

    path: Path
    raw: dict
    datasets: tuple[str, ...]
    architectures: tuple[str, ...]
    tuning_seeds: tuple[int, ...]
    num_candidates: int
    candidate_seed: int
    history_path: Path
    failures_path: Path
    selected_path: Path
    aggregate_path: Path


@dataclass(frozen=True)
class SearchSummary:
    """Summarize one search-runner invocation."""

    run_id: str
    total_runs: int
    succeeded: int
    failed: int
    skipped: int
    selected: int


@dataclass(frozen=True)
class _PreparationFailure:
    """Store a shared dataset or split preparation failure."""

    stage: str
    error: Exception


def _mapping(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a mapping")
    return value


def _values(space, name):
    space = _mapping(space, name)
    values = space.get("values")
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name}.values must be a non-empty list")
    if len({json.dumps(value, sort_keys=True) for value in values}) != len(values):
        raise ValueError(f"{name}.values must not contain duplicates")
    if space.get("sampling") not in {"random", "balanced", "stratified"}:
        raise ValueError(f"{name}.sampling is not supported")
    return values


def load_search_config(path):
    """Load and validate a version-1 hyperparameter-search specification."""
    path = Path(path)
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    raw = _mapping(raw, "search specification")
    if raw.get("version") != 1 or raw.get("kind") != "hyperparameter-search":
        raise ConfigurationError(
            "expected a version-1 hyperparameter-search specification",
            component="search-config",
        )

    artifacts = _mapping(raw.get("artifacts"), "artifacts")
    required_artifacts = {"history", "failures", "selected_configurations"}
    if not required_artifacts.issubset(artifacts):
        raise ValueError("artifacts must define history, failures, and selections")
    for name in required_artifacts:
        if not isinstance(artifacts[name], str) or not artifacts[name]:
            raise ValueError(f"artifacts.{name} must be a non-empty path")

    execution = _mapping(raw.get("execution"), "execution")
    if not isinstance(execution.get("device"), str) or not execution["device"]:
        raise ValueError("execution.device must be a non-empty string")
    if not isinstance(execution.get("data_root"), str) or not execution["data_root"]:
        raise ValueError("execution.data_root must be a non-empty string")
    if not isinstance(execution.get("fail_fast"), bool):
        raise ValueError("execution.fail_fast must be boolean")

    datasets = raw.get("datasets")
    if not isinstance(datasets, list) or not datasets or len(set(datasets)) != len(datasets):
        raise ValueError("datasets must be a non-empty unique list")
    supported_datasets = set(available_datasets())
    if any(dataset not in supported_datasets for dataset in datasets):
        raise ValueError("search specification contains an unknown dataset")

    search = _mapping(raw.get("search"), "search")
    if search.get("strategy") != "seeded-random":
        raise ValueError("only seeded-random search is supported")
    num_candidates = search.get("num_candidates")
    candidate_seed = search.get("candidate_seed")
    tuning_seeds = search.get("tuning_seeds")
    if not isinstance(num_candidates, int) or num_candidates <= 0:
        raise ValueError("num_candidates must be a positive integer")
    if not isinstance(candidate_seed, int) or candidate_seed < 0:
        raise ValueError("candidate_seed must be a non-negative integer")
    if (
        not isinstance(tuning_seeds, list) or not tuning_seeds
        or len(set(tuning_seeds)) != len(tuning_seeds)
        or any(not isinstance(seed, int) or seed < 0 for seed in tuning_seeds)
    ):
        raise ValueError("tuning_seeds must be unique non-negative integers")
    if search.get("include_default") is not True:
        raise ValueError("the pilot search must include its default candidate")
    if search.get("require_all_seeds_for_selection") is not True:
        raise ValueError("the pilot search requires every tuning seed")

    selection = _mapping(raw.get("selection"), "selection")
    if selection.get("granularity") != ["dataset", "architecture"]:
        raise ValueError("selection granularity must be dataset and architecture")
    if selection.get("validation_setting") != "semi-inductive":
        raise ValueError("selection must use semi-inductive validation")
    if selection.get("tuning_conv_layer") != "GCNConv":
        raise ValueError("the pilot tuning convolution must be GCNConv")
    if selection.get("metric") != "roc_auc" or selection.get("aggregation") != "mean":
        raise ValueError("selection must use mean validation ROC-AUC")
    tolerance = selection.get("tie_tolerance")
    if not isinstance(tolerance, (int, float)) or tolerance < 0:
        raise ValueError("tie_tolerance must be non-negative")
    if selection.get("tie_breakers") != [
        {"metric": "average_precision", "direction": "maximize"},
        {"metric": "trainable_parameter_count", "direction": "minimize"},
    ]:
        raise ValueError("unsupported tie-breaker policy")

    training = _mapping(raw.get("training"), "training")
    if not isinstance(training.get("max_epochs"), int) or training["max_epochs"] <= 0:
        raise ValueError("training.max_epochs must be positive")
    stopping = _mapping(training.get("early_stopping"), "early_stopping")
    if stopping.get("enabled") is not True:
        raise ValueError("early stopping must be enabled")
    if not isinstance(stopping.get("patience"), int) or stopping["patience"] <= 0:
        raise ValueError("early-stopping patience must be positive")
    if not isinstance(stopping.get("min_delta"), (int, float)) or stopping["min_delta"] < 0:
        raise ValueError("early-stopping min_delta must be non-negative")
    scheduler = _mapping(
        training.get("scheduler", {"type": "none"}), "training.scheduler"
    )
    scheduler_type = scheduler.get("type")
    scheduler_parameters = validate_scheduler_config(
        scheduler_type,
        {name: value for name, value in scheduler.items() if name != "type"},
        epochs=training["max_epochs"],
    )
    training["scheduler"] = {
        "type": scheduler_type,
        **scheduler_parameters,
    }

    splits = _mapping(raw.get("splits"), "splits")
    if splits.get("protocol") != "matched-semi-inductive":
        raise ValueError("unsupported split protocol")
    if not 0 < splits.get("train_ratio", 0) < 1:
        raise ValueError("split train_ratio must be between zero and one")
    for name in ("validation_queries", "test_queries"):
        fraction = _mapping(splits.get(name), name).get("fraction")
        if not isinstance(fraction, (int, float)) or not 0 < fraction < 0.5:
            raise ValueError(f"{name}.fraction must be between zero and 0.5")
    negative = _mapping(splits.get("negative_sampling"), "negative_sampling")
    if negative.get("method") != "uniform" or negative.get("ratio") != 1.0:
        raise ValueError("the pilot requires uniform 1:1 negative sampling")
    if negative.get("exclude_original_positives") is not True:
        raise ValueError("negative sampling must exclude original positives")
    if negative.get("paired_across_candidates") is not True:
        raise ValueError("negative sampling must be paired across candidates")
    if _mapping(splits.get("audit"), "audit").get("enabled") is not True:
        raise ValueError("split auditing must be enabled")

    common = _mapping(raw.get("common_space"), "common_space")
    expected_common = {
        "hidden_channels", "out_channels", "learning_rate", "weight_decay"
    }
    if set(common) != expected_common:
        raise ValueError("common_space has unexpected dimensions")
    for name in expected_common:
        values = _values(common[name], f"common_space.{name}")
        if any(not isinstance(value, (int, float)) or value <= 0 for value in values) and name != "weight_decay":
            raise ValueError(f"common_space.{name} values must be positive")
    if any(value < 0 for value in common["weight_decay"]["values"]):
        raise ValueError("weight_decay values must be non-negative")
    if common["weight_decay"].get("applies_to") != "all-trainable-models":
        raise ValueError("weight decay must apply to every trainable model")

    normalization = _mapping(raw.get("preprocessing"), "preprocessing").get(
        "normalize_features"
    )
    if _values(normalization, "normalize_features") != [True, False]:
        raise ValueError("normalization values must be true and false")

    models = _mapping(raw.get("models"), "models")
    if set(models) != set(_TRAINABLE_MODELS):
        raise ValueError("models must define every trainable architecture")
    _validate_model_spaces(models, num_candidates)

    history_path = Path(artifacts["history"])
    return SearchConfig(
        path=path,
        raw=raw,
        datasets=tuple(datasets),
        architectures=tuple(models),
        tuning_seeds=tuple(tuning_seeds),
        num_candidates=num_candidates,
        candidate_seed=candidate_seed,
        history_path=history_path,
        failures_path=Path(artifacts["failures"]),
        selected_path=Path(artifacts["selected_configurations"]),
        aggregate_path=history_path.with_name(
            history_path.stem + "_aggregates.csv"
        ),
    )


def _validate_model_spaces(models, num_candidates):
    if models["gae"] != {}:
        raise ValueError("GAE model space must be empty")
    for model_type in ("agae", "agae-mlp"):
        model = _mapping(models[model_type], model_type)
        reducers = _values(model.get("reduction_method"), "reduction_method")
        unknown_reducers = set(reducers).difference(
            available_reduction_preprocessors()
        )
        if unknown_reducers:
            raise ValueError(
                f"unknown asymmetric reducer: {sorted(unknown_reducers)[0]}"
            )
        regularization = _mapping(model.get("regularization"), "regularization")
        if regularization.get("type") != "cosine-alignment":
            raise ValueError("asymmetric regularization must use cosine alignment")
        strengths = _values(regularization.get("strength"), "strength")
        if strengths != [0.0, 0.001, 0.01]:
            raise ValueError("unexpected cosine-alignment strengths")
        if len(reducers) != num_candidates or len(strengths) != num_candidates:
            raise ValueError("stratified spaces must match num_candidates")
    for model_type in ("gae-mlp", "feature-mlp", "agae-mlp"):
        model = models[model_type]
        _values(model.get("decoder_hidden_channels"), "decoder_hidden_channels")
        dropouts = _values(model.get("decoder_dropout"), "decoder_dropout")
        if any(not 0 <= value < 1 for value in dropouts):
            raise ValueError("decoder dropout must be in [0, 1)")
    kl_weights = _values(models["vgae"].get("kl_weight"), "kl_weight")
    if len(kl_weights) != num_candidates or any(value < 0 for value in kl_weights):
        raise ValueError("VGAE KL values must cover every candidate")
    deal = _mapping(models["deal"], "deal")
    if deal.get("similarity_scale") != 1.0 or deal.get("similarity_margin") != 0.0:
        raise ValueError("unexpected fixed DEAL similarity settings")
    profiles = _mapping(deal.get("profiles"), "DEAL profiles")
    if set(profiles) != {"balanced", "attribute-focused", "alignment-focused"}:
        raise ValueError("unexpected DEAL profile names")
    required = {
        "theta_ss", "theta_aa", "theta_sa", "lambda_ss", "lambda_aa",
        "lambda_sa",
    }
    for name, profile in profiles.items():
        if set(_mapping(profile, name)) != required:
            raise ValueError(f"DEAL profile {name} has incomplete coefficients")
        if any(not isinstance(value, (int, float)) or value < 0 for value in profile.values()):
            raise ValueError(f"DEAL profile {name} contains an invalid coefficient")
    if deal.get("sampling") != "stratified" or len(profiles) != num_candidates:
        raise ValueError("DEAL profiles must be stratified across candidates")


def _stable_rng(seed, dataset, architecture):
    digest = hashlib.sha256(
        f"{seed}:{dataset}:{architecture}".encode("utf-8")
    ).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def _stratified(values, default, rng, count):
    remaining = [value for value in values if value != default]
    rng.shuffle(remaining)
    sequence = [default, *remaining]
    if len(sequence) != count:
        raise ValueError("stratified values must match the candidate count")
    return sequence


def generate_candidates(config):
    """Return deterministic, fully resolved search candidates."""
    candidates = []
    common = config.raw["common_space"]
    models = config.raw["models"]
    defaults = {
        "hidden_channels": 128,
        "out_channels": 64,
        "learning_rate": 0.01,
        "weight_decay": 0.0,
    }
    for dataset in config.datasets:
        for architecture in config.architectures:
            rng = _stable_rng(config.candidate_seed, dataset, architecture)
            normalization = [True, False, rng.choice((True, False))]
            model_space = models[architecture]
            reducers = strengths = kl_values = profile_names = None
            if architecture in {"agae", "agae-mlp"}:
                reducers = _stratified(
                    model_space["reduction_method"]["values"], "pca", rng,
                    config.num_candidates,
                )
                strengths = _stratified(
                    model_space["regularization"]["strength"]["values"],
                    0.0, rng, config.num_candidates,
                )
            if architecture == "vgae":
                kl_values = _stratified(
                    model_space["kl_weight"]["values"], 1.0, rng,
                    config.num_candidates,
                )
            if architecture == "deal":
                profile_names = _stratified(
                    list(model_space["profiles"]), "balanced", rng,
                    config.num_candidates,
                )
            resolved = []
            for index in range(config.num_candidates):
                parameters = {
                    name: (
                        default if index == 0
                        else rng.choice(common[name]["values"])
                    )
                    for name, default in defaults.items()
                }
                parameters["normalize_features"] = normalization[index]
                parameters["conv_layer_type"] = "GCNConv"
                if architecture in {"agae", "agae-mlp"}:
                    parameters.update({
                        "reduction_method": reducers[index],
                        "regularization_type": "cosine-alignment",
                        "regularization_strength": strengths[index],
                    })
                if architecture in {"gae-mlp", "feature-mlp", "agae-mlp"}:
                    parameters.update({
                        "decoder_hidden_channels": (
                            64 if index == 0 else rng.choice(
                                model_space["decoder_hidden_channels"]["values"]
                            )
                        ),
                        "decoder_dropout": (
                            0.0 if index == 0 else rng.choice(
                                model_space["decoder_dropout"]["values"]
                            )
                        ),
                    })
                if architecture == "vgae":
                    parameters["kl_weight"] = kl_values[index]
                if architecture == "deal":
                    profile_name = profile_names[index]
                    parameters.update(model_space["profiles"][profile_name])
                    parameters.update({
                        "deal_profile": profile_name,
                        "similarity_scale": model_space["similarity_scale"],
                        "similarity_margin": model_space["similarity_margin"],
                    })
                resolved.append(parameters)

            serialized = [
                json.dumps(parameters, sort_keys=True, separators=(",", ":"))
                for parameters in resolved
            ]
            if len(set(serialized)) != len(serialized):
                raise ValueError(
                    f"candidate generation produced duplicates for {dataset}/{architecture}"
                )
            for parameters, canonical in zip(resolved, serialized):
                candidate_id = hashlib.sha256(
                    f"{dataset}:{architecture}:{canonical}".encode("utf-8")
                ).hexdigest()
                candidates.append(SearchCandidate(
                    candidate_id, dataset, architecture, parameters
                ))
    return tuple(candidates)


def _read_history(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != _HISTORY_FIELDS:
            raise ValueError("existing search-history schema does not match")
        return list(reader)


def _append_history(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_HISTORY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _append_failure(path, run_id, candidate, seed, error):
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "search_run_id": run_id,
        "candidate_id": candidate.candidate_id,
        "dataset": candidate.dataset,
        "architecture": candidate.architecture,
        "tuning_seed": seed,
        "hyperparameters": candidate.hyperparameters,
        "error_type": type(error).__name__,
        "message": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _append_preparation_failure(
    path, run_id, dataset, seed, stage, error, affected_runs
):
    """Log one failure shared by candidate runs for a dataset and seed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "search_run_id": run_id,
        "scope": "dataset-seed",
        "stage": stage,
        "dataset": dataset,
        "tuning_seed": seed,
        "affected_runs": affected_runs,
        "error_type": type(error).__name__,
        "message": str(error),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def _prepare_search_split(
    config, dataset, seed, source_cache, split_cache
):
    """Load a dataset and cache either its split or a shared failure."""
    split_config = config.raw["splits"]
    protocol_parameters = {
        "query_fraction": split_config["test_queries"]["fraction"],
        "validation_query_fraction": (
            split_config["validation_queries"]["fraction"]
        ),
        "train_ratio": split_config["train_ratio"],
    }
    split_key = (
        dataset,
        seed,
        "matched-semi-inductive-validation-only",
        tuple(sorted(protocol_parameters.items())),
    )
    if split_key in split_cache:
        return split_cache[split_key]

    if dataset not in source_cache:
        try:
            source_cache[dataset] = load_registered_dataset(
                dataset,
                root=config.raw["execution"]["data_root"],
                normalize_features=False,
                device="cpu",
            )
        except Exception as error:
            source_cache[dataset] = _PreparationFailure(
                "dataset-loading", error
            )

    loaded = source_cache[dataset]
    if isinstance(loaded, _PreparationFailure):
        split_cache[split_key] = loaded
        return loaded
    source_data = loaded.data

    try:
        prepared = realize_registered_split(
            "matched-semi-inductive-validation-only",
            source_data,
            seed=seed,
            dataset_fingerprint=loaded.metadata.fingerprint,
            parameters=protocol_parameters,
        )
    except Exception as error:
        prepared = _PreparationFailure("data-splitting", error)
    split_cache[split_key] = prepared
    return prepared


def _predictor_kwargs(candidate):
    parameters = dict(candidate.hyperparameters)
    adapter = get_model_adapter(candidate.architecture)
    model_parameters = resolve_model_parameters(adapter, parameters)
    return {
        "model_type": candidate.architecture,
        "model_parameters": model_parameters,
        "learning_rate": parameters["learning_rate"],
        "weight_decay": parameters["weight_decay"],
    }


def aggregate_and_select(config, candidates, history_rows, run_id):
    """Aggregate complete candidates and write selection artifacts."""
    rows_by_key = {}
    for row in history_rows:
        if row["status"] == "success":
            rows_by_key.setdefault(
                (row["candidate_id"], int(row["tuning_seed"])), row
            )
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    aggregates = []
    selected_by_group = {}
    for candidate in candidates:
        rows = [
            rows_by_key.get((candidate.candidate_id, seed))
            for seed in config.tuning_seeds
        ]
        if any(row is None for row in rows):
            continue
        aggregate = {
            "candidate_id": candidate.candidate_id,
            "dataset": candidate.dataset,
            "architecture": candidate.architecture,
            "num_seeds": len(rows),
            "mean_validation_roc_auc": mean(
                float(row["validation_roc_auc"]) for row in rows
            ),
            "mean_validation_average_precision": mean(
                float(row["validation_average_precision"]) for row in rows
            ),
            "mean_trainable_parameter_count": mean(
                int(row["trainable_parameter_count"]) for row in rows
            ),
            "selected": False,
            "hyperparameters": json.dumps(
                candidate.hyperparameters, sort_keys=True, separators=(",", ":")
            ),
        }
        aggregates.append(aggregate)
        selected_by_group.setdefault(
            (candidate.dataset, candidate.architecture), []
        ).append(aggregate)

    tolerance = float(config.raw["selection"]["tie_tolerance"])
    winners = {}
    for group, group_rows in selected_by_group.items():
        best_roc = max(row["mean_validation_roc_auc"] for row in group_rows)
        eligible = [
            row for row in group_rows
            if best_roc - row["mean_validation_roc_auc"] <= tolerance
        ]
        best_ap = max(
            row["mean_validation_average_precision"] for row in eligible
        )
        eligible = [
            row for row in eligible
            if row["mean_validation_average_precision"] == best_ap
        ]
        winner = min(
            eligible,
            key=lambda row: (
                row["mean_trainable_parameter_count"], row["candidate_id"]
            ),
        )
        winner["selected"] = True
        winners[group] = winner

    config.aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    with config.aggregate_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_AGGREGATE_FIELDS)
        writer.writeheader()
        writer.writerows(aggregates)

    selections = {}
    for (dataset, architecture), winner in sorted(winners.items()):
        candidate = candidate_by_id[winner["candidate_id"]]
        selections.setdefault(dataset, {})[architecture] = {
            "candidate_id": candidate.candidate_id,
            "mean_validation_roc_auc": winner["mean_validation_roc_auc"],
            "mean_validation_average_precision": (
                winner["mean_validation_average_precision"]
            ),
            "mean_trainable_parameter_count": (
                winner["mean_trainable_parameter_count"]
            ),
            "hyperparameters": candidate.hyperparameters,
        }
    selected_document = {
        "version": 1,
        "kind": "selected-hyperparameters",
        "search_run_id": run_id,
        "source_specification": str(config.path),
        "selection_metric": "mean_validation_roc_auc",
        "tuning_seeds": list(config.tuning_seeds),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "selections": selections,
    }
    config.selected_path.parent.mkdir(parents=True, exist_ok=True)
    with config.selected_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(selected_document, stream, sort_keys=False)
    return winners


def execute_search(
    config, candidates, *, dry_run=False, resume=False, verbose=False,
    run_id=None,
):
    """Execute validation-only candidate runs and select complete candidates."""
    total_runs = len(candidates) * len(config.tuning_seeds)
    if dry_run:
        run_id = run_id or str(uuid4())
        return SearchSummary(run_id, total_runs, 0, 0, 0, 0)

    if (
        not resume
        and config.history_path.exists()
        and config.history_path.stat().st_size > 0
    ):
        raise FileExistsError(
            "search history already exists; use --resume or a new artifact path"
        )
    existing_rows = _read_history(config.history_path) if resume else []
    existing_run_ids = {
        row["search_run_id"] for row in existing_rows if row["search_run_id"]
    }
    if len(existing_run_ids) > 1:
        raise ValueError("search history contains more than one search_run_id")
    if run_id is not None and existing_run_ids and run_id not in existing_run_ids:
        raise ValueError("run_id does not match the resumed search history")
    run_id = run_id or (
        next(iter(existing_run_ids)) if existing_run_ids else str(uuid4())
    )
    completed = {
        (row["candidate_id"], int(row["tuning_seed"]))
        for row in existing_rows if row["status"] == "success"
    }
    source_cache = {}
    split_cache = {}
    reported_preparation_failures = set()
    succeeded = failed = skipped = 0
    progress = tqdm(
        total=total_runs,
        desc="Search runs",
        unit="run",
        disable=not verbose,
    )
    device = torch.device(config.raw["execution"]["device"])
    training = config.raw["training"]
    stopping = training["early_stopping"]
    scheduler = training["scheduler"]
    fail_fast = config.raw["execution"]["fail_fast"]

    for candidate in candidates:
        for seed in config.tuning_seeds:
            progress.set_postfix_str(
                f"{candidate.dataset}/{candidate.architecture}/{seed}"
            )
            if (candidate.candidate_id, seed) in completed:
                skipped += 1
                progress.update(1)
                continue

            split_key = (candidate.dataset, seed)
            prepared_split = _prepare_search_split(
                config,
                candidate.dataset,
                seed,
                source_cache,
                split_cache,
            )
            if isinstance(prepared_split, _PreparationFailure):
                failed += 1
                if split_key not in reported_preparation_failures:
                    affected_runs = sum(
                        other.dataset == candidate.dataset
                        and (other.candidate_id, seed) not in completed
                        for other in candidates
                    )
                    _append_preparation_failure(
                        config.failures_path,
                        run_id,
                        candidate.dataset,
                        seed,
                        prepared_split.stage,
                        prepared_split.error,
                        affected_runs,
                    )
                    reported_preparation_failures.add(split_key)
                progress.update(1)
                if fail_fast:
                    progress.close()
                    raise prepared_split.error
                continue

            try:
                loaded_dataset = source_cache[candidate.dataset]
                source_data = loaded_dataset.data
                parameters = candidate.hyperparameters
                validate_registered_compatibility(
                    candidate.architecture,
                    "matched-semi-inductive-validation-only",
                    loaded_dataset.metadata,
                    device=device,
                )
                model_adapter = get_model_adapter(candidate.architecture)
                validation_data = prepare_validation_data(
                    source_data,
                    prepared_split,
                    requires_reduced_features=(
                        model_adapter.capabilities.requires_reduced_features
                    ),
                    normalize_features=parameters["normalize_features"],
                    out_channels=parameters["out_channels"],
                    reduction_method=parameters.get("reduction_method", "pca"),
                    random_state=seed,
                )
                train_data = validation_data.splits.train.to(device)
                validation_split = validation_data.splits.validation.to(device)
                torch.manual_seed(seed)
                predictor = LinkPrediction(
                    train_data, **_predictor_kwargs(candidate)
                )
                predictor.model = predictor.model.to(device)
                parameter_count = count_trainable_parameters(predictor.model)
                summary = train_with_validation_selection(
                    predictor,
                    train_data,
                    validation_split,
                    training_policy=TrainingPolicy(
                        max_epochs=training["max_epochs"],
                        optimizer_parameters={
                            "learning_rate": parameters["learning_rate"],
                            "weight_decay": parameters["weight_decay"],
                        },
                        selection_metric="roc_auc",
                        early_stopping_patience=stopping["patience"],
                        early_stopping_min_delta=stopping["min_delta"],
                        scheduler=scheduler["type"],
                        scheduler_parameters={
                            name: value for name, value in scheduler.items()
                            if name != "type"
                        },
                    ),
                    seed=seed,
                    device=device,
                    verbose=verbose,
                    progress_label=(
                        f"{candidate.dataset}/{candidate.architecture}/{seed}"
                    ),
                    progress_position=1,
                )
                probabilities = predictor.test(validation_split)
                metrics = calculate_metrics(
                    validation_split.edge_label, probabilities
                )
                row = {
                    "search_run_id": run_id,
                    "candidate_id": candidate.candidate_id,
                    "dataset": candidate.dataset,
                    "architecture": candidate.architecture,
                    "tuning_seed": seed,
                    "status": "success",
                    "hyperparameters": json.dumps(
                        candidate.hyperparameters,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "validation_roc_auc": metrics["roc_auc"],
                    "validation_average_precision": metrics["average_precision"],
                    "best_epoch": summary.best_epoch,
                    "epochs_trained": summary.epochs_executed,
                    "stopped_early": summary.stopped_early,
                    "training_time_seconds": summary.training_time_seconds,
                    "reducer_fit_time_seconds": (
                        validation_data.preprocessing_time_seconds
                        if validation_data.preprocessing_time_seconds is not None
                        else ""
                    ),
                    "trainable_parameter_count": parameter_count,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
                _append_history(config.history_path, row)
                existing_rows.append({key: str(value) for key, value in row.items()})
                completed.add((candidate.candidate_id, seed))
                succeeded += 1
            except Exception as error:
                failed += 1
                _append_failure(
                    config.failures_path, run_id, candidate, seed, error
                )
                if fail_fast:
                    progress.close()
                    raise
            finally:
                progress.update(1)
    progress.close()
    winners = aggregate_and_select(
        config, candidates, _read_history(config.history_path), run_id
    )
    return SearchSummary(
        run_id, total_runs, succeeded, failed, skipped, len(winners)
    )


def parse_args():
    """Parse search-runner CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", help="path to a search YAML specification")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main(cli_args=None):
    """Load, expand, and optionally execute a validation-only search."""
    args = parse_args() if cli_args is None else cli_args
    config = load_search_config(args.config)
    candidates = generate_candidates(config)
    total_runs = len(candidates) * len(config.tuning_seeds)
    print(json.dumps({
        "datasets": len(config.datasets),
        "architectures": len(config.architectures),
        "candidates": len(candidates),
        "training_runs": total_runs,
        "maximum_epochs": total_runs * config.raw["training"]["max_epochs"],
    }, indent=2, sort_keys=True))
    if args.dry_run:
        for candidate in candidates:
            print(
                f"{candidate.dataset}/{candidate.architecture}/"
                f"{candidate.candidate_id[:10]} "
                f"{json.dumps(candidate.hyperparameters, sort_keys=True)}"
            )
    summary = execute_search(
        config,
        candidates,
        dry_run=args.dry_run,
        resume=args.resume,
        verbose=args.verbose,
    )
    print(
        f"Search {summary.run_id}: {summary.succeeded} succeeded, "
        f"{summary.failed} failed, {summary.skipped} skipped, "
        f"{summary.selected} selected"
    )
    return summary


if __name__ == "__main__":
    main()
