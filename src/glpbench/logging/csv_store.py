"""CSV logging for link-prediction experiments."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from glpbench.training import validate_scheduler_config


def make_configuration_id(configuration):
    """Return a stable SHA-256 identifier for a configuration mapping.

    Dictionary keys are serialized in sorted order, so insertion order does
    not affect the identifier. Values must be JSON serializable.
    """
    if not isinstance(configuration, dict):
        raise TypeError("configuration must be a dictionary")
    serialized = json.dumps(
        configuration, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


@dataclass
class ExperimentResult:
    """Describe one link-prediction experiment result.

    Parameters
    ----------
    dataset : str
        Dataset name.
    model_type : str
        Registered prediction architecture.
    split_type : str
        ``"inductive"`` or ``"transductive"``.
    conv_layer_type : str
        Message-passing layer type.
    hidden_channels : int
        Hidden-layer size.
    out_channels : int
        Latent representation size.
    query_size : int
        Number of cold-start query nodes.
    train_ratio : float
        Training ratio.
    epochs : int
        Number of training epochs.
    random_seed : int
        Random seed used by the experiment.
    normalize_features : bool, optional
        Whether input node features were normalized.
    configuration_id : str, optional
        Stable identifier shared by repetitions of the same configuration.
    training_time_seconds : float, optional
        Total wall-clock time spent training and selecting a checkpoint.
    trainable_parameter_count : int, optional
        Number of trainable neural-model parameters.
    reducer_fit_time_seconds : float, optional
        Wall-clock time spent fitting and applying feature reduction.
    epochs_trained : int, optional
        Number of completed training epochs.
    best_epoch : int, optional
        One-based epoch with the best validation score.
    selected_epoch : int, optional
        One-based epoch of the checkpoint used for test evaluation.
    training_policy : str, optional
        Human-readable training-policy label.
    checkpoint_strategy : str, optional
        ``"best-validation"`` or ``"final"``.
    selection_metric : str, optional
        Validation metric used for checkpoint selection.
    best_validation_score : float, optional
        Best observed validation score.
    early_stopping_patience : int, optional
        Allowed consecutive non-improving epochs, or ``None`` when disabled.
    early_stopping_min_delta : float, optional
        Minimum validation-score improvement required when stopping is enabled.
    early_stopping_minimum_epochs : int, optional
        Epoch at which early-stopping patience begins.
    stopped_early : bool, optional
        Whether training ended before the maximum epoch budget.
    regularization_type : str, optional
        AGAE regularization method.
    regularization_strength : float, optional
        Explicit regularization coefficient.
    weight_decay : float, optional
        Adam weight-decay coefficient.
    lr_scheduler : str, optional
        Learning-rate scheduler name.
    lr_scheduler_kwargs : dict, optional
        Scheduler construction parameters.
    final_learning_rate : float, optional
        Learning rate after the final completed epoch.
    roc_auc, average_precision : float, optional
        Ranking metrics in the unit interval.
    loss : float, optional
        Non-negative selected-checkpoint loss.
    accuracy, precision, recall, f1 : float, optional
        Thresholded classification metrics in the unit interval.
    extra_fields : dict, optional
        Additional JSON-serializable experiment settings.
    """

    dataset: str
    model_type: str
    split_type: str
    conv_layer_type: str
    hidden_channels: int
    out_channels: int
    query_size: int
    train_ratio: float
    epochs: int
    random_seed: int
    normalize_features: bool = True
    configuration_id: str | None = None
    training_time_seconds: float | None = None
    trainable_parameter_count: int | None = None
    reducer_fit_time_seconds: float | None = None
    epochs_trained: int | None = None
    best_epoch: int | None = None
    selected_epoch: int | None = None
    training_policy: str = "default"
    checkpoint_strategy: str | None = None
    selection_metric: str | None = None
    best_validation_score: float | None = None
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    early_stopping_minimum_epochs: int = 0
    stopped_early: bool = False
    regularization_type: str = "none"
    regularization_strength: float = 0.0
    weight_decay: float = 0.0
    lr_scheduler: str = "none"
    lr_scheduler_kwargs: dict[str, Any] = field(default_factory=dict)
    final_learning_rate: float | None = None
    roc_auc: float | None = None
    average_precision: float | None = None
    loss: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    extra_fields: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate experiment parameters and metrics."""
        if not isinstance(self.dataset, str) or not self.dataset.strip():
            raise ValueError("dataset must be a non-empty string")
        if self.model_type not in {
            "symmetric", "asymmetric", "mlp", "asymmetric-mlp", "variational",
            "feature-cosine", "feature-mlp", "deal",
        }:
            raise ValueError("model_type is not supported")
        if self.split_type not in {"inductive", "transductive"}:
            raise ValueError("split_type must be 'inductive' or 'transductive'")
        if self.conv_layer_type not in {
            "GCNConv", "GraphConv", "SAGEConv", "GATv2Conv", "None"
        }:
            raise ValueError("conv_layer_type is not supported")
        if self.regularization_type not in {
            "none", "latent-l2", "mse-alignment", "cosine-alignment",
            "graph-smoothness",
        }:
            raise ValueError("regularization_type is not supported")
        if (
            self.model_type not in {"asymmetric", "asymmetric-mlp"}
            and self.regularization_type != "none"
        ):
            raise ValueError(
                "explicit regularization is only supported by asymmetric models"
            )

        for name in ("hidden_channels", "out_channels", "epochs"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.query_size, int) or self.query_size < 0:
            raise ValueError("query_size must be a non-negative integer")
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")
        if not isinstance(self.normalize_features, bool):
            raise ValueError("normalize_features must be a boolean")
        if self.configuration_id is not None and (
            not isinstance(self.configuration_id, str)
            or not self.configuration_id.strip()
        ):
            raise ValueError("configuration_id must be a non-empty string")
        if self.training_time_seconds is not None and (
            not isinstance(self.training_time_seconds, (int, float))
            or isinstance(self.training_time_seconds, bool)
            or not math.isfinite(self.training_time_seconds)
            or self.training_time_seconds < 0
        ):
            raise ValueError(
                "training_time_seconds must be a finite non-negative number"
            )
        if self.trainable_parameter_count is not None and (
            not isinstance(self.trainable_parameter_count, int)
            or isinstance(self.trainable_parameter_count, bool)
            or self.trainable_parameter_count < 0
        ):
            raise ValueError(
                "trainable_parameter_count must be a non-negative integer"
            )
        if self.reducer_fit_time_seconds is not None and (
            not isinstance(self.reducer_fit_time_seconds, (int, float))
            or isinstance(self.reducer_fit_time_seconds, bool)
            or not math.isfinite(self.reducer_fit_time_seconds)
            or self.reducer_fit_time_seconds < 0
        ):
            raise ValueError(
                "reducer_fit_time_seconds must be a finite non-negative number"
            )
        if self.epochs_trained is not None and (
            not isinstance(self.epochs_trained, int)
            or isinstance(self.epochs_trained, bool)
            or not 0 <= self.epochs_trained <= self.epochs
        ):
            raise ValueError("epochs_trained must be between zero and epochs")
        if self.best_epoch is not None and (
            not isinstance(self.best_epoch, int)
            or isinstance(self.best_epoch, bool)
            or self.epochs_trained is None
            or not 1 <= self.best_epoch <= self.epochs_trained
        ):
            raise ValueError("best_epoch must be within the trained epochs")
        if self.selected_epoch is not None and (
            not isinstance(self.selected_epoch, int)
            or isinstance(self.selected_epoch, bool)
            or self.epochs_trained is None
            or not 1 <= self.selected_epoch <= self.epochs_trained
        ):
            raise ValueError("selected_epoch must be within the trained epochs")
        if not isinstance(self.training_policy, str) or not self.training_policy.strip():
            raise ValueError("training_policy must be a non-empty string")
        if self.checkpoint_strategy not in {None, "best-validation", "final"}:
            raise ValueError("checkpoint_strategy is not supported")
        if (self.selected_epoch is None) != (self.checkpoint_strategy is None):
            raise ValueError(
                "selected_epoch and checkpoint_strategy must be provided together"
            )
        if self.selection_metric not in {None, "roc_auc", "average_precision"}:
            raise ValueError("selection_metric is not supported")
        if self.best_validation_score is not None and (
            not isinstance(self.best_validation_score, (int, float))
            or isinstance(self.best_validation_score, bool)
            or not math.isfinite(self.best_validation_score)
            or not 0 <= self.best_validation_score <= 1
        ):
            raise ValueError(
                "best_validation_score must be a finite value between zero and one"
            )
        selection_values = (
            self.best_epoch,
            self.selection_metric,
            self.best_validation_score,
        )
        if any(value is None for value in selection_values) and any(
            value is not None for value in selection_values
        ):
            raise ValueError(
                "best_epoch, selection_metric, and best_validation_score must "
                "be provided together"
            )
        if self.early_stopping_patience is not None and (
            not isinstance(self.early_stopping_patience, int)
            or isinstance(self.early_stopping_patience, bool)
            or self.early_stopping_patience <= 0
        ):
            raise ValueError(
                "early_stopping_patience must be a positive integer"
            )
        if (
            not isinstance(self.early_stopping_min_delta, (int, float))
            or isinstance(self.early_stopping_min_delta, bool)
            or not math.isfinite(self.early_stopping_min_delta)
            or self.early_stopping_min_delta < 0
        ):
            raise ValueError(
                "early_stopping_min_delta must be a finite non-negative number"
            )
        if (
            not isinstance(self.early_stopping_minimum_epochs, int)
            or isinstance(self.early_stopping_minimum_epochs, bool)
            or not 0 <= self.early_stopping_minimum_epochs <= self.epochs
        ):
            raise ValueError(
                "early_stopping_minimum_epochs must be between zero and epochs"
            )
        if not isinstance(self.stopped_early, bool):
            raise ValueError("stopped_early must be a boolean")
        if self.stopped_early and (
            self.early_stopping_patience is None
            or self.epochs_trained is None
            or self.epochs_trained >= self.epochs
        ):
            raise ValueError(
                "stopped_early requires enabled early stopping and fewer "
                "trained epochs than the maximum"
            )

        if not isinstance(self.train_ratio, (int, float)) or not 0 <= self.train_ratio <= 1:
            raise ValueError("train_ratio must be between 0 and 1")
        for name in ("regularization_strength", "weight_decay"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.model_type not in {"asymmetric", "asymmetric-mlp"}
            and self.regularization_strength != 0
        ):
            raise ValueError(
                "regularization_strength is only supported by asymmetric models"
            )
        validate_scheduler_config(
            self.lr_scheduler, self.lr_scheduler_kwargs, epochs=self.epochs
        )
        if self.final_learning_rate is not None and (
            not isinstance(self.final_learning_rate, (int, float))
            or isinstance(self.final_learning_rate, bool)
            or not math.isfinite(self.final_learning_rate)
            or self.final_learning_rate < 0
        ):
            raise ValueError(
                "final_learning_rate must be a finite non-negative number"
            )
        for name in ("roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, (int, float)) or not 0 <= value <= 1):
                raise ValueError(f"{name} must be between 0 and 1")
        if self.loss is not None and (
            not isinstance(self.loss, (int, float)) or self.loss < 0
        ):
            raise ValueError("loss must be non-negative")
        if not isinstance(self.extra_fields, dict):
            raise ValueError("extra_fields must be a dictionary")


class CSVResultLogger:
    """Append link-prediction experiment results to a CSV file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination CSV file.
    run_id : str, optional
        Identifier shared by all rows logged by this instance.
    """

    FIELDNAMES = [
        "run_id",
        "configuration_id",
        "timestamp_utc",
        "dataset",
        "model_type",
        "split_type",
        "conv_layer_type",
        "hidden_channels",
        "out_channels",
        "query_size",
        "train_ratio",
        "epochs",
        "random_seed",
        "normalize_features",
        "training_time_seconds",
        "trainable_parameter_count",
        "reducer_fit_time_seconds",
        "epochs_trained",
        "best_epoch",
        "selected_epoch",
        "training_policy",
        "checkpoint_strategy",
        "selection_metric",
        "best_validation_score",
        "early_stopping_patience",
        "early_stopping_min_delta",
        "early_stopping_minimum_epochs",
        "stopped_early",
        "regularization_type",
        "regularization_strength",
        "weight_decay",
        "lr_scheduler",
        "lr_scheduler_kwargs",
        "final_learning_rate",
        "roc_auc",
        "average_precision",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "extra_fields",
    ]

    def __init__(self, path, run_id=None):
        self.path = Path(path)
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id or str(uuid4())

    def reset_run_id(self, run_id=None):
        """Start a new run and return its identifier.

        Parameters
        ----------
        run_id : str, optional
            Identifier to use. By default, a new UUID is generated.

        Returns
        -------
        str
            New run identifier.
        """
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id or str(uuid4())
        return self.run_id
    def log(self, result):
        """Append one result and return the logger's current run ID.

        Parameters
        ----------
        result : ExperimentResult
            Validated experiment result.

        Returns
        -------
        str
            Identifier shared by rows in the current run.
        """
        if not isinstance(result, ExperimentResult):
            raise TypeError("result must be an ExperimentResult")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        if file_exists and self.path.stat().st_size:
            with self.path.open("r", newline="", encoding="utf-8") as csv_file:
                header = next(csv.reader(csv_file), None)
            if header != self.FIELDNAMES:
                raise ValueError("existing CSV schema does not match the logger schema")

        row = asdict(result)
        row["extra_fields"] = json.dumps(row["extra_fields"], sort_keys=True)
        row["lr_scheduler_kwargs"] = json.dumps(
            row["lr_scheduler_kwargs"], sort_keys=True
        )
        row = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **row,
        }

        write_header = not file_exists or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return self.run_id


@dataclass
class EpochHistoryRecord:
    """Describe metrics recorded after one completed training epoch.

    All metric values are validated before a record can be logged.
    """

    configuration_id: str
    dataset: str
    model_type: str
    random_seed: int
    training_policy: str
    epoch: int
    training_loss: float
    learning_rate: float
    validation_roc_auc: float
    validation_average_precision: float

    def __post_init__(self):
        """Validate epoch metadata and metrics."""
        for name in (
            "configuration_id", "dataset", "model_type", "training_policy"
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("random_seed", "epoch"):
            value = getattr(self, name)
            minimum = 0 if name == "random_seed" else 1
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value < minimum
            ):
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if (
            not isinstance(self.training_loss, (int, float))
            or isinstance(self.training_loss, bool)
            or not math.isfinite(self.training_loss)
            or self.training_loss < 0
        ):
            raise ValueError("training_loss must be a finite non-negative number")
        if (
            not isinstance(self.learning_rate, (int, float))
            or isinstance(self.learning_rate, bool)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate < 0
        ):
            raise ValueError("learning_rate must be a finite non-negative number")
        for name in ("validation_roc_auc", "validation_average_precision"):
            value = getattr(self, name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or not 0 <= value <= 1
            ):
                raise ValueError(f"{name} must be between zero and one")


class CSVEpochHistoryLogger:
    """Append per-epoch training and validation metrics to a CSV file.

    Parameters
    ----------
    path : str or pathlib.Path
        Destination CSV file.
    run_id : str, optional
        Identifier shared by all rows logged by this instance.
    """

    FIELDNAMES = [
        "run_id",
        "timestamp_utc",
        "configuration_id",
        "dataset",
        "model_type",
        "random_seed",
        "training_policy",
        "epoch",
        "training_loss",
        "learning_rate",
        "validation_roc_auc",
        "validation_average_precision",
    ]

    def __init__(self, path, run_id=None):
        self.path = Path(path)
        if run_id is not None and (
            not isinstance(run_id, str) or not run_id.strip()
        ):
            raise ValueError("run_id must be a non-empty string")
        self.run_id = run_id or str(uuid4())

    def log(self, record):
        """Append one epoch-history record and return the current run ID.

        Parameters
        ----------
        record : EpochHistoryRecord
            Validated epoch metrics.

        Returns
        -------
        str
            Identifier shared by rows in the current run.
        """
        if not isinstance(record, EpochHistoryRecord):
            raise TypeError("record must be an EpochHistoryRecord")

        self.path.parent.mkdir(parents=True, exist_ok=True)
        file_exists = self.path.exists()
        if file_exists and self.path.stat().st_size:
            with self.path.open("r", newline="", encoding="utf-8") as csv_file:
                header = next(csv.reader(csv_file), None)
            if header != self.FIELDNAMES:
                raise ValueError(
                    "existing CSV schema does not match the epoch logger schema"
                )

        row = {
            "run_id": self.run_id,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **asdict(record),
        }
        write_header = not file_exists or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        return self.run_id
