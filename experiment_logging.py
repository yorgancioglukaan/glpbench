"""CSV logging for link-prediction experiments."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


@dataclass
class ExperimentResult:
    """Describe one link-prediction experiment result.

    Parameters
    ----------
    dataset : str
        Dataset name.
    model_type : str
        ``"symmetric"`` or ``"asymmetric"``.
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
        if self.model_type not in {"symmetric", "asymmetric"}:
            raise ValueError("model_type must be 'symmetric' or 'asymmetric'")
        if self.split_type not in {"inductive", "transductive"}:
            raise ValueError("split_type must be 'inductive' or 'transductive'")
        if self.conv_layer_type not in {"GCNConv", "GraphConv", "SAGEConv"}:
            raise ValueError("conv_layer_type is not supported")

        for name in ("hidden_channels", "out_channels", "epochs"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.query_size, int) or self.query_size < 0:
            raise ValueError("query_size must be a non-negative integer")
        if not isinstance(self.random_seed, int) or self.random_seed < 0:
            raise ValueError("random_seed must be a non-negative integer")

        if not isinstance(self.train_ratio, (int, float)) or not 0 <= self.train_ratio <= 1:
            raise ValueError("train_ratio must be between 0 and 1")
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
        """Append one result and return its generated run ID.

        Parameters
        ----------
        result : ExperimentResult
            Validated experiment result.

        Returns
        -------
        str
            Generated run UUID.
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
