"""Deterministic content fingerprints for datasets and realized splits."""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from collections.abc import Mapping

import torch

from glpbench.contracts import DatasetMetadata, SplitBundle


def _write_text(hasher, value):
    encoded = value.encode("utf-8")
    hasher.update(struct.pack("<Q", len(encoded)))
    hasher.update(encoded)


def _canonical_value(value):
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    return value


def _write_json(hasher, value):
    _write_text(
        hasher,
        json.dumps(
            _canonical_value(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ),
    )


def _write_tensor(hasher, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("fingerprinted values must be torch.Tensor objects")
    value = tensor.detach().cpu().contiguous()
    _write_text(hasher, str(value.dtype))
    _write_json(hasher, list(value.shape))
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    if sys.byteorder == "big" and value.element_size() > 1:
        size = value.element_size()
        raw = b"".join(
            raw[offset:offset + size][::-1]
            for offset in range(0, len(raw), size)
        )
    hasher.update(struct.pack("<Q", len(raw)))
    hasher.update(raw)


def _sorted_columns(tensor, *, labels=None, undirected=False):
    columns = tensor.detach().cpu().t().tolist()
    if undirected:
        columns = [[min(source, target), max(source, target)]
                   for source, target in columns]
        columns = list({tuple(column) for column in columns})
    if labels is not None:
        label_values = labels.detach().cpu().tolist()
        columns = [column + [label] for column, label in zip(columns, label_values)]
    columns.sort(key=tuple)
    width = 3 if labels is not None else 2
    if not columns:
        return torch.empty((width, 0), dtype=torch.int64)
    return torch.tensor(columns, dtype=torch.int64).t().contiguous()


def fingerprint_dataset(data, metadata=None, *, directed=None, dataset_version=None):
    """Return a device- and edge-order-independent SHA-256 graph fingerprint."""
    if metadata is not None and not isinstance(metadata, DatasetMetadata):
        raise TypeError("metadata must be DatasetMetadata")
    if directed is None:
        directed = metadata.directed if metadata is not None else False
    if dataset_version is None and metadata is not None:
        dataset_version = metadata.dataset_version
    if not isinstance(directed, bool):
        raise TypeError("directed must be a boolean")

    hasher = hashlib.sha256()
    _write_text(hasher, "glpbench:dataset:v1")
    _write_json(hasher, {
        "num_nodes": int(data.num_nodes),
        "directed": directed,
        "dataset_version": dataset_version,
    })
    _write_tensor(
        hasher,
        _sorted_columns(data.edge_index, undirected=not directed),
    )
    features = getattr(data, "x", None)
    _write_json(hasher, {"has_node_features": features is not None})
    if features is not None:
        _write_tensor(hasher, features)
    return f"sha256:{hasher.hexdigest()}"


def fingerprint_split(bundle, dataset_fingerprint):
    """Return a deterministic SHA-256 fingerprint for a realized split."""
    if not isinstance(bundle, SplitBundle):
        raise TypeError("bundle must be a SplitBundle")
    if not isinstance(dataset_fingerprint, str) or not dataset_fingerprint:
        raise ValueError("dataset_fingerprint must be a non-empty string")

    hasher = hashlib.sha256()
    _write_text(hasher, "glpbench:split:v1")
    _write_json(hasher, {
        "dataset_fingerprint": dataset_fingerprint,
        "protocol": bundle.metadata.name,
        "schema_version": bundle.metadata.schema_version,
        "seed": bundle.metadata.seed,
        "parameters": bundle.metadata.parameters,
        "requested_counts": bundle.metadata.requested_counts,
        "realized_counts": bundle.metadata.realized_counts,
        "validation_query_nodes": bundle.validation_query_nodes,
        "test_query_nodes": bundle.test_query_nodes,
    })
    if bundle.visible_node_mask is not None:
        _write_tensor(hasher, bundle.visible_node_mask)

    partitions = {"train": bundle.train}
    if bundle.validation is not None:
        partitions["validation"] = bundle.validation
    partitions.update({f"test:{name}": split for name, split in bundle.tests.items()})
    for name in sorted(partitions):
        split = partitions[name]
        _write_text(hasher, name)
        _write_tensor(hasher, _sorted_columns(split.edge_index))
        _write_tensor(
            hasher,
            _sorted_columns(split.edge_label_index, labels=split.edge_label),
        )
    return f"sha256:{hasher.hexdigest()}"


__all__ = ["fingerprint_dataset", "fingerprint_split"]
