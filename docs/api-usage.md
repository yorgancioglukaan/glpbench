# API usage

The Python API exposes the same datasets, split protocols, models, and result
loggers used by the command-line runners. This is useful when an experiment
needs custom orchestration beyond the CLI and YAML configuration formats.

```python
from glpbench.datasets import load_dataset
from glpbench.protocols import generate_comparable_splits

dataset = load_dataset("cora", root="data", device="cpu")
splits = generate_comparable_splits(
    dataset.data, query_size=50, train_ratio=0.8, seed=42
)
train = splits.train
validation = splits.validation
test_inductive = splits.tests["semi-inductive"]
```

See the {doc}`api` reference for the currently exported classes and functions.
A broader API usage guide, including custom components and complete training
examples, will be added as the public interfaces stabilize.

## Strict configuration parsing

Canonical configuration documents use a `schema_version` and `kind` field.
They can be validated independently of command execution:

```python
from glpbench.configuration import load_config, resolve_experiment_specs

config = load_config("canonical_benchmark.yaml")
specifications = resolve_experiment_specs(config)
```

The parser returns frozen configuration dataclasses, rejects unknown fields at
every nesting level, and includes full field paths in validation errors.
Existing `version`-based CLI files are routed through behavior-preserving
compatibility parsers by the same general loader. See the
{doc}`configuration` guide for the supported format matrix.
