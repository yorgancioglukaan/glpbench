# API usage

The Python API exposes the same datasets, split protocols, models, and result
loggers used by the command-line runners. This is useful when an experiment
needs custom orchestration beyond the CLI and YAML configuration formats.

```python
from glpbench.datasets import load_dataset
from glpbench.protocols import generate_comparable_splits

data = load_dataset("cora", root="data", device="cpu")
train, validation, test_transductive, test_inductive = (
    generate_comparable_splits(data, query_size=50, train_ratio=0.8, seed=42)
)
```

See the {doc}`api` reference for the currently exported classes and functions.
A broader API usage guide, including custom components and complete training
examples, will be added as the public interfaces stabilize.
