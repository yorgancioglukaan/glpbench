# Quickstart

## Run one experiment

The experiment command loads a dataset, creates matched transductive and
semi-inductive splits, trains the requested models, and appends result rows to
a CSV file:

```bash
glpbench experiment cora out/results.csv \
  --model-type symmetric \
  --model-type feature-cosine \
  --epochs 100 \
  --seed 42 \
  --device cpu \
  --verbose
```

Omit `--model-type` to evaluate every registered model on the same split.
Repeat `--seed` to run paired repetitions.

## Run a benchmark grid

Inspect a benchmark configuration without training:

```bash
glpbench benchmark configs/example_benchmark.yaml --dry-run
```

Run or resume the benchmark:

```bash
glpbench benchmark configs/example_benchmark.yaml --verbose
glpbench benchmark configs/example_benchmark.yaml --resume --verbose
```

## Load a dataset from Python

```python
from glpbench.datasets import load_dataset

data = load_dataset("cora", root="data", device="cpu")
print(data.dataset_metadata)
```

See the {doc}`cli` and {doc}`api` pages for the complete entry points.
