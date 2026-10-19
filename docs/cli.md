# Command-line interface

GLPBench exposes four subcommands through one `glpbench` entry point:

| Command | Purpose |
| --- | --- |
| `experiment` | Run models on one dataset and one matched split |
| `benchmark` | Expand and execute a YAML benchmark grid |
| `search` | Select configurations using validation data only |
| `audit` | Record validation convergence without test evaluation |

Display the top-level help or the complete options for any command with:

```bash
glpbench --help
glpbench <command> --help
```

## `experiment`: run one experiment

`experiment` is the simplest entry point. It loads one dataset, creates a
matched split, trains one or more models, evaluates both test settings, and
appends the results to a CSV file.

```bash
glpbench experiment cora out/results.csv \
  --model-type symmetric \
  --model-type feature-cosine \
  --conv-layer-type GCNConv \
  --query-size 50 \
  --validation-query-size 50 \
  --epochs 100 \
  --seed 42 \
  --device cpu \
  --verbose
```

The dataset is the first positional argument and the result CSV is the second.
Repeat `--model-type` or `--seed` to select several values. When
`--model-type` is omitted, all registered models are evaluated on the same
prepared split. Run `glpbench experiment --help` for model-specific options,
feature reduction, early stopping, scheduling, and query selection.

## `benchmark`: run a test grid

`benchmark` expands a YAML specification into dataset, model, convolution, and
seed combinations. Use `--dry-run` before a large benchmark to inspect the
resolved jobs without loading datasets or training models.

```bash
glpbench benchmark configs/example_benchmark.yaml --dry-run
glpbench benchmark configs/example_benchmark.yaml --verbose
```

Add `--resume` to skip completed configuration–seed pairs already present in
the output CSV. `--fail-fast` stops at the first failed job; otherwise failures
are recorded and the remaining jobs continue. `--verbose-details` shows the
nested experiment messages in addition to overall progress.

## `search`: select hyperparameters

`search` evaluates candidate configurations only on the semi-inductive
validation protocol. It does not inspect either test split. Candidate results,
aggregates, failures, and selected configurations are written to the artifact
paths declared in the YAML specification.

```bash
glpbench search configs/example_search.yaml --dry-run
glpbench search configs/example_search.yaml --resume --verbose
```

The selected-configuration YAML can subsequently be consumed by a compatible
benchmark configuration, keeping model selection separate from final test
evaluation.

## `audit`: inspect convergence

`audit` runs selected configurations for a specified full training budget and
logs validation trajectories without consulting test performance. It uses
best-validation checkpoint selection with early stopping and learning-rate
scheduling disabled.

```bash
glpbench audit configs/example_audit.yaml --dry-run
glpbench audit configs/example_audit.yaml --resume --verbose
```

An audit configuration normally points to the selected configurations produced
by a search and declares its own seeds, epoch budget, and output artifacts. Use
a training-policy ablation benchmark when comparing checkpoint, early-stopping,
or scheduler policies.

## Creating configuration files

Configuration-driven commands use versioned YAML files. Keep configurations in
the repository so that a result can be traced to the exact grid and training
policy used to produce it. Paths inside a configuration are resolved from the
working directory where the command is run.

The CLI currently consumes the documented legacy `version` formats. Canonical
`schema_version` configurations introduced in `0.1.0a3` are available through
the Python configuration API but are not yet direct CLI inputs.

### Benchmark configuration

A benchmark file defines output paths, shared defaults, datasets, seeds, and
models. Lists expand into a Cartesian grid where applicable. A small benchmark
can be written as:

```yaml
version: 1

output: out/example_results.csv
error_output: out/example_errors.jsonl
data_root: data
seeds: [0, 1, 2]

defaults:
  query_size: 50
  validation_query_size: 50
  train_ratio: 0.8
  epochs: 100
  hidden_channels: 64
  out_channels: 32
  learning_rate: 0.01
  selection_metric: roc_auc
  normalize_features: true
  device: cpu

datasets:
  - Cora
  - CiteSeer

models:
  - type: symmetric
    conv_layer_types: [GCNConv, SAGEConv]

  - type: feature-cosine
```

Model entries may add architecture-specific `parameters`, reduction methods,
or regularization profiles. Start from
[`configs/example_benchmark.yaml`](https://github.com/yorgancioglukaan/glpbench/blob/main/configs/example_benchmark.yaml)
for a broader example.

### Search configuration

A search specification declares:

- artifact paths for candidate history, failures, and selected configurations;
- datasets and execution settings;
- candidate-generation strategy and tuning seeds;
- validation metric and tie-breakers;
- split and training policies;
- common and model-specific search spaces.

Start from
[`configs/example_search.yaml`](https://github.com/yorgancioglukaan/glpbench/blob/main/configs/example_search.yaml),
reduce `num_candidates` and the dataset list for an initial trial, and inspect
the expansion with `--dry-run`.

### Audit configuration

An audit specification identifies a selected-configuration file and declares
the dataset–model pairs, validation split, seeds, and maximum epoch budget. Its
checkpoint, early-stopping, and scheduler settings are fixed by the audit
protocol. Start from
[`configs/example_audit.yaml`](https://github.com/yorgancioglukaan/glpbench/blob/main/configs/example_audit.yaml)
and use `--dry-run` to confirm every requested dataset–model pair resolves.

## Reproducibility

Seeds determine query selection, graph splits, evaluation negatives, model
initialization, and the per-epoch training-negative schedule. Repeat `--seed`
to evaluate several seeds in one experiment invocation. Models evaluated within
one seed receive the same prepared graph splits and evaluation candidates.

## Result files

Experiment and benchmark commands append analysis-ready rows to CSV files.
Rows record metrics, model and protocol settings, random seeds, checkpoint
metadata, runtime, and trainable parameter count. Rows from one invocation
share a run identifier, while equivalent configurations share a stable
configuration identifier.
