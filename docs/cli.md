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

Configuration-driven commands use versioned YAML files. Paths are resolved
from the working directory where the command is run. The CLI currently consumes
the documented legacy `version` formats; canonical `schema_version`
configurations introduced in `0.1.0a3` are available through the Python API but
are not yet direct CLI inputs.

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

The dataset is the first positional argument. The result CSV is optional and
defaults to `out/results.csv`. Repeat `--model-type` or `--seed` to select
several values. When
`--model-type` is omitted, all registered models are evaluated on the same
prepared split. Run `glpbench experiment --help` for model-specific options,
feature reduction, early stopping, scheduling, and query selection.

To run an unpackaged research model, pass a Python file that exports one
adapter as `MODEL_ADAPTER`:

```bash
glpbench experiment cora out/local-model.csv \
  --model-type ./models/my_model.py
```

The adapter is loaded only for the current process. See [Models](models.md)
for the file convention and adapter contract.

`experiment` is configured entirely with command-line arguments; it does not
use a YAML configuration file. The output CSV contains one row per model,
seed, and test split. Compare `roc_auc` or `average_precision` across rows with
the same seed and split type. Rows from the invocation share a `run_id`, while
`configuration_id` identifies an equivalent resolved model-and-protocol
configuration across invocations.

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

### Benchmark configuration

A benchmark file defines result paths, shared defaults, datasets, seeds, and
models. Lists expand into a Cartesian grid where applicable. For example:

```yaml
version: 1
output: out/benchmark.csv
error_output: out/benchmark.errors.jsonl
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

datasets: [Cora, CiteSeer]
models:
  - type: symmetric
    conv_layer_types: [GCNConv, SAGEConv]
  - type: feature-cosine
```

A model `type` may also be a local `.py` adapter file. Unlike other legacy
configuration paths, a relative model-file path is resolved from the
configuration file's directory, making the configuration portable as a unit.

Model entries may add architecture-specific parameters, reduction methods, or
regularization profiles. See
[`configs/example_benchmark.yaml`](https://github.com/yorgancioglukaan/glpbench/blob/main/configs/example_benchmark.yaml)
for a broader example.

The result CSV has the same row structure as `experiment`, making rows directly
comparable after grouping by dataset, seed, and split type. Failed jobs are
written as JSON lines to `error_output`; an empty or absent error file means no
job failure was recorded. With `--resume`, existing configuration–seed results
are retained and only missing jobs are run.

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

### Search configuration

A search file declares its artifacts, candidate-generation strategy,
validation rule, training and split policies, and parameter spaces. This
excerpt shows the main structure for sampling three candidates on one dataset:

```yaml
version: 1
kind: hyperparameter-search

artifacts:
  history: out/search_history.csv
  failures: out/search_errors.jsonl
  selected_configurations: out/selected_configurations.yaml

execution:
  device: cpu
  data_root: data
  fail_fast: false

datasets: [cora]
search:
  strategy: seeded-random
  num_candidates: 3
  candidate_seed: 2026
  include_default: true
  tuning_seeds: [0, 1, 2]
  require_all_seeds_for_selection: true

selection:
  granularity: [dataset, architecture]
  validation_setting: semi-inductive
  tuning_conv_layer: GCNConv
  metric: roc_auc
  aggregation: mean
  tie_tolerance: 0.001
  tie_breakers:
    - {metric: average_precision, direction: maximize}
    - {metric: trainable_parameter_count, direction: minimize}

training:
  max_epochs: 100
  early_stopping:
    enabled: true
    patience: 10
    min_delta: 0.001

common_space:
  hidden_channels: {values: [32, 64], sampling: random}

models:
  symmetric: {}
  # The complete file must also define split, preprocessing, remaining common
  # spaces, and model-specific spaces for every trainable architecture.
```

See
[`configs/example_search.yaml`](https://github.com/yorgancioglukaan/glpbench/blob/main/configs/example_search.yaml)
for the complete required set of model-specific spaces. The excerpt above is
intentionally abbreviated; the search runner requires every trainable
architecture to be defined. The history CSV contains one row per candidate,
dataset, architecture, and tuning seed. Its companion
`*_aggregates.csv` reports seed-aggregated validation metrics and marks the
winning row with `selected`. The selected-configuration YAML contains the
resolved winners for later benchmarks, while the JSONL failure artifact records
unsuccessful candidate runs. No test metric is used for selection.

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

### Audit configuration

An audit specification identifies selected hyperparameters and declares the
dataset–model pairs, validation split, seeds, epoch budget, and output paths:

```yaml
version: 1
kind: validation-convergence-audit

artifacts:
  results: out/audit.csv
  epoch_history: out/audit_epochs.csv
  failures: out/audit.errors.jsonl

execution:
  device: cpu
  data_root: data
  fail_fast: false

seeds: [0]
selected_configurations:
  path: configs/example_selected_configurations.yaml
  granularity: [dataset, architecture]
  require_all_pairs: true

datasets: [cora]
models: [symmetric, asymmetric]
splits:
  protocol: matched-semi-inductive-validation-only
  query_fraction: 0.05
  validation_query_fraction: 0.05
  train_ratio: 0.8
training:
  max_epochs: 100
  selection_metric: roc_auc
  checkpoint: best-validation
  early_stopping: {enabled: false}
  scheduler: {type: none}
```

See
[`configs/example_audit.yaml`](https://github.com/yorgancioglukaan/glpbench/blob/main/configs/example_audit.yaml)
and its illustrative selected-configuration file for a runnable pair. The
summary CSV contains one row per audited configuration and seed, including the
best epoch and selected-checkpoint validation metrics. The epoch-history CSV
contains the validation trajectory needed for convergence plots. The failure
JSONL records jobs that did not complete; the audit never writes test-set
performance.

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
