# GLPBench

GLPBench is a Python library and command-line toolkit for reproducible graph
link-prediction evaluation. It provides benchmark datasets, matched
transductive and semi-inductive protocols, leakage audits, reference models,
hyperparameter search, benchmark grids, and structured result logging.

This is an alpha release. Its public API and configuration schema may change
before the first stable release.

Install GLPBench from PyPI with:

```bash
python -m pip install glpbench
```

Run `glpbench --help` to list the available experiment, benchmark, search, and
audit commands.

## Contents

- [Key features](#key-features)
- [Experiment CLI](#experiment-cli)
- [Benchmark grids](#benchmark-grids)
- [Hyperparameter search](#hyperparameter-search)
- [Convergence audits](#convergence-audits)
- [Configuration formats](#configuration-formats)
- [Python API](#python-api)
  - [Dataset loading](#dataset-loading)
  - [Feature preprocessing and graph splits](#feature-preprocessing-and-graph-splits)
  - [Link-prediction models](#link-prediction-models)
  - [Result logging](#result-logging)
- [License](#license)

## Key features

- Load attributed graph benchmarks through a consistent dataset interface.
- Generate comparable transductive and semi-inductive splits with disjoint
  validation queries and automatic leakage checks.
- Evaluate graph autoencoders, MLP decoders, variational models, DEAL,
  feature-only baselines, and asymmetric decoders under the same protocol.
- Run individual experiments, YAML-defined benchmark grids, validation-only
  hyperparameter searches, and convergence audits from one CLI.
- Reuse random seeds, graph splits, and evaluation negatives across models for
  paired comparisons.
- Record metrics, configuration, runtime, checkpoint, and parameter-count
  metadata in analysis-ready CSV files.

GLPBench currently targets simple, undirected, attributed graphs.

## Experiment CLI

Run an experiment with a supported dataset:

```bash
glpbench experiment cora out/results.csv
```

When `--model-type` is omitted, the script runs every registered architecture
on the same splits. The eight current architectures produce sixteen result
rows under one run ID after evaluation on both test settings. The
feature-cosine baseline is evaluated directly without training.

For trainable architectures, each result row records
`training_time_seconds`, the total wall-clock time spent across all executed
training epochs, validation, and checkpoint selection. Every architecture
also records its exact `trainable_parameter_count`. Reduced-feature models
record `reducer_fit_time_seconds` separately; other models leave that field
empty. Paired transductive and inductive rows share these values because they
use the same fitted reducer and trained model. Non-trainable baselines leave
training time empty and report zero trainable parameters.

Add `--verbose` to show dataset preparation, architecture progress, epoch progress, and live loss components:

```bash
glpbench experiment cora out/results.csv --verbose
```

Common options include:

```bash
glpbench experiment citeseer out/results.csv \
  --conv-layer-type GATv2Conv \
  --gatv2-kwargs '{"heads": 2, "concat": false, "dropout": 0.1}' \
  --hidden-channels 128 \
  --out-channels 64 \
  --reduction-method pca \
  --decoder-hidden-channels 64 \
  --decoder-dropout 0.2 \
  --kl-weight 1.0 \
  --learning-rate 0.01 \
  --lr-scheduler reduce-on-plateau \
  --lr-scheduler-kwargs \
    '{"factor": 0.5, "patience": 25, "threshold": 0.001, "minimum_learning_rate": 0.00001}' \
  --weight-decay 0.0 \
  --query-size 10 \
  --validation-query-size 10 \
  --train-ratio 0.8 \
  --epochs 100 \
  --selection-metric roc_auc \
  --early-stopping-patience 20 \
  --early-stopping-min-delta 0.001 \
  --seed 42 \
  --device cpu
```

Use `--query-fraction` instead of `--query-size` to select a fraction of
eligible, non-isolated nodes. The two options are mutually exclusive; when
both are omitted, the CLI selects 10 query nodes.

Validation uses a disjoint set of cold-start query nodes. Its size defaults to
the resolved test-query size; use `--validation-query-size` or
`--validation-query-fraction` to override it. Both query populations are
excluded from feature-reducer fitting and from the training topology.

Without early stopping, trainable models run for the full requested epoch
budget. After every epoch, the model is evaluated on the semi-inductive
validation split, and the best checkpoint is restored before either test split
is evaluated. Checkpoints use ROC-AUC by default;
`--selection-metric average_precision` selects by average precision instead.

GATv2 constructor options can be supplied as a JSON object with
`--gatv2-kwargs`. Result rows retain these options in `extra_fields` alongside
the reducer, learning rate, device, query-selection configuration, and any
architecture-specific decoder, VGAE, DEAL, or regularization settings.

Early stopping is disabled unless `--early-stopping-patience` is supplied.
When enabled, training stops after that many consecutive epochs without an
improvement greater than `--early-stopping-min-delta`. The configured epoch
count remains the maximum budget, and the best checkpoint is restored in all
cases. Logged training time includes training, validation, and checkpoint
selection.

Learning-rate scheduling is disabled by default. Use `--lr-scheduler` with
`reduce-on-plateau` or `cosine-annealing`; scheduler parameters are supplied
as a JSON object through `--lr-scheduler-kwargs`. Plateau scheduling uses the
same validation metric as checkpoint selection, while cosine annealing follows
the fixed maximum epoch budget. Result rows log the scheduler configuration
and final learning rate.

Repeat `--seed` to run paired repetitions in one invocation:

```bash
glpbench experiment cora out/results.csv --seed 1 --seed 2 --seed 3
```

Alternatively, `--num-seeds 10` runs seeds 0 through 9. All rows from the
invocation share a `run_id`; repetitions of the same model and hyperparameter
configuration share a stable `configuration_id`. Data splits and evaluation
negatives are generated once per seed and reused across architectures.

Available model types are `symmetric`, `asymmetric`, `mlp`,
`asymmetric-mlp`, `variational`, `deal`, `feature-cosine`, and `feature-mlp`.
`feature-mlp` is a trainable, graph-independent baseline that applies the
symmetric MLP decoder directly to pairs of input node features. Repeat
`--model-type` to choose several explicitly:

```bash
glpbench experiment cora \
  out/results.csv \
  --model-type mlp \
  --model-type variational
```

Supported datasets are Cora, CiteSeer, PubMed, Amazon Photo, Amazon Computers, Coauthor CS, DBLP, WikiCS, and Deezer Europe. Supported convolution layers are `GCNConv`, `GraphConv`, `SAGEConv`, and `GATv2Conv`. Run `glpbench experiment --help` for the complete option list. Newly registered architectures will automatically become available to the CLI.

For example, run the Amazon Photo benchmark with:

```bash
glpbench experiment amazon-photo out/results.csv --verbose
```

## Benchmark grids

The benchmark runner expands a versioned YAML configuration and calls the
experiment API sequentially. Start from
[`configs/example_benchmark.yaml`](configs/example_benchmark.yaml), then inspect
the expansion without training:

```bash
glpbench benchmark configs/example_benchmark.yaml --dry-run
```

Run the grid with:

```bash
glpbench benchmark configs/example_benchmark.yaml --verbose
```

Verbose benchmark output uses one overall model–seed progress bar and one
temporary epoch bar for the current run. Use `--verbose-details` instead when
diagnosing an experiment and you need the standalone seed, architecture, and
stage messages.

All newly written rows share one benchmark `run_id`. Individual failures are
appended to the configured JSONL error file while later jobs continue. Use
`--fail-fast` to stop on the first error, or `--resume` to read completed
`(configuration_id, random_seed)` pairs from the result CSV and avoid duplicate
rows. Dataset, model, convolution, reducer, regularization, seed, and common
training dimensions are validated before execution.

A version-2 benchmark can import dataset-specific architecture settings
written by the search runner. The selected-configuration document must contain
every requested dataset–architecture pair. Search fields for preprocessing and
architecture-specific settings are translated to the experiment API and
validated before execution.

## Hyperparameter search

The search runner performs validation-only model selection and never evaluates
either test split. Start with
[`configs/example_search.yaml`](configs/example_search.yaml). Inspect its
deterministic candidate expansion without loading data or training:

```bash
glpbench search configs/example_search.yaml --dry-run
```

Run or resume the example with:

```bash
glpbench search configs/example_search.yaml --verbose
glpbench search configs/example_search.yaml --resume --verbose
```

Successful candidate–seed runs are written to a search-history CSV. Selection
requires all configured tuning seeds and uses mean semi-inductive validation
ROC-AUC, then average precision and trainable parameter count as tie-breakers.
The runner also writes an aggregate CSV, a JSONL failure log, and a YAML file
containing the selected resolved configuration for each dataset–architecture
pair.

## Convergence audits

The audit runner records validation trajectories for selected model
configurations under a fixed full-epoch budget. It uses best-validation
checkpoint selection with early stopping and learning-rate scheduling disabled,
allowing convergence behavior to be inspected without evaluating either test
split.

Start from [`configs/example_audit.yaml`](configs/example_audit.yaml), inspect
the resolved jobs, and then run or resume the audit:

```bash
glpbench audit configs/example_audit.yaml --dry-run
glpbench audit configs/example_audit.yaml --verbose
glpbench audit configs/example_audit.yaml --resume --verbose
```

## Configuration formats

The benchmark, search, and audit CLI commands continue to accept the documented
`version`-based YAML formats. GLPBench also provides a strict canonical
configuration API using `schema_version`, frozen dataclasses, unknown-field
rejection, and path-aware validation errors. In `0.1.0a3`, canonical
configurations are loaded and inspected through the Python API; they are not
yet accepted directly by the CLI runners.

See the [configuration guide](docs/configuration.md) for the supported legacy
format matrix and canonical API entry points.

## Python API

### Dataset loading

`glpbench.datasets.load_dataset()` resolves a registered provider, validates
the graph, moves it to a device, and attaches dataset metadata. Its
`normalize_features` option remains available for direct API compatibility;
command-line workflows load raw features and apply normalization in the shared
preprocessing pipeline:

```python
from glpbench.datasets import load_dataset

dataset = load_dataset("cora", root="data", device="cpu")
data = dataset.data
print(dataset.metadata)
```

`glpbench.datasets.available_datasets()` returns the normalized names accepted by the loader.

### Feature preprocessing and graph splits

`glpbench.protocols.calculate_reduced_features()` fits a registered feature reducer and stores the transformed features in `data.x_reduced`. Pass a node mask to fit without exposing inductive query features:

```python
import torch

from glpbench.protocols import (
    calculate_reduced_features,
    generate_comparable_splits,
    select_query_nodes,
)

query = select_query_nodes(data, query_size=3, seed=42)
fit_mask = torch.ones(data.num_nodes, dtype=torch.bool)
fit_mask[query] = False
calculate_reduced_features(data, dim=64, fit_nodes=fit_mask)
splits = generate_comparable_splits(
    data,
    query=query,
    train_ratio=0.8,
)
train = splits.train
val = splits.validation
test_transductive = splits.tests["transductive"]
test_inductive = splits.tests["semi-inductive"]
```

Available reduction methods are `pca`, `truncated-svd`, and
`sparse-random-projection`. For example:

```bash
glpbench experiment coauthor-cs out/results.csv \
  --model-type asymmetric \
  --reduction-method truncated-svd
```

`select_query_nodes()` performs deterministic automatic selection while ensuring
that every query retains at least one visible neighbor. Both split functions can
also select queries directly by accepting `query_size` or `query_fraction` with
a `seed`. `generate_comparable_splits()` creates transductive and inductive test
sets with matching message-passing graphs and equal numbers of labeled edges.
It automatically calls `audit_comparable_splits()` to enforce node-role,
topology, positive/negative leakage, orientation, and paired-test invariants.
The split protocol expects a simple undirected graph and rejects self-loops or
duplicate directed edge entries rather than silently modifying them.
`glpbench.protocols.inductive_split()` is available when only a cold-start split is needed.

### Link-prediction models

The model registry currently includes:

| Model type | Description |
| --- | --- |
| `symmetric` | Graph autoencoder with inner-product decoding |
| `mlp` | Graph autoencoder with a learned symmetric MLP decoder |
| `variational` | Variational graph autoencoder with KL regularization |
| `deal` | Dual structural and attribute encoders for inductive prediction |
| `feature-cosine` | Non-trainable cosine similarity over node features |
| `feature-mlp` | Trainable graph-independent MLP over feature pairs |
| `asymmetric` | Graph latent-to-feature inner-product decoder |
| `asymmetric-mlp` | Learned MLP over graph-derived sources and feature-derived targets |

`glpbench.models.LinkPrediction` provides the shared training and inference interface:

```python
from glpbench.models import LinkPrediction

predictor = LinkPrediction(
    train,
    model_type="symmetric",
    conv_layer_type="SAGEConv",
    hidden_channels=128,
    out_channels=64,
)

for _ in range(100):
    loss = predictor.train()

probabilities = predictor.test(test_inductive)
```

Configure MLP decoders with `decoder_hidden_channels` and `decoder_dropout`,
and the variational model with `kl_weight`:

```python
mlp = LinkPrediction(
    train,
    model_type="mlp",
    decoder_hidden_channels=64,
    decoder_dropout=0.2,
)
vgae = LinkPrediction(train, model_type="variational", kl_weight=1.0)
```

Architecture-specific options are available for MLP decoders, VGAE KL
weighting, DEAL loss and inference weights, and asymmetric-model
regularization. Run `glpbench experiment --help` for their CLI definitions;
resolved settings are retained in the result log.

GATv2-specific options can be passed directly or through `gatv2_kwargs`:

```python
predictor = LinkPrediction(
    train,
    conv_layer_type="GATv2Conv",
    heads=4,
    dropout=0.2,
)
```

### Result logging

`glpbench.logging.ExperimentResult` validates experiment metadata and metrics. `CSVResultLogger` appends results to a stable CSV schema; rows written by the same logger share a run ID:

```python
from glpbench.logging import CSVResultLogger, ExperimentResult

logger = CSVResultLogger("out/results.csv")
logger.log(ExperimentResult(...))
logger.reset_run_id()  # Start a new experiment group.
```

## License

GLPBench is distributed under the
[Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).
