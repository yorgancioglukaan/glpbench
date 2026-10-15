# Command-line interface

GLPBench exposes four subcommands:

| Command | Purpose |
| --- | --- |
| `experiment` | Run models on one dataset and one matched split |
| `benchmark` | Expand and execute a YAML benchmark grid |
| `search` | Select configurations using validation data only |
| `audit` | Compare training and checkpoint policies without test evaluation |

Display the available commands with:

```bash
glpbench --help
```

Each subcommand provides its own option reference:

```bash
glpbench experiment --help
glpbench benchmark --help
glpbench search --help
glpbench audit --help
```

## Reproducibility

Seeds determine query selection, graph splits, evaluation negatives, model
initialization, and the per-epoch training-negative schedule. Repeat `--seed`
to evaluate several seeds in one invocation. Models evaluated within one seed
receive the same prepared graph splits and evaluation candidates.

## Result files

Experiment and benchmark commands append analysis-ready rows to CSV files.
Rows record metrics, model and protocol settings, random seeds, checkpoint
metadata, runtime, and trainable parameter count. Rows from one invocation
share a run identifier, while equivalent configurations share a stable
configuration identifier.
