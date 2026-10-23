# Configuration formats

GLPBench recognizes canonical and legacy YAML configuration formats. The
canonical format uses `schema_version`; legacy files use `version`. A document
must define exactly one of these fields.

As of `0.1.0a4`, canonical documents are supported by the Python configuration
API. The command-line benchmark, search, and audit runners continue to consume
the legacy formats described below; canonical files are not yet direct CLI
inputs.

## Canonical documents

Canonical documents are strictly parsed into frozen dataclasses. Every
document declares its schema and purpose:

```yaml
schema_version: 1
kind: benchmark
```

Unknown fields are rejected at every nesting level. Errors identify the full
field path, such as `training.early_stopping.patience` or
`models[0].parameters.decoder_dropout`.

Use the Python API to load and inspect canonical configurations:

```python
from glpbench.configuration import load_config, resolve_experiment_specs

config = load_config("canonical_benchmark.yaml")
specifications = resolve_experiment_specs(config)
```

## Legacy compatibility

Existing CLI configuration files remain supported without conversion. Their
original parsers are retained as compatibility adapters so defaults, expansion
order, seeded candidate generation, configuration identifiers, and path
semantics remain unchanged.

| Document | Supported legacy schema |
| --- | --- |
| Benchmark grid | `version: 1` |
| Selected-hyperparameter benchmark | `version: 2` |
| Hyperparameter search | `version: 1` |
| Training-policy ablation | `version: 1` |
| Validation-convergence audit | `version: 1` |
| Selected-hyperparameter artifact | `version: 1` |

The `benchmark`, `search`, and `audit` CLI commands accept these legacy files.
`glpbench.configuration.load_config()` can load either format.
The command-specific canonical loaders intentionally reject legacy documents;
use the general loader when format compatibility is required.

Paths retain their existing meaning and are interpreted relative to the
working directory from which the command is run.
