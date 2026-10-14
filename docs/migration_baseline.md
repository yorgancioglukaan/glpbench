# Package Migration Baseline

This document records the package boundary established by the
`src/glpbench` migration. It describes existing behavior; it does not declare
every implementation helper to be permanent public API.

## Package and command surface

The installed distribution provides:

- `import glpbench` and `glpbench.__version__`;
- `python -m glpbench`;
- the `glpbench` console command; and
- the `experiment`, `benchmark`, `search`, and `audit` subcommands.

The package and all implementations live under `src/glpbench`. Root-level
compatibility modules were removed before the next alpha release because the
project had no external users relying on them.

## Documented Python imports

The README demonstrates package-native imports:

```python
from glpbench.datasets import available_datasets, load_dataset
from glpbench.logging import CSVResultLogger, ExperimentResult
from glpbench.models import LinkPrediction
from glpbench.protocols import (
    calculate_reduced_features,
    generate_comparable_splits,
    select_query_nodes,
)
```

## Current internal dependency direction

The package modules have the following important dependencies:

```text
runners.audit      -> runners.benchmark, datasets, logging, evaluation,
                      models, runners.experiment, protocols
runners.search     -> datasets, evaluation, preprocessing, models,
                      training, runners.experiment, protocols
runners.benchmark  -> datasets, logging, preprocessing, models,
                      training, runners.experiment
runners.experiment -> datasets, logging, evaluation, preprocessing,
                      models, training, protocols
evaluation         -> models, protocols
logging            -> training
protocols          -> preprocessing
```

Package modules must use package-relative or package-qualified imports; legacy
root-module imports are no longer supported.

## Baseline verification

Before and after each migration batch:

1. Run the complete unit and regression suite.
2. Build a wheel.
3. Inspect the wheel contents.
4. Install it outside the source checkout without dependencies.
5. Verify the package import, module entry point, console command, and all four
   subcommand help screens.

Changes to numerical behavior, split semantics, CLI arguments, or result
columns require a separate proposal and must not be hidden inside the package
layout migration.
