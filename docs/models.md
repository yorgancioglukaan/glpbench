# Models

GLPBench provides a shared training and inference adapter for several
link-prediction approaches:

| Model type | Description |
| --- | --- |
| `symmetric` | Graph autoencoder with inner-product decoding |
| `mlp` | Graph autoencoder with a learned symmetric MLP decoder |
| `variational` | Variational graph autoencoder |
| `deal` | Structural and attribute encoders for inductive prediction |
| `feature-cosine` | Non-trainable feature-cosine baseline |
| `feature-mlp` | Graph-independent MLP over feature pairs |
| `asymmetric` | Graph latent-to-feature inner-product decoder |
| `asymmetric-mlp` | MLP over graph-derived sources and feature-derived targets |

Graph encoders support `GCNConv`, `GraphConv`, `SAGEConv`, and `GATv2Conv`.
Architecture-specific settings are available through the Python constructor
and experiment CLI.

## Python interface

```python
from glpbench.models import LinkPrediction

predictor = LinkPrediction(
    train_data,
    model_type="symmetric",
    conv_layer_type="GCNConv",
    hidden_channels=128,
    out_channels=64,
)

for _ in range(100):
    loss = predictor.train()

probabilities = predictor.test(test_data)
```

Direct calls may continue to pass the bundled model options as keyword
arguments. Orchestration code can instead resolve one explicit adapter
parameter mapping, which prevents settings for unrelated architectures from
reaching shared execution:

```python
from glpbench.models import (
    LinkPrediction,
    get_model_adapter,
    resolve_model_parameters,
)

adapter = get_model_adapter("symmetric")
parameters = resolve_model_parameters(adapter, {
    "hidden_channels": 128,
    "out_channels": 64,
    "conv_layer_type": "GCNConv",
    "gatv2_kwargs": {},
})
predictor = LinkPrediction(
    train_data,
    model_type=adapter.name,
    model_parameters=parameters,
)
```

Do not combine `model_parameters` with legacy model-construction keyword
arguments in the same call. Optimizer options such as `learning_rate` and
`weight_decay` remain separate.

Use `available_model_types()` to inspect the active model registry and
`get_model_adapter()` to retrieve an adapter by its case- and
separator-insensitive name. Custom adapters can be added for the current
Python process with `register_model_adapter()`; duplicate names are rejected
unless replacement is explicitly requested.

Each bundled model also has a runtime-checkable adapter exported from
`glpbench.models`. These adapters provide parameter validation, construction,
loss, scoring, and result-metadata hooks. `LinkPrediction` and the experiment
runners resolve models through this adapter registry.

```python
from glpbench.models import get_model_adapter, register_model_adapter

adapter = get_model_adapter("feature_mlp")
register_model_adapter(MyModelAdapter())
```

## Load a model from a local file

For research code that does not need to be packaged, put one adapter in a
Python file and export it as `MODEL_ADAPTER`. For example:

```python
# my_model.py
from glpbench.models import FeatureCosineAdapter


class MyModelAdapter(FeatureCosineAdapter):
    name = "my-model"


MODEL_ADAPTER = MyModelAdapter()
```

Pass the file itself wherever a model name is accepted:

```bash
glpbench experiment cora out/results.csv --model-type ./my_model.py
```

The loader imports only the explicitly named file and requires the
conventional `MODEL_ADAPTER` export. It registers the adapter for the current
process; it does not modify the installed package. Reusing the same file in a
process is safe, while duplicate adapter names and files changed after loading
are rejected. Result metadata records the original reference, resolved path,
and SHA-256 digest of the file.

In YAML model entries, use the `.py` path in place of the registered model
name. Relative model paths are resolved from the directory containing the
configuration file:

```yaml
models:
  - type: models/my_model.py
```

The example subclasses a bundled adapter only to keep it short. A custom
architecture can instead provide any object conforming to the public
`ModelAdapter` protocol. Each file currently exports one adapter; multiple
adapters per file and installed-package discovery are intentionally deferred.
