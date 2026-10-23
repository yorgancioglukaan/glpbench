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

Use `available_model_types()` to inspect the active model registry and
`get_model_adapter()` to retrieve an adapter by its case- and
separator-insensitive name. Custom adapters can be added for the current
Python process with `register_model_adapter()`; duplicate names are rejected
unless replacement is explicitly requested.

Each bundled model also has a runtime-checkable adapter exported from
`glpbench.models`. These adapters provide parameter validation, construction,
loss, scoring, and result-metadata hooks. The current runners still use
`LinkPrediction`; adapter-based registration and execution will follow in a
later migration step.

```python
from glpbench.models import get_model_adapter, register_model_adapter

adapter = get_model_adapter("feature_mlp")
register_model_adapter(MyModelAdapter())
```

`get_model_registration()` remains temporarily available as a compatibility
view for code using the legacy constructor interface.
