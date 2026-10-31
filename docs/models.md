# Models

GLPBench provides a shared training and inference adapter for several
link-prediction approaches:

| Model type | Description |
| --- | --- |
| `gae` | Graph autoencoder with inner-product decoding |
| `gae-mlp` | Graph autoencoder with a learned order-invariant MLP decoder |
| `vgae` | Variational graph autoencoder |
| `deal` | Structural and attribute encoders for inductive prediction |
| `feature-cosine` | Non-trainable feature-cosine baseline |
| `feature-mlp` | Graph-independent MLP over feature pairs |
| `agae` | Graph latent-to-feature inner-product decoder |
| `agae-mlp` | MLP over graph-derived sources and feature-derived targets |

Graph encoders support `GCNConv`, `GraphConv`, `SAGEConv`, and `GATv2Conv`.
Architecture-specific settings are available through the Python constructor
and experiment CLI.

## Python interface

```python
from glpbench.models import LinkPrediction

predictor = LinkPrediction(
    train_data,
    model_type="gae",
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

adapter = get_model_adapter("gae")
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

Do not combine `model_parameters` with model-construction keyword
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
runners resolve models through this adapter registry. An adapter's immutable
`parameter_defaults` mapping declares the model parameters accepted by the
adapter; `validate_parameters()` validates their values and any
cross-parameter constraints. Custom adapters should expose an immutable
mapping as well; the conformance helper below verifies this part of the
contract.

```python
from glpbench.models import get_model_adapter, register_model_adapter

adapter = get_model_adapter("feature-mlp")
register_model_adapter(custom_adapter)
```

## Load a model from a local file

For research code that does not need to be packaged, put the model and its
adapter in a Python file and export one adapter instance as `MODEL_ADAPTER`.
The adapter is the boundary between GLPBench's shared training loop and the
model-specific implementation.

### Required adapter interface

GLPBench validates adapters against the runtime-checkable `ModelAdapter`
protocol. Subclass `BaseModelAdapter` to reuse common parameter validation,
loss delegation, scoring, and empty result metadata, or implement the protocol
directly. Run the conformance check at the end of this page before using an
adapter in an experiment.

- `name: str` is the unique canonical registration name. It must be nonempty.
- `capabilities: ModelCapabilities` declares whether the model is trainable,
  uses message passing or convolution layers, requires reduced features or
  labels, supports the protocol's endpoint orientations, and supports the
  requested device.
- `parameter_defaults: Mapping[str, object]` declares every accepted model
  parameter and its default. It should be immutable, normally through
  `MappingProxyType`.
- `validate_parameters(parameters) -> Mapping[str, object]` rejects unknown or
  invalid values, merges omitted defaults, and returns an immutable normalized
  mapping without mutating its input.
- `build(train_data, parameters) -> torch.nn.Module` constructs a fresh model.
  It must not train the model or mutate `train_data`.
- `compute_training_loss(model, train_data, edge_label_index, edge_label,
  criterion) -> TrainingLoss` computes one scalar differentiable loss from the
  exact candidate batch supplied by GLPBench. `TrainingLoss.components` should
  contain named scalar tensors for reporting. A non-trainable adapter should
  reject this operation clearly.
- `score_edges(model, data, edge_index) -> torch.Tensor` returns one finite
  one-dimensional score per candidate edge. The score representation must
  agree with `capabilities.output_kind`; bundled evaluation adapters return
  probabilities for ordinary classification metrics.
- `result_metadata(parameters) -> Mapping[str, object]` returns serializable
  model-specific settings for result logging. It must not mutate its input or
  reuse names reserved for shared result fields.

For local-file discovery, the module must also assign exactly one conforming
adapter instance to `MODEL_ADAPTER`. Registration and loading check the
protocol structure, capabilities, parameter declaration, name collisions, and
file provenance. The conformance utility additionally exercises construction,
loss, gradients, scoring, metadata, and input-mutation behavior.

### Subclass the adapter base

`BaseModelAdapter` requires a name, capabilities, an immutable `defaults`
mapping, and a `build()` implementation. The model constructed by `build()`
can use the supplied loss and scoring methods by implementing
`compute_training_loss()`, `encode()`, and `decode_edges()` as the bundled models
do. Override either adapter method for a different loss or scoring path:

```python
from types import MappingProxyType

from glpbench.contracts import ModelCapabilities
from glpbench.models import BaseModelAdapter


class MyAdapter(BaseModelAdapter):
    name = "my-model"
    capabilities = ModelCapabilities(requires_labels=True)
    defaults = MappingProxyType({"hidden_channels": 64})

    def _validate_values(self, parameters):
        if parameters["hidden_channels"] <= 0:
            raise ValueError("hidden_channels must be positive")

    def build(self, train_data, parameters):
        parameters = self.validate_parameters(parameters)
        return MyModel(train_data.num_features, parameters["hidden_channels"])


MODEL_ADAPTER = MyAdapter()
```

The base class checks required declarations when instantiated, merges and
freezes defaults, and rejects unknown parameters. The full example below
implements the structural protocol directly and illustrates custom loss and
scoring methods.

The following complete example implements a trainable, feature-only pair MLP.
It deliberately does not use the message-passing graph. Node features are
projected independently, combined with commutative pair operations, and scored
as a candidate edge:

```python
# models/pair_mlp.py
from collections.abc import Mapping
from numbers import Real
from types import MappingProxyType

import torch

from glpbench.contracts import ModelCapabilities, TrainingLoss
from glpbench.exceptions import ConfigurationError


class PairMLP(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, dropout, score_scale):
        super().__init__()
        self.score_scale = score_scale
        self.project = torch.nn.Linear(in_channels, hidden_channels)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(2 * hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, 1),
        )

    def logits(self, data, edge_index):
        node_features = self.project(data.x).relu()
        source, target = edge_index
        source_features = node_features[source]
        target_features = node_features[target]
        pair_features = torch.cat(
            [
                source_features * target_features,
                (source_features - target_features).abs(),
            ],
            dim=-1,
        )
        return self.score_scale * self.classifier(pair_features).view(-1)


class PairMLPAdapter:
    name = "pair-mlp"
    capabilities = ModelCapabilities(
        uses_message_passing=False,
        uses_conv_layer=False,
        requires_labels=True,
    )
    parameter_defaults = MappingProxyType({
        "hidden_channels": 64,
        "decoder_dropout": 0.0,
        "pair_score_scale": 1.0,
    })

    def validate_parameters(self, parameters):
        if not isinstance(parameters, Mapping):
            raise ConfigurationError(
                "model parameters must be a mapping",
                component="pair-mlp",
            )
        unknown = set(parameters).difference(self.parameter_defaults)
        if unknown:
            raise ConfigurationError(
                f"unsupported parameter(s): {', '.join(sorted(unknown))}",
                component="pair-mlp",
            )
        values = {**self.parameter_defaults, **dict(parameters)}
        hidden = values["hidden_channels"]
        dropout = values["decoder_dropout"]
        score_scale = values["pair_score_scale"]
        if not isinstance(hidden, int) or isinstance(hidden, bool) or hidden <= 0:
            raise ConfigurationError(
                "hidden_channels must be a positive integer",
                component="pair-mlp",
            )
        if (
            not isinstance(dropout, Real)
            or isinstance(dropout, bool)
            or not 0 <= dropout < 1
        ):
            raise ConfigurationError(
                "decoder_dropout must be in [0, 1)",
                component="pair-mlp",
            )
        if (
            not isinstance(score_scale, Real)
            or isinstance(score_scale, bool)
            or score_scale <= 0
        ):
            raise ConfigurationError(
                "pair_score_scale must be positive",
                component="pair-mlp",
            )
        return MappingProxyType(values)

    def build(self, train_data, parameters):
        values = self.validate_parameters(parameters)
        return PairMLP(
            in_channels=train_data.num_features,
            hidden_channels=values["hidden_channels"],
            dropout=values["decoder_dropout"],
            score_scale=values["pair_score_scale"],
        )

    def compute_training_loss(
        self,
        model,
        train_data,
        edge_label_index,
        edge_label,
        criterion,
    ):
        logits = model.logits(train_data, edge_label_index)
        reconstruction = criterion(logits, edge_label)
        return TrainingLoss(
            total=reconstruction,
            components={"reconstruction": reconstruction},
        )

    @torch.no_grad()
    def score_edges(self, model, data, edge_index):
        model.eval()
        return model.logits(data, edge_index).sigmoid()

    def result_metadata(self, parameters):
        values = self.validate_parameters(parameters)
        return MappingProxyType({
            "decoder_dropout": values["decoder_dropout"],
            "pair_score_scale": values["pair_score_scale"],
        })


MODEL_ADAPTER = PairMLPAdapter()
```

The experiment CLI already exposes `--hidden-channels` and
`--decoder-dropout`, so those values are routed to this adapter automatically:

```bash
glpbench experiment cora out/results.csv \
  --model-type ./models/pair_mlp.py \
  --hidden-channels 32 \
  --decoder-dropout 0.1
```

### Current custom-parameter limitation

The `experiment` and `benchmark` commands currently route only the model
parameters for which the experiment CLI provides explicit options. This is a
known limitation: an additional parameter declared by a custom adapter cannot
yet be set through those commands unless it happens to share an existing CLI
option. GLPBench plans to replace this fixed routing with a general
adapter-parameter mapping.

As a temporary workaround, use the search runner. It accepts model-specific
keys declared by the adapter's `parameter_defaults` mapping and validates
their values through `validate_parameters()` without requiring a GLPBench
source change. For example, the `pair_score_scale` parameter from the adapter
above can be searched with this model section:

```yaml
models:
  ./models/pair_mlp.py:
    pair_score_scale:
      values: [0.5, 1.0, 2.0]
      sampling: random
```

Place this section in a complete search configuration such as
[`configs/example_search.yaml`](https://github.com/yorgancioglukaan/glpbench/blob/main/configs/example_search.yaml),
then run:

```bash
glpbench search configs/pair_mlp_search.yaml --verbose
```

The local adapter path is resolved relative to the configuration file. Use
the Python API instead when a fixed custom parameter value is needed without
running a search.

The loader imports only the explicitly named file and requires the
conventional `MODEL_ADAPTER` export. It registers the adapter for the current
process; it does not modify the installed package. Reusing the same file in a
process is safe, while duplicate adapter names and files changed after loading
are rejected. Experiment and benchmark result rows record the original
reference, resolved path, and SHA-256 digest of the file. Search artifacts do
not currently persist this source provenance.

In YAML, use the `.py` path in the same position as the registered model name.
The surrounding syntax depends on the configuration type. Use only the form
for the file being created.

Legacy benchmark:

```yaml
models:
  - type: models/pair_mlp.py
    parameters:
      hidden_channels: 32
      decoder_dropout: 0.1
```

Canonical benchmark, ablation, or audit document:

```yaml
models:
  - architecture: models/pair_mlp.py
    parameters:
      hidden_channels: 32
      decoder_dropout: 0.1
```

Canonical search document:

```yaml
models:
  - architecture: models/pair_mlp.py
    spaces:
      pair_score_scale:
        values: [0.5, 1.0, 2.0]
```

Legacy search:

```yaml
models:
  models/pair_mlp.py:
    pair_score_scale:
      values: [0.5, 1.0, 2.0]
      sampling: random
```

Legacy convergence audit:

```yaml
models: [models/pair_mlp.py]
```

Relative model paths are resolved from the directory containing the
configuration file. Direct Python construction with `LinkPrediction` expects
an already registered adapter name; call `resolve_model_reference()` first if
an API workflow begins with a local file path:

```python
from glpbench.models import LinkPrediction, resolve_model_reference

model_name = resolve_model_reference("./models/pair_mlp.py")
predictor = LinkPrediction(
    train_data,
    model_type=model_name,
    model_parameters={
        "hidden_channels": 32,
        "decoder_dropout": 0.1,
        "pair_score_scale": 2.0,
    },
)
```

To adapt a message-passing model, declare `uses_message_passing=True` and
`uses_conv_layer=True`, then use `train_data.edge_index` inside the model's
encoding path. The adapter still owns only construction, loss, scoring, and
metadata; GLPBench continues to own epoch iteration, negative sampling,
checkpoint selection, timing, and result persistence.

Each local file currently exports one adapter. Multiple adapters per file and
installed-package discovery are intentionally deferred.

## Check adapter conformance

Use the same contract checks applied to GLPBench's bundled models before
running a custom adapter. Supply a small graph containing every input declared
by the adapter, including reduced features or graph context when required:

```python
from glpbench.testing import assert_model_adapter_conforms

assert_model_adapter_conforms(
    MODEL_ADAPTER,
    toy_data,
    {
        "hidden_channels": 8,
        "decoder_dropout": 0.0,
        "pair_score_scale": 1.0,
    },
)
```

The helper checks parameter declarations, construction, loss and gradient
behavior, edge-score shape and finiteness, result metadata, and unintended
mutation of the supplied graph. Its own contract checks raise `AssertionError`;
exceptions raised by adapter methods are allowed to propagate with their
original type and context. The helper has no dependency on a particular test
framework.
