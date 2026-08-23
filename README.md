# Asymmetric Graph Autoencoder Link Prediction

This project compares conventional, asymmetric, MLP-decoder, and variational graph autoencoders for link prediction. The asymmetric decoder combines latent source embeddings with reduced target-node features, while the baselines provide inner-product, learned symmetric, and probabilistic alternatives. The asymmetric model learns latent representation $Z \in R^{n \times n}$, such that:

$$Z.X_{reduced}^{T} = \hat{A}$$

where $\hat{A}$ is the reconstructed adjacency matrix and $X_{reduced}$ is the dimensionality-reduced feature matrix.


Models are evaluated on transductive splits (edge-uniform), where edges are held out, and inductive splits (Cold-start), where all edges incident to selected query nodes are held out to simulate cold-start prediction.

## Running an experiment

Run an experiment with a supported Planetoid dataset:

```bash
python main.py cora out/results.csv
```

When `--model-type` is omitted, the script trains every registered architecture on the same splits. The four current architectures produce eight result rows under one run ID after evaluation on both test settings.

Add `--verbose` to show dataset preparation, architecture progress, epoch progress, and live loss components:

```bash
python main.py cora out/results.csv --verbose
```

Common options include:

```bash
python main.py citeseer out/results.csv \
  --conv-layer-type GATv2Conv \
  --hidden-channels 128 \
  --out-channels 64 \
  --reduction-method pca \
  --decoder-hidden-channels 64 \
  --decoder-dropout 0.2 \
  --kl-weight 1.0 \
  --query-size 10 \
  --train-ratio 0.8 \
  --epochs 100 \
  --seed 42 \
  --device cpu
```

Available model types are `symmetric`, `asymmetric`, `mlp`, and `variational`. Repeat `--model-type` to choose several explicitly:

```bash
python main.py cora \
  --model-type mlp \
  --model-type variational
```

Supported datasets are Cora, CiteSeer, PubMed, Amazon Photo, Amazon Computers, Coauthor CS, DBLP, WikiCS, and Deezer Europe. Supported convolution layers are `GCNConv`, `GraphConv`, `SAGEConv`, and `GATv2Conv`. Run `python main.py --help` for the complete option list. Newly registered architectures will automatically become available to the CLI.

For example, run the Amazon Photo benchmark with:

```bash
python main.py amazon-photo out/results.csv --verbose
```

## API overview

### Dataset loading

`datasets.load_dataset()` loads a supported benchmark, optionally normalizes its features, moves it to a device, and attaches dataset metadata:

```python
from datasets import load_dataset

data = load_dataset("cora", root="data", device="cpu")
```

`datasets.available_datasets()` returns the normalized names accepted by the loader.

### Features and graph splits

`utils.calculate_reduced_features()` fits a registered feature reducer and stores the transformed features in `data.x_reduced`. Pass a node mask to fit without exposing inductive query features:

```python
import torch

from utils import calculate_reduced_features, generate_comparable_splits

fit_mask = torch.ones(data.num_nodes, dtype=torch.bool)
fit_mask[[10, 25, 40]] = False
calculate_reduced_features(data, dim=64, fit_nodes=fit_mask)
train, val, test_transductive, test_inductive = generate_comparable_splits(
    data,
    query=[10, 25, 40],
    train_ratio=0.8,
)
```

`generate_comparable_splits()` creates transductive and inductive test sets with matching message-passing graphs and equal numbers of labeled edges. `utils.inductive_split()` is available when only a cold-start split is needed.

### Link-prediction models

The model classes are:

- `GAE_Model`: inner-product symmetric graph autoencoder.
- `AGAE_Model`: asymmetric latent-to-feature decoder.
- `MLP_GAE_Model`: learned symmetric MLP edge decoder.
- `VGAE_Model`: variational encoder with inner-product decoding and KL regularization.

`link_prediction.LinkPrediction` provides the shared training and inference interface:

```python
from link_prediction import LinkPrediction

predictor = LinkPrediction(
    train,
    model_type="asymmetric",
    conv_layer_type="SAGEConv",
    hidden_channels=128,
    out_channels=64,
)

for _ in range(100):
    loss = predictor.train()

probabilities = predictor.test(test_inductive)
```

Configure the MLP and variational baselines with `decoder_hidden_channels`, `decoder_dropout`, and `kl_weight`:

```python
mlp = LinkPrediction(
    train,
    model_type="mlp",
    decoder_hidden_channels=64,
    decoder_dropout=0.2,
)
vgae = LinkPrediction(train, model_type="variational", kl_weight=1.0)
```

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

`experiment_logging.ExperimentResult` validates experiment metadata and metrics. `CSVResultLogger` appends results to a stable CSV schema; rows written by the same logger share a run ID:

```python
from experiment_logging import CSVResultLogger, ExperimentResult

logger = CSVResultLogger("out/results.csv")
logger.log(ExperimentResult(...))
logger.reset_run_id()  # Start a new experiment group.
```
