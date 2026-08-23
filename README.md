# Asymmetric Graph Autoencoder Link Prediction

This project compares conventional graph autoencoders with asymmetric graph autoencoders for link prediction. The symmetric decoder reconstructs links from latent embeddings, while the asymmetric decoder combines latent source embeddings with reduced target-node features. Asymmetric model learns latent representation $Z \in R^{n \times n}$, such that:  

$$Z.X_{reduced}^{T} = \hat{A}$$

where $\hat{A}$ is the reconstructed adjacency matrix and $X_{reduced}$ is the dimensionality-reduced feature matrix.


Models are evaluated on transductive splits (edge-uniform), where edges are held out, and inductive splits (Cold-start), where all edges incident to selected query nodes are held out to simulate cold-start prediction.

## Running an experiment

Run an experiment with a supported Planetoid dataset:

```bash
python main.py cora out/results.csv
```

When `--model-type` is omitted, the script trains both symmetric and asymmetric models on the same splits. It evaluates each model on the transductive and inductive test sets and writes all four result rows under one run ID.

Common options include:

```bash
python main.py citeseer out/results.csv \
  --conv-layer-type GATv2Conv \
  --hidden-channels 128 \
  --out-channels 64 \
  --query-size 10 \
  --train-ratio 0.8 \
  --epochs 100 \
  --seed 42 \
  --device cpu
```

Use `--model-type symmetric` or `--model-type asymmetric` to run only one architecture. Supported datasets are Cora, CiteSeer, and PubMed; supported convolution layers are `GCNConv`, `GraphConv`, `SAGEConv`, and `GATv2Conv`. Run `python main.py --help` for the complete option list.

## API overview

### Dataset loading

`datasets.load_dataset()` loads a supported benchmark, optionally normalizes its features, moves it to a device, and attaches dataset metadata:

```python
from datasets import load_dataset

data = load_dataset("cora", root="data/Planetoid", device="cpu")
```

`datasets.available_datasets()` returns the normalized names accepted by the loader.

### Features and graph splits

`utils.calculate_reduced_features()` fits PCA and stores the reduced features in `data.x_reduced`:

```python
from utils import calculate_reduced_features, generate_comparable_splits

calculate_reduced_features(data, dim=64)
train, val, test_transductive, test_inductive = generate_comparable_splits(
    data,
    query=[10, 25, 40],
    train_ratio=0.8,
)
```

`generate_comparable_splits()` creates transductive and inductive test sets with matching message-passing graphs and equal numbers of labeled edges. `utils.inductive_split()` is available when only a cold-start split is needed.

### Link-prediction models

`link_prediction.GAE_Model` implements the symmetric autoencoder, and `link_prediction.AGAE_Model` implements the asymmetric decoder. `link_prediction.LinkPrediction` provides the main training and inference interface:

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
