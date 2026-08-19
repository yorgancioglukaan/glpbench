## Asymmetric Graph Auto-Encoder Model for Link Prediction

Learns latent representation $Z \in R^{n \times n}$, such that:  

$$Z.X_{reduced}^{T} = \hat{A}$$

where $\hat{A}$ is the reconstructed adjacency matrix and $X_{reduced}$ is the dimensionality-reduced feature matrix.
