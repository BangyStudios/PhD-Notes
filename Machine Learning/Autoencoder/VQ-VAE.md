---
tags:
  - needs-review
  - ai-suspected
ai-review-score: 13.1
ai-review-flagged: 2026-10-07
---
> [!warning] Under review
> Flagged by an AI-text heuristic (score 13.1/100; note last modified 2026-07-02) as possibly AI-written. Verify the content and rewrite in your own words. When done, delete this callout and the `needs-review` and `ai-suspected` tags.

A Vector Quantized Variational Autoencoder (VQ-VAE) is a [VAE](VAE.md) variant that replaces the continuous latent space with a discrete codebook, enabling crisp reconstructions and learning of discrete representations without posterior collapse.

---
## Definition
A VQ-VAE trains an $\text{Encoder}$, a codebook $\{e_i\}_{i=1}^N \subset \mathbb{R}^d$, and a $\text{Decoder}$ via three loss terms:
$$
\mathcal{L} = \underbrace{\|\mathbf{x} - \text{Decoder}(\mathbf{z}_q)\|^2}_{\text{reconstruction}} + \underbrace{\|\mathrm{sg}[\mathbf{z}_e] - \mathbf{z}_q\|^2}_{\text{codebook}} + \underbrace{\beta\,\|\mathbf{z}_e - \mathrm{sg}[\mathbf{z}_q]\|^2}_{\text{commitment}}
$$
where $\mathrm{sg}[\cdot]$ is the stop-gradient operator (used so gradients can be "straight-through" copied without differentiating through the non-differentiable nearest-neighbor lookup). **Vector quantization** maps each encoder output to its nearest codebook entry:
$$
\mathbf{z}_q = e_{i^*}, \qquad i^* = \arg\min_{k}\,\|\mathbf{z}_e - e_i\|_2
$$
Because $\arg\min$ has no gradient, a **straight-through estimator** copies gradients from $\mathbf{z}_q$ back to $\mathbf{z}_e$ during the backward pass. Where:
- **Encoder $\text{Encoder}(\mathbf{x}) = \mathbf{z}_e$:** Maps the input to a continuous pre-quantized representation.
- **Codebook $\{e_i\}$:** A learned embedding table; the $N$ entries are updated by the codebook loss.
- **Decoder $\text{Decoder}(\mathbf{z}_q)$:** Reconstructs the input from the discrete code $i^*$.
- **Commitment loss $\beta$:** Prevents the encoder from oscillating between codebook entries; typically $\beta = 0.25$.
---
## Example
Two-entry codebook, $d = 2$. Codebook: $e_1 = [1, 0]^\top$, $e_2 = [0, 1]^\top$.
### Step 1: Encode
$$
\mathbf{z}_e = \text{Encoder}(\mathbf{x}) = [0.8,\; 0.3]^\top
$$
### Step 2: Quantize
$$
\|z_e - e_1\|^2 = (0.8-1)^2 + (0.3-0)^2 = 0.04 + 0.09 = 0.13
$$
$$
\|z_e - e_2\|^2 = (0.8-0)^2 + (0.3-1)^2 = 0.64 + 0.49 = 1.13
$$
$$
i^* = 1, \qquad \mathbf{z}_q = e_1 = [1,\; 0]^\top
$$
### Step 3: Decode
$$
\hat{\mathbf{x}} = \text{Decoder}(\mathbf{z}_q)
$$
### Step 4: Loss
$$
\mathcal{L}_{\text{codebook}} = \|\mathrm{sg}[\mathbf{z}_e] - \mathbf{z}_q\|^2 = \|[0.8, 0.3] - [1, 0]\|^2 = 0.13
$$
$$
\mathcal{L}_{\text{commit}} = \beta\,\|\mathbf{z}_e - \mathrm{sg}[\mathbf{z}_q]\|^2 = 0.25 \times 0.13 = 0.0325
$$
The codebook loss pulls $e_1$ toward $\mathbf{z}_e$; the commitment loss pulls $\mathbf{z}_e$ toward $e_1$. Gradients for $\mathbf{z}_e$ are copied straight through from $\mathbf{z}_q$ to update $\text{Encoder}(\mathbf{x})$.