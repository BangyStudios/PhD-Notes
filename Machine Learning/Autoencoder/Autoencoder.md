---
tags:
  - needs-review
  - ai-suspected
ai-review-score: 19.4
ai-review-flagged: 2026-10-07
---
> [!warning] Under review
> Flagged by an AI-text heuristic (score 19.4/100; note last modified 2026-10-05) as possibly AI-written. Verify the content and rewrite in your own words. When done, delete this callout and the `needs-review` and `ai-suspected` tags.

An autoencoder is a neural network trained to reconstruct its input by compressing it through a bottleneck, forcing the network to learn a compact latent representation.

---
## Definition
An autoencoder jointly trains an **encoder** $f_\phi$ and a **decoder** $g_\theta$ by minimizing reconstruction loss:
$$
\mathcal{L}(\phi, \theta) = \frac{1}{N}\sum_{n=1}^{N} \|\mathbf{x}_n - g_\theta(f_\phi(\mathbf{x}_n))\|^2
$$ where:
- **$f_\phi: \mathbb{R}^{d} \to \mathbb{R}^{k}$:** Encoder; maps input $\mathbf{x}$ to latent code $\mathbf{z} = f_\phi(\mathbf{x})$, with $k \ll d$.
- **$g_\theta: \mathbb{R}^{k} \to \mathbb{R}^{d}$:** Decoder; maps $\mathbf{z}$ back to a reconstruction $\hat{\mathbf{x}} = g_\theta(\mathbf{z})$.
- **$k$:** Bottleneck dimension; controls the information capacity of the latent space.

The bottleneck prevents the network from learning an identity mapping, so $\mathbf{z}$ must capture the most salient structure in $\mathbf{x}$. A [VAE](Variational%20Autoencoder.md) extends this by imposing a prior on $\mathbf{z}$ to enable generation, and a [periodic autoencoder](Periodic%20Autoencoder.md) specializes it to periodic signals.

---
## Example
Input $\mathbf{x} \in \mathbb{R}^3$, bottleneck $k = 1$.
$$
\mathbf{x} = \begin{bmatrix}1 \\ 0 \\ 1\end{bmatrix}, \quad
W_\mathrm{enc} = \begin{bmatrix}1 & 0 & 1\end{bmatrix}, \quad
W_\mathrm{dec} = \begin{bmatrix}0.5 \\ 0 \\ 0.5\end{bmatrix}
$$
### Step 1: Encode
$$
\mathbf{z} = f_\phi(\mathbf{x}) = W_\mathrm{enc}\, \mathbf{x} = 1 \cdot 1 + 0 \cdot 0 + 1 \cdot 1 = 2
$$
### Step 2: Decode
$$
\hat{\mathbf{x}} = g_\theta(\mathbf{z}) = W_\mathrm{dec}\, \mathbf{z} = \begin{bmatrix}0.5 \\ 0 \\ 0.5\end{bmatrix} \cdot 2 = \begin{bmatrix}1 \\ 0 \\ 1\end{bmatrix}
$$
### Step 3: Loss
$$
\mathcal{L} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \left\|\begin{bmatrix}0 \\ 0 \\ 0\end{bmatrix}\right\|^2 = 0
$$
Perfect reconstruction here because $W_\theta W_\phi \mathbf{x} = \mathbf{x}$ for this input — in practice the bottleneck forces a lossy compression and the network minimizes the average error across all training samples.
