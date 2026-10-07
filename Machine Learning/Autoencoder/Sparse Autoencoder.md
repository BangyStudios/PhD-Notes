A sparse autoencoder (SAE) is an [autoencoder](Autoencoder.md) that adds an $\ell_1$ penalty to the reconstruction loss, forcing the latent representation to use only a small number of active features for any given input.

---
## Definition
A sparse autoencoder trains an **encoder** $f_\phi$ and a **decoder** $g_\theta$ by minimizing:
$$
\mathcal{L}(\phi, \theta) = \underbrace{\frac{1}{N}\sum_{n=1}^{N} \|\mathbf{x}_n - g_\theta(f_\phi(\mathbf{x}_n))\|^2}_{\text{reconstruction}} + \underbrace{\lambda \frac{1}{N}\sum_{n=1}^{N} \|f_\phi(\mathbf{x}_n)\|_1}_{\text{sparsity}}
$$ where:
- **$f_\phi: \mathbb{R}^{d} \to \mathbb{R}^{k}$:** Encoder; maps input $\mathbf{x}$ to latent code $\mathbf{z} = f_\phi(\mathbf{x})$.
- **$g_\theta: \mathbb{R}^{k} \to \mathbb{R}^{d}$:** Decoder; maps $\mathbf{z}$ back to a reconstruction $\hat{\mathbf{x}} = g_\theta(\mathbf{z})$.
- **$\lambda \geq 0$:** Sparsity coefficient; controls the trade-off between reconstruction fidelity and activation sparsity.
- **$\|\mathbf{z}\|_1 = \sum_i |z_i|$:** $\ell_1$ norm; penalizes the total magnitude of activations, preferring solutions where most $z_i = 0$.

Unlike a standard [autoencoder](Autoencoder.md), SAEs are typically **overcomplete** ($k \gg d$): the sparsity penalty — not a narrow bottleneck — prevents the trivial identity solution. The encoder commonly applies a ReLU activation to produce non-negative codes:
$$
\mathbf{z} = \mathrm{ReLU}(W_\mathrm{enc}\,\mathbf{x} + \mathbf{b}_\mathrm{enc}), \qquad \hat{\mathbf{x}} = W_\mathrm{dec}\,\mathbf{z} + \mathbf{b}_\mathrm{dec}
$$ where:
- **$W_\mathrm{enc} \in \mathbb{R}^{k \times d}$:** Encoder weight matrix.
- **$W_\mathrm{dec} \in \mathbb{R}^{d \times k}$:** Decoder weight matrix; columns are often unit-normalized to prevent the model from absorbing sparsity into small decoder norms.
- **$\mathbf{b}_\mathrm{enc} \in \mathbb{R}^k,\ \mathbf{b}_\mathrm{dec} \in \mathbb{R}^d$:** Bias vectors.

---
## Example
Input $\mathbf{x} \in \mathbb{R}^2$, overcomplete latent $\mathbf{z} \in \mathbb{R}^4$, sparsity coefficient $\lambda = 0.1$, all biases zero.
$$
\mathbf{x} = \begin{bmatrix}1\\0\end{bmatrix}, \quad
W_\mathrm{enc} = \begin{bmatrix}1&0\\0&1\\1&1\\-1&0\end{bmatrix}, \quad
W_\mathrm{dec} = \begin{bmatrix}1&0&0.5&0\\0&1&0.5&0\end{bmatrix}
$$
### Step 1: Encode
$$
W_\mathrm{enc}\,\mathbf{x} = \begin{bmatrix}1\\0\\1\\-1\end{bmatrix}, \qquad \mathbf{z} = \mathrm{ReLU}\!\begin{bmatrix}1\\0\\1\\-1\end{bmatrix} = \begin{bmatrix}1\\0\\1\\0\end{bmatrix}
$$
Two of four units are active; the negative pre-activation in unit 4 is gated to zero by ReLU.
### Step 2: Decode
$$
\hat{\mathbf{x}} = W_\mathrm{dec}\,\mathbf{z} = \begin{bmatrix}1&0&0.5&0\\0&1&0.5&0\end{bmatrix}\begin{bmatrix}1\\0\\1\\0\end{bmatrix} = \begin{bmatrix}1.5\\0.5\end{bmatrix}
$$
### Step 3: Reconstruction loss
$$
\mathcal{L}_\mathrm{recon} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2 = \left\|\begin{bmatrix}-0.5\\-0.5\end{bmatrix}\right\|^2 = 0.5
$$
### Step 4: Sparsity penalty
$$
\|\mathbf{z}\|_1 = 1 + 0 + 1 + 0 = 2, \qquad \mathcal{L}_\mathrm{sparse} = \lambda\|\mathbf{z}\|_1 = 0.1 \times 2 = 0.2
$$
### Step 5: Total loss
$$
\mathcal{L} = 0.5 + 0.2 = 0.7
$$
Reconstruction error dominates here; increasing $\lambda$ would pressure the network to deactivate more units at the cost of higher reconstruction error.
