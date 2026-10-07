A generative adversarial network (GAN) is a framework in which a generator $G$ and a discriminator $D$ are trained simultaneously via a minimax game: $G$ tries to fool $D$ into accepting synthetic samples as real, while $D$ tries to distinguish real from generated data.

---
## Definition
Let $p_{\text{data}}(\mathbf{x})$ be the real data distribution and $p_z(\mathbf{z})$ a simple prior (e.g. $\mathcal{N}(0, I)$). The GAN objective is:
$$
\min_G \max_D\; \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}\!\left[\log D(\mathbf{x})\right] + \mathbb{E}_{\mathbf{z} \sim p_z}\!\left[\log\!\left(1 - D(G(\mathbf{z}))\right)\right]
$$
- **$G: \mathbb{R}^{d_z} \to \mathcal{X}$:** Generator; maps latent noise $\mathbf{z}$ to a synthetic sample.
- **$D: \mathcal{X} \to [0,1]$:** Discriminator; outputs the probability that its input is real.
- **Optimal equilibrium:** $G$ recovers $p_{\text{data}}$ and $D(\mathbf{x}) = \tfrac{1}{2}$ everywhere.

Training alternates between:
1. **Update $D$:** maximize $\log D(\mathbf{x}) + \log(1 - D(G(\mathbf{z})))$ for a batch of real $\mathbf{x}$ and generated $G(\mathbf{z})$.
2. **Update $G$:** minimize $\log(1 - D(G(\mathbf{z})))$, equivalently maximize $\log D(G(\mathbf{z}))$ (the non-saturating form used in practice).

---
## Example
Scalar, one training step. Let $D(x) = \sigma(w_D x)$ with $w_D = 1$, and $G(z) = w_G z$ with $w_G = 0.5$.

Real sample $x = 1$, latent $z = 2$ so $G(z) = 1$.
### Discriminator outputs
$$
D(x_{\text{real}}) = \sigma(1 \cdot 1) \approx 0.73, \qquad D(G(z)) = \sigma(1 \cdot 1) \approx 0.73
$$
### Discriminator loss (to be maximized)
$$
\mathcal{L}_D = \log(0.73) + \log(1 - 0.73) \approx -0.31 + (-1.31) = -1.62
$$
$D$ cannot distinguish real from generated here because $G$ happened to map $z=2$ to $x=1$. Updating $w_D$ via the gradient of $\mathcal{L}_D$ will push $D$ to assign different scores.
### Generator loss (non-saturating, to be minimized)
$$
\mathcal{L}_G = -\log D(G(z)) = -\log(0.73) \approx 0.31
$$

Updating $w_G$ via $\nabla_{w_G} \mathcal{L}_G$ adjusts $G$ to produce samples $D$ scores higher.
