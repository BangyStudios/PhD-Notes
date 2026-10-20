A Variational Autoencoder (VAE) is a generative model that learns to encode data into a structured latent space and decode it back, while regularizing that latent space with a prior distribution. It uses [KL Divergence](KL%20Divergence.md) as the regularization term.

---
## Definition

A VAE jointly trains an **encoder** $q_\phi(\mathbf{z}\mid\mathbf{x})$ and a **decoder** $p_\theta(\mathbf{x}\mid\mathbf{z})$ by maximizing the **Evidence Lower Bound (ELBO)**:

$$
\mathcal{L}(\phi, \theta;\mathbf{x}) = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}\mid\mathbf{x})}\!\left[\log p_\theta(\mathbf{x}\mid\mathbf{z})\right]}_{\text{reconstruction loss}} - \underbrace{D_{\mathrm{KL}}\!\left(q_\phi(\mathbf{z}\mid\mathbf{x})\,\|\,p(\mathbf{z})\right)}_{\text{regularization}}
$$

- **Encoder** $q_\phi(\mathbf{z}\mid\mathbf{x})$: approximates the true (intractable) posterior; typically outputs a mean $\mu$ and log-variance $\log\sigma^2$ for a Gaussian.
- **Decoder** $p_\theta(\mathbf{x}\mid\mathbf{z})$: reconstructs data from a latent sample $\mathbf{z}$.
- **Prior** $p(\mathbf{z})$: usually $\mathcal{N}(0, I)$; keeps the latent space smooth and continuous.

### Reparameterization Trick

Sampling $\mathbf{z} \sim q_\phi(\mathbf{z}\mid\mathbf{x})$ is not differentiable. The **reparameterization trick** reformulates it as:

$$
\mathbf{z} = \mu + \sigma \odot \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0, I)
$$

This separates the stochasticity ($\varepsilon$) from the learned parameters ($\mu, \sigma$), allowing gradients to flow through $\mu$ and $\sigma$.

---
## Example

Given an image $\mathbf{x}$:

1. **Encode**: $\mu, \log\sigma^2 = \mathrm{Encoder}_\phi(\mathbf{x})$, e.g. $\mu = [0.3, -1.2]$, $\sigma = [0.8, 0.5]$.
2. **Sample**: $\varepsilon \sim \mathcal{N}(0, I)$, then $\mathbf{z} = \mu + \sigma \odot \varepsilon$.
3. **Decode**: $\hat{\mathbf{x}} = \mathrm{Decoder}_\theta(\mathbf{z})$.
4. **Loss**: penalise reconstruction error + $D_{\mathrm{KL}}$ pushing $q_\phi$ toward $\mathcal{N}(0,I)$.
