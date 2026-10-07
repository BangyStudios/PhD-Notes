---
tags:
  - needs-review
  - ai-suspected
ai-review-score: 16.9
ai-review-flagged: 2026-10-07
---
> [!warning] Under review
> Flagged by an AI-text heuristic (score 16.9/100; note last modified 2026-10-05) as possibly AI-written. Verify the content and rewrite in your own words. When done, delete this callout and the `needs-review` and `ai-suspected` tags.

A periodic autoencoder is a [variational autoencoder](Variational%20Autoencoder.md) built from circular convolutions, so its decoder generates exactly one period of a signal that tiles seamlessly, optionally under a slew-rate limit.

---
## Definition
The model is an [autoencoder](Autoencoder.md) $(f_\phi, g_\theta)$ with a Gaussian posterior $q_\phi(\mathbf{z}\mid\mathbf{x})$ and two periodic components:

**Circular convolution** replaces the zero (or causal, as in a [TCN](../Common%20Modules/Architectures/TCN.md)) padding of a [convolution](../Convolution/Convolution.md) by wrap-around padding of $\lfloor (k-1)d/2 \rfloor$ samples on each side, so the layer commutes with circular shifts and maps one period to one period.

**Slew-rate limiter** bounds the step between neighboring samples of one period, including the wrap-around step:
$$
\boxed{
\tilde{\Delta}_t = \operatorname{clip}(\Delta_t, -\delta_{\max}, \delta_{\max}) - \tfrac{1}{T}\sum_{s=1}^{T}\operatorname{clip}(\Delta_s, -\delta_{\max}, \delta_{\max}), \qquad \hat{x}_t = x_1 + \sum_{s<t}\tilde{\Delta}_s
}
$$ where:
- **$\Delta_t = x_{t+1} - x_t$:** Step to the next sample, with $x_{T+1} = x_1$.
- **$\delta_{\max}$:** Maximum allowed step, in the units of the signal.
- **$T$:** Samples per period.

The mean subtraction forces $\sum_t \tilde{\Delta}_t = 0$ so the integrated signal closes the period; a final rescale keeps $|\tilde{\Delta}_t| \le \delta_{\max}$ exactly. Gradients flow through every step below the limit.

The encoder halves the length with strided circular convolutions, flattens, and projects to $(\mu, \log\sigma^2)$; the decoder projects $\mathbf{z}$ back, doubles the length with upsampling and circular convolutions, and applies the limiter.

---
## Example
Period $\mathbf{x} = (0, 3, 0, 0)$, $\delta_{\max} = 1$.
### Step 1: Differences
$$
\Delta = (3, -3, 0, 0)
$$
The last entry is $x_1 - x_4 = 0$.
### Step 2: Clip and re-center
$$
\operatorname{clip}(\Delta) = (1, -1, 0, 0), \quad \text{mean} = 0 \;\Rightarrow\; \tilde{\Delta} = (1, -1, 0, 0)
$$
### Step 3: Integrate
$$
\hat{\mathbf{x}} = (0,\; 0 + 1,\; 1 - 1,\; 0 + 0) = (0, 1, 0, 0)
$$
The spike is flattened to the largest step the limit allows, and the period still closes.
