---
tags:
  - needs-review
  - ai-suspected
ai-review-score: 27.6
ai-review-flagged: 2026-10-07
---
> [!warning] Under review
> Flagged by an AI-text heuristic (score 27.6/100; note last modified 2026-05-08) as possibly AI-written. Verify the content and rewrite in your own words. When done, delete this callout and the `needs-review` and `ai-suspected` tags.

A loss function $\mathcal{L}$ is a scalar measure of how far the model's predictions $\hat{y}$ are from the true targets $y$, providing the training signal for [backpropagation](6-Backpropagation.md).

---
## Definition

### Mean Squared Error (Regression)
**Continuous (population) form** — expected squared error over the data distribution $p(x, y)$:
$$
\mathcal{L}_{\mathrm{MSE}} = \mathbb{E}_{(x,y)\sim p}\!\left[\|\,y - \hat{y}\,\|^2\right]
$$
**Discrete (empirical) form** — averaged over a mini-batch of $N$ samples:
$$
\mathcal{L}_{\mathrm{MSE}} = \frac{1}{N}\sum_{i=1}^{N}\|\,y_i - \hat{y}_i\,\|^2
$$
### Cross-Entropy (Classification)
**Continuous (population) form** — negative expected log-likelihood:
$$
\mathcal{L}_{\mathrm{CE}} = -\mathbb{E}_{(x,y)\sim p}\!\left[\sum_{k=1}^{C} y_k \log \hat{y}_k\right]
$$
**Discrete (empirical) form** — averaged over a mini-batch, where $y_i \in \{0,1\}^C$ is a one-hot label and $\hat{y}_i \in (0,1)^C$ is a softmax probability vector:
$$
\mathcal{L}_{\mathrm{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{C} y_{ik} \log \hat{y}_{ik}
$$
- **$N$:** Number of samples in the mini-batch.
- **$C$:** Number of output classes.
- **$y_{ik}$:** Ground-truth indicator — 1 if sample $i$ belongs to class $k$, else 0.
- **$\hat{y}_{ik}$:** Predicted probability of sample $i$ belonging to class $k$.
- **$\log$:** Natural logarithm. Penalises confident wrong predictions heavily: $\log \hat{y}_{ik} \to -\infty$ as $\hat{y}_{ik} \to 0$.
### Numerical Stability
Library implementations (e.g. `nn.CrossEntropyLoss` in PyTorch) combine the log-softmax and cross-entropy into a single operation to avoid numerical instability from computing $\log(\mathrm{softmax}(z))$ in two steps (prone to underflow for large logits). Given logits $z_i \in \mathbb{R}^C$ and true class index $k^*$:
$$
\mathcal{L}_{\mathrm{CE}} = -\frac{1}{N}\sum_{i=1}^{N}\left(z_{i,k^*} - \log\!\sum_{k=1}^{C} e^{z_{ik}}\right)
$$
- **$z_{ik}$:** Raw (pre-softmax) logit for sample $i$ and class $k$.
- **$k^*$:** The index of the true class for sample $i$.
Note: `nn.CrossEntropyLoss` therefore expects **raw logits**, not softmax probabilities.

---
## Example
### Setup
Using the forward pass output from [4-Forward_Pass](4-Forward_Pass.md):
$$
\hat{y} \approx \begin{bmatrix}0.119 \\ 0.881\end{bmatrix}, \qquad y = \begin{bmatrix}0 \\ 1\end{bmatrix}, \qquad z^{(2)} = \begin{bmatrix}1 \\ 3\end{bmatrix}
$$
The true class is **class 1** (index 1, so $k^* = 1$).

---
### Step 1: Apply cross-entropy directly
For a single sample ($N = 1$):
$$
\mathcal{L}_{\mathrm{CE}} = -\sum_{k=1}^{2} y_k \log \hat{y}_k = -\bigl(0 \cdot \log(0.119) + 1 \cdot \log(0.881)\bigr) = -\log(0.881) \approx 0.127
$$

---
### Step 2: Verify via the numerically stable form
Using logits $z^{(2)} = [1,\, 3]$ and true class $k^* = 1$ (zero-indexed as the second entry):
$$
\mathcal{L}_{\mathrm{CE}} = -\left(3 - \log(e^1 + e^3)\right) = -3 + \log(22.81) \approx -3 + 3.127 = 0.127
$$
Both forms agree.

---
### Interpretation
- The loss $\approx 0.127$ is small because the model is already confident ($\hat{y}_1 \approx 0.881$) about the correct class.
- If the model predicted $\hat{y} = [0.5,\, 0.5]$ (maximum uncertainty), the loss would be $-\log(0.5) \approx 0.693$ — a higher penalty reflecting the lack of confidence.
- If the model predicted $\hat{y} = [0.999,\, 0.001]$ (confident and wrong), the loss would be $-\log(0.001) \approx 6.91$ — a severe penalty.
- The scalar $\mathcal{L}$ is passed to [6-Backpropagation](6-Backpropagation.md) to compute parameter gradients.
