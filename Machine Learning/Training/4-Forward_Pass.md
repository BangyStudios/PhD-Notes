---
tags:
  - needs-review
  - ai-suspected
ai-review-score: 18.9
ai-review-flagged: 2026-10-07
---
> [!warning] Under review
> Flagged by an AI-text heuristic (score 18.9/100; note last modified 2026-05-08) as possibly AI-written. Verify the content and rewrite in your own words. When done, delete this callout and the `needs-review` and `ai-suspected` tags.

A forward pass is the computation of a prediction $\hat{y}$ from an input $x$ by passing data through each layer of the model sequentially.

---
## Definition
### Continuous Formulation
For a network with $L$ layers parameterized by $\theta = \{W^{(\ell)}, b^{(\ell)}\}_{\ell=1}^{L}$, the forward pass computes a composed function $\hat{y} = f(x;\,\theta)$:
$$
h^{(0)} = x, \qquad h^{(\ell)} = \sigma^{(\ell)}\!\left(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\right), \qquad \hat{y} = h^{(L)}
$$
- **$x \in \mathbb{R}^{d_{\text{in}}}$:** Input feature vector.
- **$h^{(\ell)} \in \mathbb{R}^{d_\ell}$:** Activation at layer $\ell$ (the output of that layer).
- **$W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$:** Learned weight matrix at layer $\ell$.
- **$b^{(\ell)} \in \mathbb{R}^{d_\ell}$:** Learned bias vector at layer $\ell$.
- **$\sigma^{(\ell)}$:** Non-linear activation function at layer $\ell$ (e.g. ReLU, softmax).
- **$\hat{y}$:** Model prediction (output of the final layer).
### Discrete / Library Formulation
In practice (e.g. PyTorch), the forward pass explicitly separates the **pre-activation** $z^{(\ell)}$ from the **post-activation** $h^{(\ell)}$:
$$
z^{(\ell)} = W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}, \qquad h^{(\ell)} = \sigma^{(\ell)}\!\left(z^{(\ell)}\right)
$$
- **$z^{(\ell)} \in \mathbb{R}^{d_\ell}$:** Pre-activation at layer $\ell$ — the result of the linear transformation before the non-linearity is applied.
This separation is retained because [backpropagation](6-Backpropagation.md) requires $z^{(\ell)}$ to evaluate $\sigma'^{(\ell)}$. PyTorch stores these in a **computation graph** automatically during the forward pass.

---
## Example
### Setup
Consider a 2-layer network classifying a 2-dimensional input into 2 classes.
$$
x = \begin{bmatrix}1 \\ 0\end{bmatrix}, \quad
W^{(1)} = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}, \quad
b^{(1)} = \begin{bmatrix}0 \\ 0\end{bmatrix}, \quad
W^{(2)} = \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}, \quad
b^{(2)} = \begin{bmatrix}0 \\ 0\end{bmatrix}
$$

Layer 1 uses **ReLU**: $\sigma^{(1)}(z) = \max(0,\, z)$ applied element-wise.  
Layer 2 uses **softmax**: $\sigma^{(2)}(z)_k = e^{z_k} \big/ \sum_j e^{z_j}$.

---
### Step 1: Layer 1 pre-activation
$$
z^{(1)} = W^{(1)} x + b^{(1)} = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}\begin{bmatrix}1 \\ 0\end{bmatrix} = \begin{bmatrix}1 \\ 0\end{bmatrix}
$$

---
### Step 2: Layer 1 activation (ReLU)
$$
h^{(1)} = \mathrm{ReLU}\!\left(z^{(1)}\right) = \begin{bmatrix}\max(0,\,1) \\ \max(0,\,0)\end{bmatrix} = \begin{bmatrix}1 \\ 0\end{bmatrix}
$$
Note: $z^{(1)}_2 = 0$ is a boundary case for ReLU — the function is non-differentiable here. By convention, most libraries set $\mathrm{ReLU}'(0) = 0$.

---
### Step 3: Layer 2 pre-activation
$$
z^{(2)} = W^{(2)} h^{(1)} + b^{(2)} = \begin{bmatrix}1 & 2 \\ 3 & 4\end{bmatrix}\begin{bmatrix}1 \\ 0\end{bmatrix} = \begin{bmatrix}1 \\ 3\end{bmatrix}
$$

---
### Step 4: Layer 2 activation (softmax)
$$
\hat{y} = \mathrm{softmax}\!\left(z^{(2)}\right) = \frac{1}{e^1 + e^3}\begin{bmatrix}e^1 \\ e^3\end{bmatrix} \approx \frac{1}{22.81}\begin{bmatrix}2.72 \\ 20.09\end{bmatrix} \approx \begin{bmatrix}0.119 \\ 0.881\end{bmatrix}
$$

---
### Interpretation
- The model assigns probability $\approx 0.119$ to class 0 and $\approx 0.881$ to class 1.
- The intermediate values $z^{(1)}, h^{(1)}, z^{(2)}$ are retained in the computation graph so that gradients can be propagated backwards during [backpropagation](6-Backpropagation.md).
- This example continues in [5-Loss_Function](5-Loss_Function.md).
