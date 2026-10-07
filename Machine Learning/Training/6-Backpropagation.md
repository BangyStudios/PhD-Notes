---
tags:
  - needs-review
  - ai-suspected
ai-review-score: 16.5
ai-review-flagged: 2026-10-07
---
> [!warning] Under review
> Flagged by an AI-text heuristic (score 16.5/100; note last modified 2026-05-08) as possibly AI-written. Verify the content and rewrite in your own words. When done, delete this callout and the `needs-review` and `ai-suspected` tags.

Backpropagation computes the gradient of the loss $\mathcal{L}$ with respect to every parameter $\theta$ by applying the **chain rule** backwards through the computation graph built during the forward pass.

---
## Definition

We express the loss as a function of all layer parameters:
$$
\mathcal{L} = \mathcal{L}(\theta^{(1)}, \theta^{(2)}, \dots, \theta^{(L)})
$$ where:
$$
\theta^{(\ell)} = \{W^{(\ell)}, b^{(\ell)}\}
$$
For each layer $\ell = 1, \dots, L$, recall from the [forward pass](4-Forward_Pass.md):
$$
\begin{aligned}
z^{(\ell)} &= W^{(\ell)} h^{(\ell-1)} + b^{(\ell)} \\
h^{(\ell)} &= \sigma^{(\ell)}\!\big(z^{(\ell)}\big)
\end{aligned}
$$
### Continuous Formulation (Chain Rule)
The gradient with respect to parameters at layer $\ell$ is:
$$
\frac{\partial \mathcal{L}}{\partial \theta^{(\ell)}} =
\frac{\partial \mathcal{L}}{\partial h^{(\ell)}} \cdot
\frac{\partial h^{(\ell)}}{\partial \theta^{(\ell)}}
$$ where:
$$
\frac{\partial \mathcal{L}}{\partial h^{(\ell)}} =
\frac{\partial \mathcal{L}}{\partial h^{(L)}} \cdot
\frac{\partial h^{(L)}}{\partial h^{(L-1)}} \cdot
\frac{\partial h^{(L-1)}}{\partial h^{(L-2)}} \cdots
\frac{\partial h^{(\ell+1)}}{\partial h^{(\ell)}}
$$ where:
$$
\frac{\partial h^{(k)}}{\partial h^{(k-1)}} =
\sigma'^{(k)}\!\big(z^{(k)}\big)\, W^{(k)}
\quad \text{for } k = \ell+1, \dots, L
$$

Alternatively:
$$
\frac{\partial h^{(\ell)}}{\partial W^{(\ell)}} =
\sigma'^{(\ell)}\!\big(z^{(\ell)}\big)\,(h^{(\ell-1)})^{T}
\qquad
\frac{\partial h^{(\ell)}}{\partial b^{(\ell)}} =
\sigma'^{(\ell)}\!\big(z^{(\ell)}\big)
$$
### Discrete Formulation
Consider the identity of the error signal:
$$
\delta^{(\ell)} \;=\; \frac{\partial \mathcal{L}}{\partial z^{(\ell)}}
$$
For the output layer:
$$
\delta^{(L)} =
\frac{\partial \mathcal{L}}{\partial h^{(L)}} \;\odot\;
\sigma'^{(L)}\!\big(z^{(L)}\big)
$$ where:
* $\frac{\partial \mathcal{L}}{\partial h^{(L)}} = \mathcal{L}(h^{(L)}, y)$ is the *base* loss function to be propagated backwards
	* For example (in MSE) $\mathcal{L}(h^{(L)}, y) = \frac{1}{2}\Vert h^{(L)} - y\Vert^2$
* $\sigma'^{(L)}$ is the derivative of the activation function $\sigma$
	* For example (in ReLU) $\sigma'(z) = \begin{cases} 1 & z > 0, \quad \ 0 & z \le 0  \end{cases}$
For each intermediate layer, we propagate the errors:
$$
\boxed{
\begin{aligned}
\delta^{(L-1)} &= \left(W^{(L)}\right)^{T} \delta^{(L)} \;\odot\; \sigma'^{(L-1)}\!\big(z^{(L-1)}\big) \\
\delta^{(L-2)} &= \left(W^{(L-1)}\right)^{T} \delta^{(L-1)} \;\odot\; \sigma'^{(L-2)}\!\big(z^{(L-2)}\big) \\
&\;\;\vdots \\
\delta^{(\ell)} &= \left(W^{(\ell+1)}\right)^{T} \delta^{(\ell+1)} \;\odot\; \sigma'^{(\ell)}\!\big(z^{(\ell)}\big) \\
&\;\;\vdots \\
\delta^{(1)} &= \left(W^{(2)}\right)^{T} \delta^{(2)} \;\odot\; \sigma'^{(1)}\!\big(z^{(1)}\big)
\end{aligned}
}
$$ where (for reference), these are the relevant identities: 
* $\frac{\partial \mathcal{L}}{\partial h^{(\ell)}} = \left(W^{(\ell+1)}\right)^{T} \delta^{(\ell+1)}$
* $\delta^{(\ell)} = \frac{\partial \mathcal{L}}{\partial h^{(\ell)}} \;\odot\; \sigma'^{(\ell)}\!\big(z^{(\ell)}\big)$
Parameter gradients for weights:
$$
\boxed{
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial W^{(L)}} &= \delta^{(L)} (h^{(L-1)})^{T} \\
\frac{\partial \mathcal{L}}{\partial W^{(L-1)}} &= \delta^{(L-1)} (h^{(L-2)})^{T} \\
&\;\;\vdots \\
\frac{\partial \mathcal{L}}{\partial W^{(\ell)}} &= \delta^{(\ell)} (h^{(\ell-1)})^{T} \\
&\;\;\vdots \\
\frac{\partial \mathcal{L}}{\partial W^{(1)}} &= \delta^{(1)} (h^{(0)})^{T}
\end{aligned}
}
$$
And for biases:
$$
\boxed{
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial b^{(L)}} &= \delta^{(L)} \\
\frac{\partial \mathcal{L}}{\partial b^{(L-1)}} &= \delta^{(L-1)} \\
&\;\;\vdots \\
\frac{\partial \mathcal{L}}{\partial b^{(\ell)}} &= \delta^{(\ell)} \\
&\;\;\vdots \\
\frac{\partial \mathcal{L}}{\partial b^{(1)}} &= \delta^{(1)}
\end{aligned}
}
$$

---
## Example
### Setup
Continuing from [4-Forward_Pass](4-Forward_Pass.md) and [5-Loss_Function](5-Loss_Function.md), using simplified cross-entropy loss:
$$
x = \begin{bmatrix}1\\0\end{bmatrix},\quad
z^{(1)} = \begin{bmatrix}1\\0\end{bmatrix},\quad
h^{(1)} = \begin{bmatrix}1\\0\end{bmatrix},\quad
\hat{y} \approx \begin{bmatrix}0.119\\0.881\end{bmatrix},\quad
y = \begin{bmatrix}0\\1\end{bmatrix}
$$
$$
W^{(1)} = \begin{bmatrix}1&0\\0&1\end{bmatrix}, \qquad W^{(2)} = \begin{bmatrix}1&2\\3&4\end{bmatrix}
$$
### Step 1: Output layer error signal
$$
\delta^{(2)} = \hat{y} - y = \begin{bmatrix}0.119\\0.881\end{bmatrix} - \begin{bmatrix}0\\1\end{bmatrix} = \begin{bmatrix}0.119\\-0.119\end{bmatrix}
$$
The symmetric values $\pm 0.119$ arise because softmax output sums to 1, so any increase in one class probability exactly decreases the other.
### Step 2: Gradients for $W^{(2)}$ and $b^{(2)}$
$$
\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \delta^{(2)}\!\left(h^{(1)}\right)^T = \begin{bmatrix}0.119\\-0.119\end{bmatrix}\begin{bmatrix}1&0\end{bmatrix} = \begin{bmatrix}0.119&0\\-0.119&0\end{bmatrix}
$$
$$
\frac{\partial \mathcal{L}}{\partial b^{(2)}} = \delta^{(2)} = \begin{bmatrix}0.119\\-0.119\end{bmatrix}
$$
The second column of $\partial \mathcal{L}/\partial W^{(2)}$ is zero because $h^{(1)}_2 = 0$ — those weights have no influence on the output for this input.
### Step 3: Propagate error signal to layer 1
Route $\delta^{(2)}$ back through $W^{(2)}$:
$$
\left(W^{(2)}\right)^T \delta^{(2)} = \begin{bmatrix}1&3\\2&4\end{bmatrix}\begin{bmatrix}0.119\\-0.119\end{bmatrix} = \begin{bmatrix}0.119 - 0.357\\0.238 - 0.476\end{bmatrix} = \begin{bmatrix}-0.238\\-0.238\end{bmatrix}
$$
Apply the ReLU derivative, $\sigma'^{(1)}(z) = \mathbf{1}[z > 0]$, at $z^{(1)} = [1,\, 0]^T$:
$$
\sigma'^{(1)}\!\left(z^{(1)}\right) = \begin{bmatrix}\mathbf{1}[1>0]\\\mathbf{1}[0>0]\end{bmatrix} = \begin{bmatrix}1\\0\end{bmatrix}
$$
$$
\delta^{(1)} = \begin{bmatrix}-0.238\\-0.238\end{bmatrix} \odot \begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}-0.238\\0\end{bmatrix}
$$
The second unit receives zero error signal because $z^{(1)}_2 = 0$ is on the ReLU boundary — gradient is zero by convention, so no signal passes through this unit.
### Step 4: Gradients for $W^{(1)}$ and $b^{(1)}$
$$
\frac{\partial \mathcal{L}}{\partial W^{(1)}} = \delta^{(1)} x^T = \begin{bmatrix}-0.238\\0\end{bmatrix}\begin{bmatrix}1&0\end{bmatrix} = \begin{bmatrix}-0.238&0\\0&0\end{bmatrix}
$$
$$
\frac{\partial \mathcal{L}}{\partial b^{(1)}} = \delta^{(1)} = \begin{bmatrix}-0.238\\0\end{bmatrix}
$$
### Discussion
- $\partial\mathcal{L}/\partial W^{(2)}_{11} > 0$: increasing this weight raises the class-0 probability, which increases loss — so it should decrease.
- $\partial\mathcal{L}/\partial W^{(2)}_{21} < 0$: increasing this weight raises the class-1 probability, which decreases loss — so it should increase.
- The entire second row of $\partial\mathcal{L}/\partial W^{(1)}$ is zero, and the second column of $\partial\mathcal{L}/\partial W^{(2)}$ is zero. This is a compound dead zone: $h^{(1)}_2 = 0$ (from ReLU at boundary) means these weights contribute nothing forward or backward for this input.
- These gradients are consumed by the [optimization step](7-Optimization_Step.md).
