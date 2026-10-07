From parameters to gradients is the geometric core of training: parameters choose a decision boundary, the loss measures boundary quality, and gradients tell us how to move parameters to improve that boundary.

---
## Core map: parameters -> geometry -> gradients
For a model $\hat{y} = f(x;\theta)$ with loss $\mathcal{L}(\theta)$:
$$
\theta \in \mathbb{R}^{p}
\xrightarrow[]{\text{forward}}
\hat{y}
\xrightarrow[]{\text{loss}}
\mathcal{L}
\xrightarrow[]{\text{backward}}
\nabla_{\theta}\mathcal{L}
$$
- **$x$:** Input feature vector.
- **$\hat{y}$:** Model prediction.
- **$\theta$:** All trainable parameters (weights and biases).
- **$p$:** Number of trainable scalar parameters.
- **$\mathcal{L}(\theta)$:** Scalar loss induced by current parameters.
- **$\nabla_\theta\mathcal{L}$:** Gradient vector; local direction of steepest increase of loss.

The update in [7-Optimization_Step](../Training/7-Optimization_Step.md) is:
$$
\theta_{t+1} = \theta_t - \eta\,\nabla_\theta\mathcal{L}(\theta_t)
$$
- **$t$:** Training step index.
- **$\eta$:** Learning rate.

---
## Visual intuition
### 1) Parameters and loss hypersurface
Treat each parameter as one axis in parameter space. Then $\mathcal{L}(\theta)$ is a hypersurface over that space.
- In 2 parameters ($\theta = (\theta_1,\theta_2)$), imagine a height map.
- Gradient is the local uphill arrow; negative gradient is downhill.
- Flat regions: small gradient, slow movement.
- Sharp regions: large gradient, unstable if $\eta$ is too large.
### 2) Class separation in input space
Parameters define the separator in input space. For linear binary classification:
$$
z = w^Tx + b, \qquad \hat{y} = \sigma(z)
$$
- **$w \in \mathbb{R}^d$:** Weight vector (orientation of separator).
- **$b \in \mathbb{R}$:** Bias (offset of separator).
- **$\sigma$:** Sigmoid.
- Decision boundary: $z=0$ (a line in 2D, a hyperplane in higher dimensions).

Changing $(w,b)$ rotates/translates the boundary; that changes classification error; that changes $\mathcal{L}$; gradient says how to adjust $(w,b)$.

---
## Elementary low-dimensional examples (with edge cases)
### A) Parameters <-> boundary (2D input, binary classes)
Model:
$$
z = w_1x_1 + w_2x_2 + b
$$
Example parameters:
$$
w = (1,-1), \quad b=0
$$
Boundary is $x_1-x_2=0$.
- Point $(2,1)$ gives $z=1$ (positive side).
- Point $(1,2)$ gives $z=-1$ (negative side).
- Edge case: point $(1,1)$ gives $z=0$, exactly on the boundary (maximally uncertain under sigmoid: $\hat{y}=0.5$).

This generalizes: in $d$ dimensions, $w^Tx+b=0$ is always a hyperplane.
### B) Parameters <-> gradient (single-sample logistic loss)
For one sample $(x,y)$ with $y\in\{0,1\}$:
$$
\mathcal{L} = -\big(y\log\hat{y} + (1-y)\log(1-\hat{y})\big), \quad \hat{y}=\sigma(w^Tx+b)
$$
Gradient forms:
$$
\frac{\partial \mathcal{L}}{\partial w} = (\hat{y}-y)x, \qquad
\frac{\partial \mathcal{L}}{\partial b} = \hat{y}-y
$$
Concrete values: $x=(2,1), y=1, w=(0,0), b=0$.

Then $z=0$, $\hat{y}=0.5$, so:
$$
\frac{\partial \mathcal{L}}{\partial w} = (-0.5)(2,1)=(-1,-0.5),
\qquad
\frac{\partial \mathcal{L}}{\partial b} = -0.5
$$
Update $w \leftarrow w-\eta\,\partial\mathcal{L}/\partial w$ increases $w$ in both coordinates, pushing this positive sample farther into the positive side.

Edge cases this covers:
- If $x=0$, then $\partial\mathcal{L}/\partial w=0$ (weights cannot move from this sample; only bias updates).
- If prediction is already perfect and confident, $\hat{y}\approx y$, then gradients are small.

### C) Hypersurface + non-separable data (why minima can be non-zero)
Suppose two identical points have opposite labels: same $x$, one with $y=0$, one with $y=1$.
- No linear separator can satisfy both simultaneously.
- Loss hypersurface has no zero-loss point.
- Gradients still exist and drive toward the best compromise (minimum non-zero empirical loss).

This generalizes to real noisy datasets: training often converges to low loss, not zero loss.

---
## How this is discretized in real implementations
The math is continuous; training code is discrete.
Background: [Gradient Flow and Euler Discretization](Gradient%20Flow%20&%20Euler%20Discretization.md).
### 1) Time discretization (optimizer steps)
Continuous gradient flow:
$$
\frac{d\theta}{dt} = -\nabla_\theta\mathcal{L}(\theta)
$$
Implemented with discrete steps (explicit Euler):
$$
\theta_{t+1}=\theta_t-\eta\,g_t, \quad g_t \approx \nabla_\theta\mathcal{L}_{\text{batch}}(\theta_t)
$$
- **$g_t$:** Mini-batch gradient estimate.
### 2) Data discretization (mini-batches)
Instead of full-dataset gradient each step, libraries use a batch of size $B$:
$$
g_t = \frac{1}{B}\sum_{i=1}^{B} \nabla_\theta \ell_i
$$
- **$\ell_i$:** Per-sample loss.
- Noise from mini-batches helps exploration but adds variance.
### 3) Numerical discretization (finite precision)
Real code uses finite-precision tensors (FP32, FP16/BF16):
- Very small gradients can underflow.
- Very large activations can overflow.
- Stable implementations (e.g., fused cross-entropy, normalization, gradient scaling) reduce these issues.
### 4) Graph discretization (autodiff ops)
Backprop is applied to a finite computation graph made of elementary ops (matmul, add, activation), not symbolic calculus over continuous functions.

---
## Minimal implementation view
```python
for x, y in loader:
	optimizer.zero_grad()          # Stage 8
	y_hat = model(x)               # Stage 4
	loss = criterion(y_hat, y)     # Stage 5
	loss.backward()                # Stage 6
	optimizer.step()               # Stage 7
```
Each loop is one discrete approximation step of continuous learning dynamics.
