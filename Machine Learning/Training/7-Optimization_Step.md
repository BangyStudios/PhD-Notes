An optimization step updates the model parameters $\theta$ using the gradients $\nabla_\theta \mathcal{L}$ computed during [backpropagation](6-Backpropagation.md), moving $\theta$ in a direction that reduces the loss.

---
## Definition
Trivially recall (*ex nilho* from your big brain) that:
$$
\begin{aligned}
\nabla_\theta\mathcal{L}(\theta) &= 
\left\{
\frac{\partial\mathcal{L}}{\partial W^{(1)}}, 
\frac{\partial\mathcal{L}}{\partial b^{(1)}},
...,
\frac{\partial\mathcal{L}}{\partial W^{(L)}}, 
\frac{\partial\mathcal{L}}{\partial b^{(L)}}
\right\} \\
-\nabla_\theta\mathcal{L}(\theta) &= 
\left\{
-\frac{\partial\mathcal{L}}{\partial W^{(1)}}, 
-\frac{\partial\mathcal{L}}{\partial b^{(1)}},
...,
-\frac{\partial\mathcal{L}}{\partial W^{(L)}}, 
-\frac{\partial\mathcal{L}}{\partial b^{(L)}}
\right\}
\end{aligned}
$$
### Stochastic Gradient Descent (SGD)
**Continuous (gradient flow) formulation** — the continuous-time limit, a differential equation whose solution minimizes $\mathcal{L}$:
$$
\frac{d\theta}{dt} = -\nabla_\theta \mathcal{L}(\theta)
$$
**Discrete formulation** — the practical update rule at step $t$:
$$
\theta_t = \theta_{t-1} - \eta\, \nabla_\theta \mathcal{L}(\theta_{t-1})
$$
- **$\eta > 0$:** Learning rate — controls the step size. Too large causes divergence; too small causes slow convergence.
- **$\nabla_\theta \mathcal{L} \in \mathbb{R}^{|\theta|}$:** Gradient of the loss with respect to parameters, computed by backpropagation. Also written as: $\frac{\partial \mathcal{L}}{\partial W^{(\mathcal{l})}}$
### Adam (Adaptive Moment Estimation)
Adam maintains a running first moment (mean) $m_t$ and second moment (uncentered variance) $v_t$ of the gradients, with bias-correction for the zero initialization:
$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\, g_t
$$
$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2)\, g_t^{\,2}
$$
$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$
$$
\theta_t = \theta_{t-1} - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
$$
- **$g_t = \nabla_\theta \mathcal{L}$:** Gradient at step $t$.
- **$m_t \in \mathbb{R}^{|\theta|}$:** First moment estimate — exponential moving average of past gradients.
- **$v_t \in \mathbb{R}^{|\theta|}$:** Second moment estimate — exponential moving average of past squared gradients.
- **$\beta_1, \beta_2 \in [0,1)$:** Decay rates for the moment estimates. Defaults: $\beta_1 = 0.9$, $\beta_2 = 0.999$.
- **$\varepsilon > 0$:** Small constant for numerical stability, preventing division by zero. Default: $\varepsilon = 10^{-8}$.
- **$\hat{m}_t,\, \hat{v}_t$:** Bias-corrected moments. Required because initializing $m_0 = v_0 = 0$ biases estimates toward zero in early steps.

Adam adapts the effective learning rate per parameter: parameters with large, consistent gradients receive smaller updates; parameters with small or noisy gradients can still make progress.

---
## Example
### Setup
Using the gradients from [backpropagation](6-Backpropagation.md) with learning rate $\eta = 0.1$ (SGD):
$$
\frac{\partial \mathcal{L}}{\partial W^{(2)}} = \begin{bmatrix}0.119&0\\-0.119&0\end{bmatrix}, \qquad \frac{\partial \mathcal{L}}{\partial W^{(1)}} = \begin{bmatrix}-0.238&0\\0&0\end{bmatrix}
$$
Current parameters:
$$
W^{(2)} = \begin{bmatrix}1&2\\3&4\end{bmatrix}, \qquad W^{(1)} = \begin{bmatrix}1&0\\0&1\end{bmatrix}
$$
### Step 1: Update $W^{(2)}$ (SGD)
$$
W^{(2)}_{\text{new}} = W^{(2)} - 0.1 \cdot \frac{\partial \mathcal{L}}{\partial W^{(2)}} = \begin{bmatrix}1&2\\3&4\end{bmatrix} - 0.1\begin{bmatrix}0.119&0\\-0.119&0\end{bmatrix} = \begin{bmatrix}0.988&2\\3.012&4\end{bmatrix}
$$
The weight from hidden unit 1 to class 0 decreased (suppresses class 0), and to class 1 increased (boosts class 1) — both in the direction of the true label.
### Step 2: Update $W^{(1)}$ (SGD)
$$
W^{(1)}_{\text{new}} = \begin{bmatrix}1&0\\0&1\end{bmatrix} - 0.1\begin{bmatrix}-0.238&0\\0&0\end{bmatrix} = \begin{bmatrix}1.024&0\\0&1\end{bmatrix}
$$
Entries with zero gradient are unchanged — no update occurs where the network had no influence.
### Step 3: Adam update (single entry, $t=1$)
For the scalar entry $W^{(2)}_{11}$, with $g_1 = 0.119$, $m_0 = v_0 = 0$, defaults $\beta_1=0.9$, $\beta_2=0.999$, $\varepsilon=10^{-8}$, $\eta=0.001$:
$$
m_1 = 0.9 \cdot 0 + 0.1 \cdot 0.119 = 0.0119
$$
$$
v_1 = 0.999 \cdot 0 + 0.001 \cdot (0.119)^2 = 1.42 \times 10^{-5}
$$
$$
\hat{m}_1 = \frac{0.0119}{1 - 0.9^1} = 0.119, \qquad \hat{v}_1 = \frac{1.42 \times 10^{-5}}{1 - 0.999^1} = 0.0142
$$
$$
W^{(2)}_{11,\,\text{new}} = 1 - 0.001 \cdot \frac{0.119}{\sqrt{0.0142} + 10^{-8}} \approx 1 - 0.001 \cdot \frac{0.119}{0.119} = 1 - 0.001 = 0.999
$$

At step $t=1$, bias correction produces $\hat{m}_1 / \sqrt{\hat{v}_1} \approx 1$ regardless of gradient magnitude, so the first update is always approximately $\eta$ in size — well-behaved even for unusual gradient scales.
### Discussion
- SGD applies a uniform step along the negative gradient direction; Adam scales each parameter's step by the history of its gradient magnitude.
- Parameters with zero gradient (e.g. entire second column of $W^{(1)}$) are never updated — they are not in the computation path for the given input and cannot learn from it.
- Updated parameters are stored and used in the next [forward pass](4-Forward_Pass.md). Gradients are [zeroed](8-Zero_Gradients.md) before the next batch.
