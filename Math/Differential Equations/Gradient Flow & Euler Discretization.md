Gradient descent can be viewed as a discrete approximation of a differential equation.

---
## Continuous view (ODE)

Let $\theta(t)$ be parameters evolving in continuous time:

$$
\frac{d\theta}{dt} = -\nabla_\theta \mathcal{L}(\theta)
$$

- **$\theta(t)$:** Parameter state at time $t$.
- **$\mathcal{L}(\theta)$:** Loss surface over parameter space.
- **$-\nabla_\theta \mathcal{L}$:** Steepest local descent direction.

Interpretation: parameters "flow downhill" on the loss surface.

---
## Discrete view (explicit Euler)

Using step size $\eta > 0$:

$$
\theta_{k+1} = \theta_k + \eta \frac{d\theta}{dt}\bigg|_{\theta_k}
= \theta_k - \eta\,\nabla_\theta \mathcal{L}(\theta_k)
$$

This is exactly the SGD-style update (ignoring mini-batch noise).

- **$k$:** Integer step index.
- **$\eta$:** Time-step size in the discretization (learning rate in ML).

---
## 1D elementary example

Take:
$$
\mathcal{L}(\theta)=\frac{1}{2}\theta^2
\Rightarrow
\nabla_\theta\mathcal{L}=\theta
$$

ODE:
$$
\frac{d\theta}{dt}=-\theta
$$
Exact solution:
$$
\theta(t)=\theta(0)e^{-t}
$$

Euler update:
$$
\theta_{k+1}=(1-\eta)\theta_k
$$

Edge cases:
- $0<\eta<1$: monotone decay.
- $1<\eta<2$: alternating sign but still convergent.
- $\eta\ge2$: unstable/divergent.

This is the simplest stability intuition for learning-rate selection.