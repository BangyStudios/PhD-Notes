## Definition
The **orthogonal projection** of a vector $x \in \mathbb{R}^n$ onto a unit vector $u \in \mathbb{R}^n$ ($\|u\| = 1$) is the component of $x$ in the direction of $u$:
$$
\text{proj}_u(x) = (u^T x)\, u
$$
where $u^T x \in \mathbb{R}$ is the scalar magnitude of that component.

The **residual** — the part of $x$ perpendicular to $u$ — is:
$$
x_\perp = x - (u^T x)\, u
$$

### Example in $\mathbb{R}^2$
Let $u = \begin{bmatrix}1\\0\end{bmatrix}$ and $x = \begin{bmatrix}3\\4\end{bmatrix}$.
The projection onto $u$ (the $x$-axis):
$$
\text{proj}_u(x) = (u^T x)\, u = 3 \begin{bmatrix}1\\0\end{bmatrix} = \begin{bmatrix}3\\0\end{bmatrix}
$$
The residual perpendicular to $u$:
$$
x_\perp = \begin{bmatrix}3\\4\end{bmatrix} - \begin{bmatrix}3\\0\end{bmatrix} = \begin{bmatrix}0\\4\end{bmatrix}
$$
Visually,
![](app://60c4849b146753bbb797ba455dff7454a877/Users/ICB/ICBDrive/Data/Documents/School/2025-PhD/Notes/Topics/Math/Linear%20Algebra/Attachments/output-2.png?1777951569093)
This residual is exactly the formula used in [[Hyperplane]] projection, where $u = w$ is the hyperplane's normal and $x_\perp$ is the projected-onto-hyperplane vector.