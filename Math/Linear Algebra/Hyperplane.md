## Definition
A **hyperplane** in $\mathbb{R}^n$ is a flat subspace of dimension $n-1$, defined by a normal vector $w$ and a bias:
$$
\mathcal{H}_w = \{ x \in \mathbb{R}^n \mid w^T x = 0 \}
$$
where $n$ is the dimension of the ambient space, $w \in \mathbb{R}^n$ is a unit normal vector ($\|w\| = 1$) orthogonal to the hyperplane, and $x \in \mathbb{R}^n$ is any point in that space.

The [orthogonal projection](Orthogonal%20Projection.md) of a vector $x$ onto $\mathcal{H}_w$ removes its component along $w$:
$$
x_\perp = x - (w^T x)\, w
$$
where $w^T x \in \mathbb{R}$ is the scalar projection of $x$ onto $w$, and $x_\perp$ is the projected vector lying in $\mathcal{H}_w$.
### Example in $\mathbb{R}^3$
Let $w = \begin{bmatrix}0\\0\\1\end{bmatrix}$ (the $z$-axis). The hyperplane $\mathcal{H}_w$ is the $xy$-plane.

For $x = \begin{bmatrix}2\\3\\5\end{bmatrix}$, the projection onto $\mathcal{H}_w$ discards the $z$-component:
$$
x_\perp = \begin{bmatrix}2\\3\\5\end{bmatrix} - (w^T x)\,w = \begin{bmatrix}2\\3\\5\end{bmatrix} - 5\begin{bmatrix}0\\0\\1\end{bmatrix} = \begin{bmatrix}2\\3\\0\end{bmatrix}
$$
Visually:
![](output-2%201.png)