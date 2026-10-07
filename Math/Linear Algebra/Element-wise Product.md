## Definition
The element-wise product (Hadamard product) of two vectors $\mathbf{a}, \mathbf{b} \in \mathbb{F}^d$ multiplies corresponding components:
$$
\mathbf{a} \odot \mathbf{b} = \begin{bmatrix} a_1 b_1 \\ a_2 b_2 \\ \vdots \\ a_d b_d \end{bmatrix} \in \mathbb{F}^d
$$
where $\mathbb{F}$ is $\mathbb{R}$ or $\mathbb{C}$. The result has the same shape as the inputs, unlike the dot product which collapses to a scalar.

### Example in $\mathbb{R}^3$
$$
\begin{bmatrix}1\\2\\3\end{bmatrix} \odot \begin{bmatrix}4\\5\\6\end{bmatrix} = \begin{bmatrix}4\\10\\18\end{bmatrix}
$$
