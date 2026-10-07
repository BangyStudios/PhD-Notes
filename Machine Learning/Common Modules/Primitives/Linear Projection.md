A **linear projection** maps an input matrix $X \in \mathbb{R}^{n \times d_{\text{in}}}$ to an output matrix $Y \in \mathbb{R}^{n \times d_{\text{out}}}$ via a learned weight matrix $W \in \mathbb{R}^{d_{\text{in}} \times d_{\text{out}}}$:
$$
Y = XW
$$
Each row of $X$ (one sample or token) is independently projected into the output space by the same $W$. The weight matrix $W$ is learned end-to-end by gradient descent.
- **Bias term (optional):** A bias $b \in \mathbb{R}^{d_{\text{out}}}$ can be added: $Y = XW + b$, broadcast across rows.
- **Dimensionality change:** $d_{\text{out}}$ can be larger (expansion), smaller (compression/bottleneck), or equal to $d_{\text{in}}$ (re-representation).
- **Interpretation:** Each column of $W$ defines a direction in the input space; the projection measures how much each input aligns with each direction, producing a coordinate in the output space.
---
## Example
### Setup
Suppose we have 3 samples with 2-dimensional features and want to project to 3 dimensions:
$$
X = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \qquad
W = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}
$$
where $X \in \mathbb{R}^{3 \times 2}$ and $W \in \mathbb{R}^{2 \times 3}$.
---
### Computation
$$
Y = XW = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 1 & 2 \end{bmatrix}
$$
---
### Interpretation
- Row 1 of $Y$ is the projection of sample $[1, 0]$: it picks out the first direction of $W$ with weight 1 and ignores the second.
- Row 3 of $Y$ is the projection of $[1, 1]$: both directions contribute equally, producing $[1, 1, 2]$.
- The third output dimension (last column of $W$) sums both input features, acting as a learned combination.
