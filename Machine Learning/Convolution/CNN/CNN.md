A convolutional neural network (CNN) is a feedforward architecture that applies learned [convolution](Overview.md) filters to extract hierarchical spatial features, followed by a task-specific head.

---
## Definition
A CNN with $L$ layers alternates convolution, activation, and pooling stages. For a 2-D feature map $H^{(\ell)} \in \mathbb{R}^{C_\ell \times H_\ell \times W_\ell}$, a single convolutional layer computes:
$$
H^{(\ell)}_{c}(x,y) = \sigma\!\left(\sum_{c'} \sum_{i,j} K^{(\ell)}_{c,c'}(i,j)\, H^{(\ell-1)}_{c'}(x+i,\, y+j) + b^{(\ell)}_c\right)
$$ where:
- **$K^{(\ell)}_{c,c'} \in \mathbb{R}^{k \times k}$:** Learned kernel mapping input channel $c'$ to output channel $c$.
- **$b^{(\ell)}_c \in \mathbb{R}$:** Bias for output channel $c$.
- **$\sigma$:** Element-wise activation (e.g. ReLU).
- **$H^{(\ell)}_c(x,y)$:** Activation at spatial position $(x,y)$ of output channel $c$.

A typical CNN stacks several such layers, interleaved with **max pooling** to downsample spatial dimensions, then flattens the result and passes it through one or more fully connected layers to produce the final output.

---
## Example
Consider a grayscale $5 \times 5$ input ($C_0 = 1$) with a single $3 \times 3$ filter ($C_1 = 1$, stride 1, no padding).
$$
H^{(0)} = \begin{bmatrix}
1 & 0 & 1 & 0 & 1 \\
0 & 1 & 0 & 1 & 0 \\
1 & 0 & 1 & 0 & 1 \\
0 & 1 & 0 & 1 & 0 \\
1 & 0 & 1 & 0 & 1
\end{bmatrix}, \qquad
K = \begin{bmatrix}
1 & 0 & -1 \\
1 & 0 & -1 \\
1 & 0 & -1
\end{bmatrix}, \quad b = 0
$$
### Step 1: Convolve (top-left $3 \times 3$ patch)

$$
z_{1,1} = \sum_{i,j} K(i,j)\, H^{(0)}(i,j) = (1)(1)+(0)(0)+(-1)(1)+(1)(0)+(0)(1)+(-1)(0)+(1)(1)+(0)(0)+(-1)(1) = 0
$$
Sliding the filter across all valid positions produces a $3 \times 3$ pre-activation map $Z$.
### Step 2: Activation (ReLU)
$$
H^{(1)} = \mathrm{ReLU}(Z), \qquad H^{(1)}_{x,y} = \max(0,\, Z_{x,y})
$$
### Step 3: Max pooling ($2 \times 2$, stride 2)
$$
P_{x,y} = \max_{i,j \in \{0,1\}} H^{(1)}(2x+i,\, 2y+j)
$$
Reduces the $3 \times 3$ feature map to a $1 \times 1$ scalar (largest activation), which captures the dominant edge response regardless of exact position — the source of [translation invariance](CNN-Invariances.md).
### Step 4: Classification head
The pooled feature is flattened and passed through a linear layer with softmax to produce class probabilities.
