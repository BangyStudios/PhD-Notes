A multi-layer perceptron (MLP) is a fully connected feedforward network that applies a sequence of affine transformations interleaved with nonlinear activations to map an input to an output.

---
## Definition
For a network with $L$ layers parameterized by $\{W^{(\ell)}, b^{(\ell)}\}_{\ell=1}^{L}$:
$$
h^{(0)} = x, \qquad h^{(\ell)} = \sigma^{(\ell)}\!\left(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\right), \qquad \hat{y} = h^{(L)}
$$ where:
- **$x \in \mathbb{R}^{d_{\text{in}}}$:** Input vector.
- **$W^{(\ell)} \in \mathbb{R}^{d_\ell \times d_{\ell-1}}$:** Weight matrix at layer $\ell$.
- **$b^{(\ell)} \in \mathbb{R}^{d_\ell}$:** Bias vector at layer $\ell$.
- **$\sigma^{(\ell)}$:** Pointwise nonlinearity (e.g. ReLU for hidden layers, softmax for the output).
- **$\hat{y}$:** Network prediction.

Every unit in layer $\ell$ is connected to every unit in layer $\ell-1$ — there is no weight sharing, unlike a [CNN](../Convolution/CNN.md).

---
## Example
A 2-layer MLP with $d_{\text{in}} = 2$, hidden size $d_1 = 2$, output size $d_2 = 1$.
$$
x = \begin{bmatrix}1 \\ 2\end{bmatrix}, \quad
W^{(1)} = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}, \quad
b^{(1)} = \begin{bmatrix}0 \\ 0\end{bmatrix}, \quad
W^{(2)} = \begin{bmatrix}1 & -1\end{bmatrix}, \quad
b^{(2)} = \begin{bmatrix}0\end{bmatrix}
$$
### Step 1: Hidden layer (ReLU)
$$
z^{(1)} = W^{(1)} x = \begin{bmatrix}1 \\ 2\end{bmatrix}, \qquad h^{(1)} = \mathrm{ReLU}(z^{(1)}) = \begin{bmatrix}1 \\ 2\end{bmatrix}
$$
### Step 2: Output layer
$$
z^{(2)} = W^{(2)} h^{(1)} = \begin{bmatrix}1 & -1\end{bmatrix}\begin{bmatrix}1 \\ 2\end{bmatrix} = -1, \qquad \hat{y} = \sigma(z^{(2)})
$$
With a sigmoid output: $\hat{y} = 1/(1+e^{1}) \approx 0.27$.
