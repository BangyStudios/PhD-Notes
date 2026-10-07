A temporal convolutional network (TCN) models sequences with stacked causal, dilated 1-D [convolutions](../../Convolution/Convolution.md), so each output depends only on present and past inputs while the receptive field grows exponentially with depth.

---
## Theory
A sequence model maps $x_1, \ldots, x_T$ to $y_1, \ldots, y_T$ under the **causality** constraint that $y_t$ depends only on $x_1, \ldots, x_t$. An [RNN](RNN.md) enforces this by carrying a hidden state forward one step at a time; a TCN enforces it structurally, by letting each convolution look only backward in time.

Three ideas make this work:
- **Causal convolution:** The kernel is applied to $x_t, x_{t-1}, \ldots$ only, with zero padding on the left so the output has the same length $T$ as the input and no future value leaks in.
- **Dilation:** A dilation factor $d$ spaces the kernel taps $d$ steps apart. Doubling $d$ at each layer ($d_\ell = 2^{\ell-1}$) makes the receptive field grow exponentially with depth $L$, whereas an undilated stack grows only linearly.
- **Residual connections:** Each block adds its input to its output, as in [ResNet](ResNet.md), so deep stacks remain trainable.

For kernel size $k$ and dilations $d_1, \ldots, d_L$, the **receptive field** (number of past inputs, including $x_t$, that can influence $y_t$) is
$$
\boxed{
R = 1 + (k - 1)\sum_{\ell=1}^{L} d_\ell
}
$$ where:
- **$k$:** Kernel size (number of taps per convolution).
- **$d_\ell$:** Dilation factor of layer $\ell$.
- **$L$:** Number of convolutional layers.

With $d_\ell = 2^{\ell-1}$ this gives $R = 1 + (k-1)(2^L - 1)$. Compared with an RNN, a TCN computes all timesteps in parallel, and the gradient path from $y_t$ to $x_{t-\tau}$ has length $L$ rather than $\tau$, avoiding vanishing gradients through time; the trade-off is a finite memory of $R$ steps fixed by the architecture.

---
## Definition
Let $h^{(0)}_t = x_t \in \mathbb{R}^{C_0}$ for $t = 1, \ldots, T$, with $h^{(\ell-1)}_t = \mathbf{0}$ for $t \le 0$ (left zero padding). A **causal dilated convolution** layer $\ell$ computes
$$
\boxed{
h^{(\ell)}_t = \phi\!\left(\sum_{i=0}^{k-1} W^{(\ell)}_i\, h^{(\ell-1)}_{t - d_\ell\, i} + b^{(\ell)}\right)
}
$$ where:
- **$h^{(\ell)}_t \in \mathbb{R}^{C_\ell}$:** Feature vector of layer $\ell$ at time $t$ ($C_\ell$ channels).
- **$W^{(\ell)}_i \in \mathbb{R}^{C_\ell \times C_{\ell-1}}$:** Kernel weights for tap $i$; tap $i = 0$ reads the current timestep, tap $i$ reads $d_\ell\, i$ steps into the past.
- **$b^{(\ell)} \in \mathbb{R}^{C_\ell}$:** Bias.
- **$d_\ell \in \mathbb{N}$:** Dilation factor ($d_\ell = 1$ is an ordinary causal convolution).
- **$\phi$:** Element-wise activation (e.g. ReLU).

Weights are shared across all timesteps $t$. A **TCN** stacks such layers, usually grouped into **residual blocks**, each containing one or more causal dilated convolutions $\mathcal{F}$ with a common dilation:
$$
h^{(\ell)} = \phi\!\left(V^{(\ell)} h^{(\ell-1)} + \mathcal{F}^{(\ell)}\big(h^{(\ell-1)}\big)\right)
$$ where:
- **$\mathcal{F}^{(\ell)}$:** The block's convolution stack (the standard TCN uses two layers with normalization and dropout).
- **$V^{(\ell)}$:** Identity if $C_\ell = C_{\ell-1}$, otherwise a learned $1 \times 1$ convolution matching channel counts.

The output is read per timestep, e.g. $\hat{y}_t = W_y\, h^{(L)}_t + b_y$, and $\hat{y}_t$ depends only on $x_{t-R+1}, \ldots, x_t$.

---
## Example
Single channel ($C_\ell = 1$), kernel size $k = 2$, $L = 2$ layers with dilations $d_1 = 1$, $d_2 = 2$, all weights $W_0 = W_1 = 1$, biases $0$, identity activation $\phi(a) = a$, no residual connection.
$$
x = (x_1, x_2, x_3, x_4) = (1, 2, 3, 4)
$$
### Step 1: Layer 1 ($d_1 = 1$)
$$
h^{(1)}_t = x_t + x_{t-1} \;\Rightarrow\; h^{(1)} = (1 + 0,\; 2 + 1,\; 3 + 2,\; 4 + 3) = (1, 3, 5, 7)
$$
The zero-padded $x_0 = 0$ keeps $h^{(1)}_1$ causal and the length at $T = 4$.
### Step 2: Layer 2 ($d_2 = 2$)
$$
h^{(2)}_t = h^{(1)}_t + h^{(1)}_{t-2} \;\Rightarrow\; h^{(2)} = (1 + 0,\; 3 + 0,\; 5 + 1,\; 7 + 3) = (1, 3, 6, 10)
$$
### Step 3: Receptive field
$$
R = 1 + (2 - 1)(1 + 2) = 4, \qquad h^{(2)}_4 = x_1 + x_2 + x_3 + x_4 = 10
$$
The final output sees all four inputs using only two layers, and no output depends on a future input (e.g. $h^{(2)}_2 = x_1 + x_2$).
