A recurrent neural network (RNN) processes sequences by maintaining a hidden state that is updated at each timestep, allowing information to persist across the sequence.

---
## Definition
Given an input sequence $x_1, x_2, \ldots, x_T$ with $x_t \in \mathbb{R}^{d_x}$, an RNN computes:
$$
h_t = \sigma\!\left(W_h\, h_{t-1} + W_x\, x_t + b\right), \qquad \hat{y}_t = W_y\, h_t + b_y
$$ where:
- **$h_t \in \mathbb{R}^{d_h}$:** Hidden state at time $t$; initialized as $h_0 = \mathbf{0}$.
- **$W_h \in \mathbb{R}^{d_h \times d_h}$:** Recurrent weight matrix (transitions between hidden states).
- **$W_x \in \mathbb{R}^{d_h \times d_x}$:** Input weight matrix.
- **$b \in \mathbb{R}^{d_h}$:** Hidden bias.
- **$\sigma$:** Nonlinearity (typically $\tanh$).
- **$W_y, b_y$:** Output projection weights.

The same weights $\{W_h, W_x, W_y\}$ are reused at every timestep. [Gradients](5-Loss_Function.md) flow back through time via [backpropagation](6-Backpropagation.md) through time (BPTT), but can vanish or explode over long sequences — motivating [LSTM](LSTM.md).

---
## Example
Sequence of length $T = 2$, $d_x = d_h = 2$, $\sigma = \tanh$.
$$
W_x = I_2, \quad W_h = \mathbf{0}, \quad b = \mathbf{0}, \qquad x_1 = \begin{bmatrix}1 \\ 0\end{bmatrix}, \quad x_2 = \begin{bmatrix}0 \\ 1\end{bmatrix}
$$
### Step 1: $t = 1$
$$
h_1 = \tanh(W_h \cdot \mathbf{0} + W_x x_1) = \tanh\!\begin{bmatrix}1 \\ 0\end{bmatrix} = \begin{bmatrix}0.76 \\ 0\end{bmatrix}
$$
### Step 2: $t = 2$
$$
h_2 = \tanh(W_h h_1 + W_x x_2) = \tanh\!\begin{bmatrix}0 \\ 1\end{bmatrix} = \begin{bmatrix}0 \\ 0.76\end{bmatrix}
$$
Because $W_h = \mathbf{0}$ here the states are independent; in practice $W_h \neq \mathbf{0}$ couples them so $h_2$ depends on $x_1$ through $h_1$.
