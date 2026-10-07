A long short-term memory network (LSTM) is an [RNN](RNN.md) variant that adds a cell state and three multiplicative gates to selectively retain or forget information over long sequences, mitigating the vanishing gradient problem.
![](Pasted%20image%2020260811131720.png)

---
## Definition
At each timestep $t$, given input $x_t \in \mathbb{R}^{d_x}$ and previous state $(h_{t-1}, c_{t-1})$:
$$
\begin{aligned}
f_t &= \sigma\!\left(W_f [h_{t-1};\, x_t] + b_f\right) & &\text{(forget gate)}\\
i_t &= \sigma\!\left(W_i [h_{t-1};\, x_t] + b_i\right) & &\text{(input gate)}\\
g_t &= \tanh\!\left(W_c [h_{t-1};\, x_t] + b_c\right) & &\text{(candidate cell)}\\
c_t &= f_t \odot c_{t-1} + i_t \odot g_t & &\text{(cell state update)}\\
o_t &= \sigma\!\left(W_o [h_{t-1};\, x_t] + b_o\right) & &\text{(output gate)}\\
h_t &= o_t \odot \tanh(c_t) & &\text{(hidden state)}
\end{aligned}
$$ where:
- **$[h_{t-1};\, x_t]$:** Concatenation of the previous hidden state and current input.
- **$f_t, i_t, o_t \in [0,1]^{d_h}$:** Gate vectors controlling what to forget, write, and read.
- **$c_t \in \mathbb{R}^{d_h}$:** Cell state; acts as long-term memory conveyed across timesteps.
- **$\odot$:** Element-wise (Hadamard) product.
- **$\sigma$:** Sigmoid nonlinearity.

---
## Example
Scalar case ($d_h = d_x = 1$), $h_0 = c_0 = 0$.

Let all gate biases be $0$ and let weights produce the following gate values at $t = 1$ for input $x_1 = 1$:
$$
f_1 = 0.2, \quad i_1 = 0.8, \quad \tilde{c}_1 = \tanh(1) \approx 0.76, \quad o_1 = 0.9
$$
### Cell state update
$$
c_1 = f_1 \cdot c_0 + i_1 \cdot \tilde{c}_1 = 0.2 \cdot 0 + 0.8 \cdot 0.76 = 0.61
$$
### Hidden state
$$
h_1 = o_1 \cdot \tanh(c_1) = 0.9 \cdot \tanh(0.61) \approx 0.9 \cdot 0.54 = 0.49
$$
The forget gate $f_1 = 0.2$ nearly wipes the previous cell state; the input gate $i_1 = 0.8$ writes most of the candidate, so $c_1$ now carries information from $x_1$.
