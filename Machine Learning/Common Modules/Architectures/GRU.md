A gated recurrent unit (GRU) is an [RNN](RNN.md) variant that uses two multiplicative gates to interpolate between the previous hidden state and a new candidate state, retaining long-range information like an [LSTM](LSTM.md) with fewer parameters and no separate cell state.

---
## Theory
A vanilla [RNN](RNN.md) overwrites its hidden state at every timestep, so the Jacobian $\partial h_t / \partial h_{t-1}$ is a product of a weight matrix and a squashing derivative; over many steps these products shrink or grow geometrically (vanishing/exploding gradients). A GRU replaces the overwrite with a learned convex combination:
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t
$$ where:
- **$z_t \in (0,1)^{d_h}$:** **Update gate**; per-coordinate fraction of the state that is replaced.
- **$\tilde{h}_t \in (-1,1)^{d_h}$:** **Candidate state** proposed from the current input and (reset) past state.

Because the update is additive, the Jacobian contains the term $\operatorname{diag}(1 - z_t)$. When $z_t \to \mathbf{0}$ the state is copied unchanged, $h_t \approx h_{t-1}$, and the gradient along that path is close to the identity, so information and gradients can travel across many timesteps. A second gate, the **reset gate** $r_t$, controls how much of $h_{t-1}$ is used when forming $\tilde{h}_t$; with $r_t \to \mathbf{0}$ the candidate ignores the past, letting the unit start a fresh representation.

Compared with an [LSTM](LSTM.md), the GRU merges the forget and input gates into the single update gate (forget $= 1 - z_t$, input $= z_t$) and merges the cell and hidden states, giving three weight blocks instead of four. Setting $z_t = r_t = \mathbf{1}$ recovers the vanilla [RNN](RNN.md) with $\sigma = \tanh$.

---
## Definition
Given an input sequence $x_1, \ldots, x_T$ with $x_t \in \mathbb{R}^{d_x}$ and initial state $h_0 \in \mathbb{R}^{d_h}$ (typically $\mathbf{0}$), a GRU computes for $t = 1, \ldots, T$:
$$
\boxed{
\begin{aligned}
z_t &= \sigma\!\left(W_z\, x_t + U_z\, h_{t-1} + b_z\right) & &\text{(update gate)}\\
r_t &= \sigma\!\left(W_r\, x_t + U_r\, h_{t-1} + b_r\right) & &\text{(reset gate)}\\
\tilde{h}_t &= \tanh\!\left(W_h\, x_t + U_h\,(r_t \odot h_{t-1}) + b_h\right) & &\text{(candidate state)}\\
h_t &= (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t & &\text{(hidden state)}
\end{aligned}
}
$$ where:
- **$h_t \in \mathbb{R}^{d_h}$:** Hidden state at time $t$; also the output of the unit.
- **$W_z, W_r, W_h \in \mathbb{R}^{d_h \times d_x}$:** Input weight matrices.
- **$U_z, U_r, U_h \in \mathbb{R}^{d_h \times d_h}$:** Recurrent weight matrices.
- **$b_z, b_r, b_h \in \mathbb{R}^{d_h}$:** Biases.
- **$\sigma$:** Element-wise logistic sigmoid, $\sigma(a) = 1/(1 + e^{-a})$.
- **$\odot$:** Element-wise (Hadamard) product.

The same parameters are shared across all timesteps, giving $3\,(d_h d_x + d_h^2 + d_h)$ parameters in total. A task output is read from the states, e.g. $\hat{y}_t = W_y\, h_t + b_y$.

Two common variants are equivalent in expressive power: some libraries (e.g. PyTorch) swap the roles of $z_t$ and $1 - z_t$ in the hidden-state update, and apply the reset gate after the recurrent product, $\tilde{h}_t = \tanh\!\left(W_h\, x_t + b_h + r_t \odot (U_h\, h_{t-1} + b'_h)\right)$.

---
## Example
Scalar case ($d_x = d_h = 1$), one timestep with $h_0 = 0.5$ and $x_1 = 1$.
$$
W_z = 1,\; U_z = 0,\; b_z = 0, \qquad W_r = U_r = b_r = 0, \qquad W_h = 1,\; U_h = 1,\; b_h = 0
$$
### Step 1: Update gate
$$
z_1 = \sigma(1 \cdot 1 + 0 \cdot 0.5 + 0) = \sigma(1) \approx 0.731
$$
### Step 2: Reset gate
$$
r_1 = \sigma(0) = 0.5
$$
### Step 3: Candidate state
$$
\tilde{h}_1 = \tanh\!\left(1 \cdot 1 + 1 \cdot (0.5 \cdot 0.5) + 0\right) = \tanh(1.25) \approx 0.848
$$
The reset gate halves the contribution of $h_0$ to the candidate.
### Step 4: Hidden state
$$
h_1 = (1 - 0.731) \cdot 0.5 + 0.731 \cdot 0.848 \approx 0.134 + 0.620 = 0.754
$$
About 27% of the old state is kept and 73% is replaced by the candidate; had $z_1 = 0$, the unit would have copied $h_1 = h_0 = 0.5$ exactly.
