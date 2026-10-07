A residual network (ResNet) is a deep [CNN](CNN.md) architecture that adds shortcut (skip) connections around layer groups, so each block learns a residual function rather than an unrestricted mapping, enabling training of very deep networks.

---
## Definition
A **residual block** with input $x \in \mathbb{R}^d$ is defined as:
$$
y = \mathcal{F}(x,\, \{W_i\}) + x
$$

where $\mathcal{F}$ is a stack of layers (typically two $3 \times 3$ convolutions with batch norm and ReLU) and $+\,x$ is the identity shortcut. The network learns $\mathcal{F} = y - x$ (the residual) rather than $y$ directly.

When the input and output dimensions differ, the shortcut uses a $1 \times 1$ convolution to project:$$
y = \mathcal{F}(x,\, \{W_i\}) + W_s\, x
$$ where:
- **$\mathcal{F}(x, \{W_i\})$:** Residual mapping computed by the stacked layers.
- **$W_s$:** Linear projection for dimension matching.
- **Gradient flow:** The identity path $\frac{\partial y}{\partial x} \ni I$ ensures [gradients](5-Loss_Function.md) reach early layers without vanishing.

---
## Example
Scalar illustration, $d = 1$.
$$
x = 2, \qquad \mathcal{F}(x) = \mathrm{ReLU}(W_2\, \mathrm{ReLU}(W_1 x))
$$
Let $W_1 = 0.5$, $W_2 = 0.5$.
### Step 1: Residual branch
$$
z_1 = \mathrm{ReLU}(0.5 \cdot 2) = \mathrm{ReLU}(1) = 1, \qquad \mathcal{F}(x) = \mathrm{ReLU}(0.5 \cdot 1) = 0.5
$$
### Step 2: Shortcut addition
$$
y = \mathcal{F}(x) + x = 0.5 + 2 = 2.5
$$
If $\mathcal{F}(x) \to 0$ (the block learns to do nothing), the output is simply $y = x$ — the network gracefully degrades to a shallower model, which is why residual networks are easier to optimize than plain deep networks.
