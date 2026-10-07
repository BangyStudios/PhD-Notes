## Example: Translation Invariance in a CNN

A convolutional neural network (CNN) with convolution + max pooling layers can learn **translation invariance** — the ability to recognize an object regardless of its exact position in the input image.

---
### Scenario
Suppose we train a CNN to detect whether an image contains a **vertical edge** (a transition from black to white).
- Input images are $5 \times 5$ grids of pixels with values $0$ (black) or $1$ (white).
- The CNN applies a convolution with the kernel
$$
K = \begin{bmatrix} -1 & 1 \end{bmatrix}
$$
which responds strongly to vertical transitions.

---
### Discrete Invariance

If the edge appears at column $2$ or column $4$, the convolutional filter still activates strongly.

After applying **max pooling**, the network retains only the strongest response, discarding the exact location.

Thus, the CNN becomes invariant to **discrete horizontal translations** of the edge.

---
### Mathematical Representation
Let the input image be $I(x,y)$.  
The convolution operation is:
$$
S(x,y) = \sum_{i,j} K(i,j) \, I(x+i, y+j)
$$
where $K$ is the kernel.

After applying a **max pooling** operation:
$$
P = \max_{x,y} S(x,y)
$$
the pooled response $P$ is **invariant** to discrete translations of the edge in the input, since shifting the edge changes the location of the maximum but not its value.

---
✅ **Key Point:**  
This is the simplest learned discrete invariance in CNNs: a feature (vertical edge) is detected regardless of _where_ it occurs, due to **convolution + pooling**.