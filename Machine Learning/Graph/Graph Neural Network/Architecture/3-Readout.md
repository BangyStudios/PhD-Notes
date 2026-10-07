A readout (global pooling) function $R$ aggregates all node embeddings after $K$ message-passing layers into a single fixed-size graph-level vector, enabling graph-level prediction tasks such as graph classification or regression.
## Definition
$$h_G = R\!\left(\{h^{(K)}_v : v \in \mathcal{V}\}\right)$$
Like [Aggregation](2.2-Aggregation.md), $R$ must be **permutation-invariant** with respect to node ordering. Common forms:
- **Sum:** $h_G = \sum_{v \in \mathcal{V}} h^{(K)}_v$ — sensitive to graph size; theoretically the most expressive simple readout (GIN, Xu et al. 2019)
- **Mean:** $h_G = \frac{1}{N}\sum_{v} h^{(K)}_v$ — normalized; invariant to the number of nodes; useful when comparing graphs of different sizes
- **Max:** $h_G = \max_v h^{(K)}_v$ — element-wise maximum; captures the most salient feature per dimension regardless of graph size
- **Gated / attention:** $h_G = \sum_v \text{softmax}_v\!\left(f(h_v)\right) \odot g(h_v)$ — learnable per-node importance gate; $f$ and $g$ are MLPs; used in Set2Vec (Vinyals et al. 2016) and gated global pooling
After readout, $h_G$ is typically passed to an MLP for the final prediction.
---
## Example
3-node graph with 2-dimensional node embeddings after $K$ message-passing layers. Sum readout then linear classifier for 3 classes.
$$
h_G = \sum_v h^{(K)}_v = \begin{bmatrix}0.5\\0.1\end{bmatrix} + \begin{bmatrix}0.2\\0.8\end{bmatrix} + \begin{bmatrix}0.3\\0.4\end{bmatrix} = \begin{bmatrix}1.0\\1.3\end{bmatrix}
$$
```python
import torch
import torch.nn as nn

# Final node embeddings after K layers [N, d] = [3, 2]
H_final = torch.tensor([[0.5, 0.1],
                         [0.2, 0.8],
                         [0.3, 0.4]])

# --- Sum readout ---
h_G_sum = H_final.sum(dim=0)           # [d] = [2]
print("Sum graph embedding:", h_G_sum)  # [1.0, 1.3]

# --- Mean readout ---
h_G_mean = H_final.mean(dim=0)         # [d] = [2]

# --- Max readout ---
h_G_max = H_final.max(dim=0).values    # [d] = [2]

# --- Gated readout (soft attention) ---
gate_fn = nn.Linear(2, 2)              # f: compute gate logits
feat_fn = nn.Linear(2, 2)              # g: transform features
gates = torch.softmax(gate_fn(H_final), dim=0)   # [N, d], softmax over nodes
h_G_gated = (gates * feat_fn(H_final)).sum(dim=0) # [d]

# Classifier on top of sum readout
classifier = nn.Linear(2, 3)           # 3 output classes
logits = classifier(h_G_sum)           # [3]
probs = torch.softmax(logits, dim=0)
print("Class probabilities:", probs.detach())
```
