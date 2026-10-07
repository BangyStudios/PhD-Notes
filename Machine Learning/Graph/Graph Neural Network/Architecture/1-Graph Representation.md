A graph $G = (\mathcal{V}, \mathcal{E})$ provides the structural backbone that GNNs operate on, encoding both topology and features.
## Definition
- $\mathcal{V}$: set of $N$ nodes
- $\mathcal{E} \subseteq \mathcal{V} \times \mathcal{V}$: set of edges; $(u, v) \in \mathcal{E}$ means there is a directed edge from $u$ to $v$; undirected graphs add both $(u,v)$ and $(v,u)$
- $A \in \{0,1\}^{N \times N}$: **adjacency matrix**, $A_{uv} = 1$ if $(u,v) \in \mathcal{E}$, else $0$
- $H \in \mathbb{R}^{N \times d}$: **node feature matrix**, row $h_v \in \mathbb{R}^d$ is the initial feature vector of node $v$
- $e_{uv} \in \mathbb{R}^{d_e}$: (optional) **edge feature vector** for edge $(u,v)$, e.g. bond type in molecular graphs
The **edge index** stores edges as a tensor of shape $[2, |\mathcal{E}|]$ containing source and destination node indices. This sparse format is preferred over a dense adjacency matrix for large, sparse graphs because it avoids storing $O(N^2)$ zeros.
$\mathcal{N}(v) = \{u \in \mathcal{V} : (u,v) \in \mathcal{E}\}$ denotes the set of **neighbors** of $v$ (nodes with an edge into $v$).
---
## Example
A triangle graph with 3 nodes and directed edges $0 \to 1$, $1 \to 2$, $2 \to 0$. Each node has a 2-dimensional feature vector.
$$
A = \begin{bmatrix} 0 & 1 & 0 \\ 0 & 0 & 1 \\ 1 & 0 & 0 \end{bmatrix}, \quad
H = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
$$
```python
import torch

# Node features: shape [N, d] = [3, 2]
H = torch.tensor([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])

# Adjacency matrix: shape [N, N] = [3, 3]
A = torch.tensor([[0, 1, 0],
                  [0, 0, 1],
                  [1, 0, 0]], dtype=torch.float)

# Edge index (sparse COO format): shape [2, |E|] = [2, 3]
# Row 0 = source nodes, Row 1 = destination nodes
edge_index = torch.tensor([[0, 1, 2],
                            [1, 2, 0]])

# Optional edge features: shape [|E|, d_e] = [3, 1]
edge_attr = torch.tensor([[0.5], [0.8], [0.3]])

# Neighbors of node 0: nodes u where edge_index[:, e] = [u, 0]
dst = edge_index[1]
neighbors_of_0 = edge_index[0][dst == 0].tolist()
print("Neighbors of node 0:", neighbors_of_0)  # [2]
```
