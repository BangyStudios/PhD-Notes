A Graph Neural Network (GNN) models non-Euclidean relationships by iteratively aggregating information from a node's neighborhood. The canonical framework is the **Message Passing Neural Network (MPNN)**:
$$
\begin{aligned}
m^{(k+1)}_v &= A^{(k)}\!\left(\{M^{(k)}(h^{(k)}_u, h^{(k)}_v, e_{uv}) : u \in \mathcal{N}(v)\}\right) \\
h^{(k+1)}_v &= U^{(k)}\!\left(h^{(k)}_v, m^{(k+1)}_v\right)
\end{aligned}
$$
where $M$ is the [message function](2.1-Message.md), $A$ is the [aggregation function](2.2-Aggregation.md), $U$ is the [update function](2.3-Update.md), and $h^{(k)}_v$ is the embedding of node $v$ at layer $k$. After $K$ layers each node's embedding encodes information from its $K$-hop neighborhood.
**Components:**
- [Graph Representation](1-Graph%20Representation.md) — nodes, edges, adjacency matrix, feature matrices
- [Message](2.1-Message.md) — compute a message from each neighbor
- [Aggregation](2.2-Aggregation.md) — combine neighbor messages permutation-invariantly
- [Update](2.3-Update.md) — update the node embedding using the aggregated message
- [Readout](3-Readout.md) — produce a graph-level representation for graph-level tasks
**Variant architectures:** [Graph Attention Network](Graph%20Attention%20Network.md), [R-GCN](Relational-Graph%20Convolution%20Network.md)

### MPNN Implementation (Example)
```python
class MessagePassingLayer(nn.Module):
	"""
	Single GNN layer implementing:
	1. Message: m_uv = MLP([h_u || h_v])
	2. Aggregation: sum over neighbors
	3. Update: h_v_new = MLP([h_v || m_v])
	"""
	def __init__(self, d_in, d_z):
		super().__init__()
		# Message function: takes concatenated source and target features
		self.proj_message = nn.Sequential(
			nn.Linear(2 * d_in, d_z),
			nn.ReLU(),
			nn.Linear(d_z, d_z)
		)
		
		# Update function: takes concatenated current node and aggregated message
		self.proj_update = nn.Sequential(
			nn.Linear(d_in + d_z, d_z),
			nn.ReLU(),
			nn.Linear(d_z, d_z)
		)
	
	def forward(self, h_v, edge_index):
		N = h_v.shape[0]
		u, v = edge_index[0], edge_index[1]
		
		# --- Message Computation ---
		# Gather source and target features
		h_u = h_v[u] # [E, in_dim]
		h_v = h_v[v] # [E, in_dim]
		
		# Concatenate source and target features for each edge
		h_u_cat_h_v = torch.cat([h_u, h_v], dim=1) # [E, 2*in_dim]
		
		# Compute messages for each edge
		m_uv = self.proj_message(h_u_cat_h_v) # [E, hidden_dim]
		
		# --- Aggregation (sum) ---
		# Sum messages for each destination node
		m_v = torch.zeros(
			(N, m_uv.shape[1]),
			device=h_v.device,
			dtype=m_uv.dtype
		)
		m_v.index_add_(0, v, m_uv)
		
		# --- Update ---
		# Concatenate current node features with aggregated messages
		h_v_cat_m_v = torch.cat([h_v, m_v], dim=1) # [N, in_dim + hidden_dim]
		
		# Apply update MLP
		h_v_new = self.proj_update(h_v_cat_m_v) # [N, hidden_dim]
		
		return h_v_new
```

### GNN Abstraction Implementation (Example)
```python

```