> Compared to [knowledge graphs (KGs)](Topics/Machine%20Learning/Graph/Knowledge%20Graph/0-Overview.md), TKGs contain additional timestamps, which are taken into account in the construction of TKGRL methods. These methods can be broadly categorized into transformation-based, decomposition-based, graph neural networks-based, and capsule network-based approaches.
> ([Cai2024SurveyKGTemporal, p.6](Cai2024SurveyKGTemporal.pdf#page=6&selection=143,0,149,1&color=yellow))

## Methods
### Transformation-based Methods
#### Translation-based
Regards the relation $r$ as a translation from the head entity $h$ to the tail entity $t$, that is, $h + r = t$ as implemented in **TransE**. 
* In **TTransE**, the temporal information is concatenated with the relation $(h, [r \mid \tau], t)$.
* In **TA-TransE** a LSTM is used to learn the relation embedding, where the score function is $\Vert h + r_{seq} - t \Vert_2$.
* In **HyTE** the entities, temporal information and relations are learned jointly. It splits the temporal knowledge graph into multiple subgraphs, each of which corresponds to a timestamp. Then it incorporates time in the entity-relation space by associating each timestamp with a corresponding [hyperplane](Hyperplane.md).
	* For a timestamp $\tau$, the corresponding [hyperplane](Hyperplane.md) is $w_\tau$, if a triple $(h, r, t)$ is valid for the timestamp, then the [projected](Orthogonal%20Projection.md) residual representations of the triple on the hyperplane are $h_\tau = h - (w^T_\tau h)w_\tau$, $r_\tau = r - (w^T_\tau r)w_\tau$ and $t_\tau = t - (w^T_\tau t)w_\tau$, and the score function is $\Vert h_\tau + r_\tau - t_\tau \Vert_\frac{1}{2}$.
#### Rotation-based
Since the translation-based methods can infer the inverse and composition relation patterns, but not symmetry relations represented by a 0 vector, rotation-based methods appear.
* In **RotatE**, a relation is regarded as a rotation from the head entity to the tail entity and expands the representation space from a real valued point-wise space $\mathbb{R}^d$ to a [complex vector space](Complex%20Vector%20Space.md) $\mathbb{C}^d$ defined by $c = a + bi, a, b \in \mathbb{R}^d$, and expects $t = h \odot r$, where $\odot$ is the [element-wise product](Element-wise%20Product.md), and the score function is $\Vert h \odot r - t \Vert_1$.
* [Redacted for brevity](Cai2024SurveyKGTemporal.pdf#page=8&selection=198,0,381,1&color=yellow)
#### Decomposition-based
The knowledge graph consists of triples and can be represented by an order 3 tensor. For the temporal knowledge graph, the additional temporal information can be represented by an order 4 tensor, and each tensor dimension is the head entity, relation, tail entity, and timestamp, respectively.
.
.
.
### Graph Neural Network-based
