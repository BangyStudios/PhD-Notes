### Matrix $\times$ Vector
Input vector size $n$ must equal the matrix column size $n$

| Scenario        | Matrix $A$                                                      | Input          | Output                                | Map                                              |
| --------------- | --------------------------------------------------------------- | -------------- | ------------------------------------- | ------------------------------------------------ |
| Compression     | $A \in \mathbb{R}^{m \times n},\ m < n$                         | $\mathbb{R}^n$ | $\mathbb{R}^m$                        | Higher $\to$ Lower dim                           |
| Expansion       | $A \in \mathbb{R}^{m \times n},\ m > n$                         | $\mathbb{R}^n$ | $\mathbb{R}^m$                        | Lower $\to$ Higher dim                           |
| Square          | $A \in \mathbb{R}^{n \times n}$                                 | $\mathbb{R}^n$ | $\mathbb{R}^n$                        | Same dim (may still collapse)                    |
| Composition     | $B \in \mathbb{R}^{p \times m},\ A \in \mathbb{R}^{m \times n}$ | $\mathbb{R}^n$ | $\mathbb{R}^p$                        | $\mathbb{R}^n \to \mathbb{R}^m \to \mathbb{R}^p$ |
| Rank constraint | $\text{rank}(A) \leq \min(m, n)$                                | $\mathbb{R}^n$ | $\text{im}(A) \subseteq \mathbb{R}^m$ | True image dim $\leq \min(m, n)$                 |
### Matrix $\times$ Matrix
The inner dimensionality must be the same (columns of $A$ and rows of $B$)

|Scenario|Matrices|Input|Output|Map|
|---|---|---|---|---|
|General|$A \in \mathbb{R}^{m \times k},\ B \in \mathbb{R}^{k \times n}$|$B$ maps $\mathbb{R}^n \to \mathbb{R}^k$|$AB \in \mathbb{R}^{m \times n}$|$\mathbb{R}^n \to \mathbb{R}^k \to \mathbb{R}^m$|
|Square|$A, B \in \mathbb{R}^{n \times n}$|$\mathbb{R}^n$|$AB \in \mathbb{R}^{n \times n}$|$\mathbb{R}^n \to \mathbb{R}^n$|
|Rank constraint|$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$|$\mathbb{R}^n$|$\text{im}(AB) \subseteq \mathbb{R}^m$|True image dim $\leq \min(m, k, n)$|
|Tall $\times$ Wide|$A \in \mathbb{R}^{m \times k},\ m > k,\ B \in \mathbb{R}^{k \times n},\ k < n$|$\mathbb{R}^n$|$AB \in \mathbb{R}^{m \times n}$|Bottleneck at $\mathbb{R}^k$|
