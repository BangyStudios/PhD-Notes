## Definition
The Kronecker product of matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{p \times q}$ is a block matrix where every entry of $A$ scales a full copy of $B$:
$$
A \otimes B = \begin{bmatrix} a_{11}B & \cdots & a_{1n}B \\ \vdots & \ddots & \vdots \\ a_{m1}B & \cdots & a_{mn}B \end{bmatrix} \in \mathbb{R}^{mp \times nq}
$$
where $a_{ij} \in \mathbb{R}$ are the entries of $A$. The output is larger than both inputs — unlike the element-wise product $\odot$, the dimensions need not match.

### Example
$$
\begin{bmatrix}1 & 2\\ 3 & 4\end{bmatrix} \otimes \begin{bmatrix}0 & 5\\ 6 & 7\end{bmatrix}
=
\begin{bmatrix}
1\cdot\begin{bmatrix}0&5\\6&7\end{bmatrix} & 2\cdot\begin{bmatrix}0&5\\6&7\end{bmatrix}\\[6pt]
3\cdot\begin{bmatrix}0&5\\6&7\end{bmatrix} & 4\cdot\begin{bmatrix}0&5\\6&7\end{bmatrix}
\end{bmatrix}
=
\begin{bmatrix}
0 & 5 & 0 & 10\\
6 & 7 & 12 & 14\\
0 & 15 & 0 & 20\\
18 & 21 & 24 & 28
\end{bmatrix}
$$
A $2\times2$ kronecker'd with a $2\times2$ yields a $4\times4$.

### Generalization: 3rd-order tensor
Chaining $\otimes$ is associative, so three matrices produce a single large matrix that encodes a 3rd-order tensor:
$$
A \otimes B \otimes C = (A \otimes B) \otimes C
$$
For $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{p \times q}$, $C \in \mathbb{R}^{r \times s}$, the result is in $\mathbb{R}^{mpr \times nqs}$.

Using $2\times2$ matrices throughout:
$$
A = \begin{bmatrix}1&0\\0&1\end{bmatrix},\quad
B = \begin{bmatrix}1&2\\3&4\end{bmatrix},\quad
C = \begin{bmatrix}0&1\\1&0\end{bmatrix}
$$

Step 1 — $A \otimes B$:
$$
A \otimes B =
\begin{bmatrix}
1\cdot B & 0\cdot B\\
0\cdot B & 1\cdot B
\end{bmatrix}
=
\begin{bmatrix}
1&2&0&0\\
3&4&0&0\\
0&0&1&2\\
0&0&3&4
\end{bmatrix}
$$

Step 2 — $(A \otimes B) \otimes C$: each scalar entry $e_{ij}$ of $A\otimes B$ is replaced by $e_{ij}\cdot C$, giving an $8\times8$ matrix. Keeping the $C$ blocks visible exposes the three tensor dimensions:
$$
(A\otimes B)\otimes C =
\begin{bmatrix}
1\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 2\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 0\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 0\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix}\\[8pt]
3\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 4\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 0\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 0\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix}\\[8pt]
0\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 0\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 1\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 2\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix}\\[8pt]
0\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 0\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 3\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix} & 4\cdot\begin{bmatrix}0&1\\1&0\end{bmatrix}
\end{bmatrix}
$$
The three levels of nesting correspond to the three tensor dimensions: the $2\times2$ block-of-blocks layout comes from $A$, the $2\times2$ scalar arrangement within each block comes from $B$, and each leaf $C$ carries the innermost dimension.
Three $2\times2$ matrices yield an $8\times8$. In general, $k$ matrices of size $n\times n$ yield $n^k \times n^k$.
