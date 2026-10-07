# Mathematical Foundations of Row-Wise and Column-Wise Matrix Operations

This document establishes a rigorous mathematical framework defining **row-wise** and **column-wise** operations using structural tensor/matrix manipulations—specifically vectorization, tensor products (Kronecker products), and matrix concatenation.

---

## 1. Vector Space and Structural Definitions

Let $\mathbb{R}^{m \times n}$ denote the vector space of $m \times n$ matrices over the real field. A matrix $A \in \mathbb{R}^{m \times n}$ can be fundamentally understood by its constituent structural components: its row vectors and its column vectors.

### Column-Wise Partitioning
A matrix $A$ can be expressed as a horizontal concatenation of $n$ column vectors $a_{\bullet, j} \in \mathbb{R}^{m \times 1}$:
$$A = \begin{bmatrix} | & | & & | \\ a_{\bullet, 1} & a_{\bullet, 2} & \cdots & a_{\bullet, n} \\ | & | & & | \end{bmatrix} = \Big[ a_{\bullet, 1} \ \Big| \ a_{\bullet, 2} \ \Big| \ \cdots \ \Big| \ a_{\bullet, n} \Big]$$

### Row-Wise Partitioning
Alternatively, $A$ can be expressed as a vertical concatenation of $m$ row vectors $a_{i, \bullet} \in \mathbb{R}^{1 \times n}$:
$$A = \begin{bmatrix} \rule[0.5ex]{2em}{0.4pt} & a_{1, \bullet} & \rule[0.5ex]{2em}{0.4pt} \\ \rule[0.5ex]{2em}{0.4pt} & a_{2, \bullet} & \rule[0.5ex]{2em}{0.4pt} \\ & \vdots & \\ \rule[0.5ex]{2em}{0.4pt} & a_{m, \bullet} & \rule[0.5ex]{2em}{0.4pt} \end{bmatrix} = \begin{bmatrix} a_{1, \bullet} \\ \hline a_{2, \bullet} \\ \hline \vdots \\ \hline a_{m, \bullet} \end{bmatrix}$$

---

## 2. Formalization via Concatenation Operators

To generalize row-wise and column-wise mappings, we define explicit concatenation operators that map sequences of vectors to matrices.

Let $\mathcal{C}_{\text{col}}: \prod_{j=1}^n \mathbb{R}^m \to \mathbb{R}^{m \times n}$ be the **column-concatenation operator**:
$$\mathcal{C}_{\text{col}}\left( \{x_1, x_2, \dots, x_n\} \right) = \sum_{j=1}^n x_j e_j^T$$
where $e_j \in \mathbb{R}^n$ is the $j$-th standard basis vector of $\mathbb{R}^n$.

Let $\mathcal{C}_{\text{row}}: \prod_{i=1}^m \mathbb{R}^n \to \mathbb{R}^{m \times n}$ be the **row-concatenation operator**:
$$\mathcal{C}_{\text{row}}\left( \{y_1, y_2, \dots, y_m\} \right) = \sum_{i=1}^m u_i y_i^T$$
where $u_i \in \mathbb{R}^m$ is the $i$-th standard basis vector of $\mathbb{R}^m$.

---

## 3. Definition of Monadic Operations

Let $f: \mathbb{R}^d \to \mathbb{R}^k$ be a vector-valued function representing a baseline operation. We extend $f$ to act on matrices in either a column-wise or row-wise manner.

### Column-Wise Monadic Operation
A column-wise operation applying $f: \mathbb{R}^m \to \mathbb{R}^k$ to a matrix $A \in \mathbb{R}^{m \times n}$ yields a matrix $F_{\text{col}}(A) \in \mathbb{R}^{k \times n}$ defined by applying $f$ to each column independently and concatenating horizontally:
$$F_{\text{col}}(A) = \mathcal{C}_{\text{col}}\left( \left\{ f(a_{\bullet, 1}), f(a_{\bullet, 2}), \dots, f(a_{\bullet, n}) \right\} \right) = \sum_{j=1}^n f(A e_j) e_j^T$$

### Row-Wise Monadic Operation
A row-wise operation applying $f: \mathbb{R}^n \to \mathbb{R}^k$ to a matrix $A \in \mathbb{R}^{m \times n}$ yields a matrix $F_{\text{row}}(A) \in \mathbb{R}^{m \times k}$ defined by applying $f$ to the transpose of each row independently and concatenating vertically:
$$F_{\text{row}}(A) = \mathcal{C}_{\text{row}}\left( \left\{ f(a_{1, \bullet}^T)^T, f(a_{2, \bullet}^T)^T, \dots, f(a_{m, \bullet}^T)^T \right\} \right) = \sum_{i=1}^m u_i f(A^T u_i)^T$$

---

## 4. Definition of Binary Operations (Broadcasting)

Let $g: \mathbb{R} \times \mathbb{R} \to \mathbb{R}$ be an element-wise binary operator (e.g., addition, multiplication). Let $v$ be a vector operand. Binary operations across axes are formally modeled via the Kronecker product $\otimes$ combined with concatenation.

### Column-Wise Binary Operation (Vector is $m \times 1$)
Let $v \in \mathbb{R}^{m \times 1}$. The operation applies $g$ between $v$ and every column vector of $A$:
$$G_{\text{col}}(A, v) = \mathcal{C}_{\text{col}}\left( \left\{ g(a_{\bullet, 1}, v), g(a_{\bullet, 2}, v), \dots, g(a_{\bullet, n}, v) \right\} \right)$$

In linear algebraic terms, if $g$ is standard addition ($+$), this is equivalent to broadcasting via the rank-one outer product matrix:
$$A +_{\text{col}} v = A + \mathcal{C}_{\text{col}}(\{v, v, \dots, v\}) = A + v \mathbf{1}_n^T$$
where $\mathbf{1}_n \in \mathbb{R}^n$ is a vector of all ones.

### Row-Wise Binary Operation (Vector is $1 \times n$)
Let $v^T \in \mathbb{R}^{1 \times n}$. The operation applies $g$ between $v^T$ and every row vector of $A$:
$$G_{\text{row}}(A, v^T) = \mathcal{C}_{\text{row}}\left( \left\{ g(a_{1, \bullet}, v^T), g(a_{2, \bullet}, v^T), \dots, g(a_{m, \bullet}, v^T) \right\} \right)$$

If $g$ is standard addition ($+$), this translates to:
$$A +_{\text{row}} v^T = A + \mathcal{C}_{\text{row}}(\{v^T, v^T, \dots, v^T\}) = A + \mathbf{1}_m v^T$$
where $\mathbf{1}_m \in \mathbb{R}^m$.

---

## 5. Algebraic Transformation via Vectorization

The relationship between column-wise and row-wise orientations can be perfectly mapped using the vectorization operator $\operatorname{vec}(\cdot)$ and the commutation matrix $K_{mn}$.

Let $\operatorname{vec}: \mathbb{R}^{m \times n} \to \mathbb{R}^{mn}$ stack columns vertically:
$$\operatorname{vec}(A) = \begin{bmatrix} a_{\bullet, 1} \\ a_{\bullet, 2} \\ \vdots \\ a_{\bullet, n} \end{bmatrix}$$

The commutation matrix $K_{mn} \in \mathbb{R}^{mn \times mn}$ is a permutation matrix that transforms $\operatorname{vec}(A)$ into $\operatorname{vec}(A^T)$:
$$K_{mn} \operatorname{vec}(A) = \operatorname{vec}(A^T)$$

### Duality Theorem
Any row-wise operation can be cast as a column-wise operation under transposition duality:
$$F_{\text{row}}(A) = \left[ F_{\text{col}}(A^T) \right]^T$$

Expressed entirely in vector space mapping configurations:
$$\operatorname{vec}(F_{\text{row}}(A)) = K_{km} \operatorname{vec}\left( F_{\text{col}}\left( \operatorname{vec}^{-1}(K_{mn} \operatorname{vec}(A)) \right) \right)$$

---

## 6. Reduction Operations

Reduction operators map an input vector down to a scalar: $r: \mathbb{R}^d \to \mathbb{R}$ (e.g., $\sum$, $\max$, $\operatorname{mean}$).

### Column-Wise Reduction (Collapsing Rows)
Reduces each column independently, mapping $\mathbb{R}^{m \times n} \to \mathbb{R}^{1 \times n}$:
$$R_{\text{col}}(A) = \Big[ r(a_{\bullet, 1}) \ \Big| \ r(a_{\bullet, 2}) \ \Big| \ \cdots \ \Big| \ r(a_{\bullet, n}) \Big]$$

### Row-Wise Reduction (Collapsing Columns)
Reduces each row independently, mapping $\mathbb{R}^{m \times n} \to \mathbb{R}^{m \times 1}$:
$$R_{\text{row}}(A) = \begin{bmatrix} r(a_{1, \bullet}^T) \\ r(a_{2, \bullet}^T) \\ \vdots \\ r(a_{m, \bullet}^T) \end{bmatrix}$$