---
tags:
  - needs-review
  - ai-suspected
ai-review-score: 17.8
ai-review-flagged: 2026-10-07
---
> [!warning] Under review
> Flagged by an AI-text heuristic (score 17.8/100; note last modified 2026-07-02) as possibly AI-written. Verify the content and rewrite in your own words. When done, delete this callout and the `needs-review` and `ai-suspected` tags.

A **shapelet** is a time-series subsequence that is maximally discriminative for a target class, enabling classification by measuring how closely each series matches a set of learned local patterns rather than comparing series globally.

---
## Definition
### Subsequence Distance
Given a time series $T = [t_1, \ldots, t_n]$ and a shapelet $S = [s_1, \ldots, s_l]$ with $l \leq n$, the **subsequence** of $T$ starting at position $i$ is:
$$
T_{i,l} = [t_i,\; t_{i+1},\; \ldots,\; t_{i+l-1}]
$$

The **subsequence distance** is the minimum Euclidean distance over all positions:
$$
\text{SubSeqDist}(S, T) = \min_{1 \leq i \leq n-l+1} \left\| S - T_{i,l} \right\|_2
$$ where:
- **$S$:** Shapelet of length $l$; the candidate discriminative pattern.
- **$T_{i,l}$:** Subsequence of $T$ of length $l$ starting at index $i$.
- **$\|\cdot\|_2$:** Euclidean norm over the $l$ coordinates.

A small $\text{SubSeqDist}$ means $S$ appears (approximately) somewhere in $T$; a large value means it is absent.
### Shapelet Selection via Information Gain
For a shapelet $S$ and a distance threshold $d^*$, the training set $D$ is split into two groups:
$$
D_{\leq} = \{T_i \in D : \text{SubSeqDist}(S,\, T_i) \leq d^*\}, \qquad
D_{>} = \{T_i \in D : \text{SubSeqDist}(S,\, T_i) > d^*\}
$$

The best $(S, d^*)$ pair maximizes **information gain**:
$$
\text{IG}(S, d^*) = H(D) - \frac{|D_{\leq}|}{|D|}\,H(D_{\leq}) - \frac{|D_{>}|}{|D|}\,H(D_{>})
$$ where:
- **$H(\cdot)$:** Shannon entropy of the class-label distribution in a set.
- **$|D|,\,|D_{\leq}|,\,|D_{>}|$:** Cardinalities of the full set and the two partitions.

The optimal threshold $d^*$ for a fixed $S$ is found by sorting all training series by their distance to $S$ and scanning the $n-1$ split points.
### Shapelet Transform
The **shapelet transform** converts each time series into a fixed-length feature vector. Given $K$ selected shapelets $S_1, \ldots, S_K$:
$$
\Phi(T) = \begin{bmatrix}\text{SubSeqDist}(S_1, T) \\ \vdots \\ \text{SubSeqDist}(S_K, T)\end{bmatrix} \in \mathbb{R}^K
$$

This decouples shapelet discovery from classification: the resulting $N \times K$ feature matrix can be passed to any standard classifier (e.g., Random Forest, SVM). Each feature encodes the absence or presence of one discriminative local pattern.

---
## Example
Two-class dataset ($n = 4$, two series per class):
$$
T_1 = [1,2,3,2],\quad T_2 = [1,3,2,1]\quad\text{(Class A)}
$$
$$
T_3 = [3,2,1,2],\quad T_4 = [3,1,2,1]\quad\text{(Class B)}
$$
Candidate shapelet $S = [1, 2, 3]$ (a rising ramp), length $l = 3$.
### Step 1: Compute SubSeqDist for each series
Each series of length 4 has $4 - 3 + 1 = 2$ windows of length 3.

For $T_1 = [1,2,3,2]$:
$$
\|S - [1,2,3]\| = 0,\qquad \|S - [2,3,2]\| = \sqrt{1+1+1} = \sqrt{3}
$$
$$
\text{SubSeqDist}(S, T_1) = 0
$$
For $T_2 = [1,3,2,1]$:
$$
\|S - [1,3,2]\| = \sqrt{0+1+1} = \sqrt{2},\qquad \|S - [3,2,1]\| = \sqrt{4+0+4} = 2\sqrt{2}
$$
$$
\text{SubSeqDist}(S, T_2) = \sqrt{2} \approx 1.41
$$
For $T_3 = [3,2,1,2]$:
$$
\|S - [3,2,1]\| = \sqrt{4+0+4} = 2\sqrt{2},\qquad \|S - [2,1,2]\| = \sqrt{1+1+1} = \sqrt{3}
$$
$$
\text{SubSeqDist}(S, T_3) = \sqrt{3} \approx 1.73
$$
For $T_4 = [3,1,2,1]$:
$$
\|S - [3,1,2]\| = \sqrt{4+1+1} = \sqrt{6},\qquad \|S - [1,2,1]\| = \sqrt{0+0+4} = 2
$$
$$
\text{SubSeqDist}(S, T_4) = 2
$$
### Step 2: Select threshold and compute information gain
Sorting by distance: $T_1(0),\; T_2(1.41),\; T_3(1.73),\; T_4(2.00)$.

Choosing threshold $d^* = 1.5$ splits the dataset cleanly:
$$
D_{\leq} = \{T_1,\, T_2\}\;\text{(both Class A)},\qquad D_{>} = \{T_3,\, T_4\}\;\text{(both Class B)}
$$

Entropy of the full dataset (2 Class A, 2 Class B, $p = 0.5$ each):
$$
H(D) = -0.5\log_2 0.5 - 0.5\log_2 0.5 = 1
$$
Both partitions are pure, so $H(D_{\leq}) = H(D_{>}) = 0$:
$$
\boxed{\text{IG}(S,\; d^* = 1.5) = 1 - \tfrac{2}{4}\cdot 0 - \tfrac{2}{4}\cdot 0 = 1}
$$
Maximum information gain — $S$ perfectly separates the two classes.
### Step 3: Apply the shapelet transform
With $S$ as the sole selected shapelet ($K = 1$), each series maps to a scalar feature:
$$
\Phi = \begin{bmatrix}0 \\ \sqrt{2} \\ \sqrt{3} \\ 2\end{bmatrix},
\qquad
\mathbf{y} = \begin{bmatrix}\text{A} \\ \text{A} \\ \text{B} \\ \text{B}\end{bmatrix}
$$
A classifier on $\Phi$ simply thresholds at $d^* = 1.5$: series with distance $\leq 1.5$ are Class A; those with distance $> 1.5$ are Class B. With $K > 1$ shapelets the same logic extends to an $N \times K$ feature matrix fed to a richer classifier.
