A personal PhD reference vault (in US English) containing notes on machine learning theory, deep learning architectures, and mathematical foundations — organized by topic and written in a consistent Markdown format.

---
## Structure

```
Topics/
├── Topic 1/
│   ├── Subtopic 1.1/
│   ├── ...
│   └── Subtopic 1.n/
├── ...
└── Topic m/
    ├── Subtopic m.1/
    └── Subtopic m.n/
```

Each subdirectory covers one topic area. Files within a topic are either standalone concept notes or a numbered sequence (e.g. `4-Forward_Pass.md`, `5-Loss_Function.md`, `6-Backpropagation.md`) that should be read in order.

---
## Note Format

Every note follows the same structure:

1. **Opening sentence** — one sentence summarizing what the concept is. No heading.
2. **`---`** — horizontal rule separating the intro from sections.
3. **`## Definition`** — formal definition, typically with LaTeX.
4. **`## Example`** — a concrete worked example with numbered steps.

There are no unnecessary newlines or line breaks after or before headings or math blocks.
### Opening Sentence
The first line is a plain sentence (no `#` heading) that states what the topic is and why it matters:

> An autoencoder is a neural network trained to reconstruct its input by compressing it through a bottleneck, forcing the network to learn a compact latent representation.

### Sections
Top-level sections use `##`, subsections use `###`. Standard section names:
- `## Definition` — formal statement of the concept
- `## Example` — worked numerical example
- `## Discussion` — interpretive notes, edge cases, or comparisons (optional)

### Notation Blocks
After an equation, a `where:` clause lists each symbol:
$$
\mathcal{L}(\phi, \theta) = \frac{1}{N}\sum_{n=1}^{N} \|\mathbf{x}_n - g_\theta(f_\phi(\mathbf{x}_n))\|^2
$$ where:
- **$f_\phi$:** Encoder mapping input to latent code.
- **$g_\theta$:** Decoder mapping latent code back to reconstruction.
- **$k$:** Bottleneck dimension.

Key terms are bolded on first use: **encoder**, **decoder**, **bottleneck**. All variables in the equations are identified and explained directly after its first use as above.

### Worked Examples
Steps use `### Step N: Description` headings and show full arithmetic:

### Step 1: Encode
$$
\mathbf{z} = W_\phi\, \mathbf{x} = 2
$$
### Step 2: Decode
$$
\hat{\mathbf{x}} = W_\theta\, \mathbf{z} = \begin{bmatrix}1 \\ 0 \\ 1\end{bmatrix}
$$

A one-sentence plain-English interpretation follows each step where the result needs explaining.

### Key Results
Boxed equations mark the most important results:
$$
\boxed{
\delta^{(\ell)} = \left(W^{(\ell+1)}\right)^{T} \delta^{(\ell+1)} \;\odot\; \sigma'^{(\ell)}\!\big(z^{(\ell)}\big)
}
$$

---
## Conventions

### File Naming
- Concept notes: `Concept Name.md` (title case, spaces allowed).
- Sequence notes: `N-Topic_Name.md` (number prefix, underscores within name).
- No abbreviations in filenames unless the abbreviation is the standard name (e.g. `CNN`, `VAE`).

### Cross-Links
Link to related notes using relative paths:

```markdown
See the [forward pass](4-Forward_Pass.md) for how activations are computed.
A [VAE](Variational%20Autoencoder.md) extends this with a prior on $\mathbf{z}$.
```

Spaces in filenames are percent-encoded (`%20`) in links.

### Math
- Inline math: `$...$`
- Display math: `$$...$$` on its own line
- Vectors: bold lowercase `$\mathbf{x}$`
- Matrices: uppercase `$W$`
- Loss: `$\mathcal{L}$`
- Hadamard product: `$\odot$`
- Layer superscripts: `$\ell$` (not `l`)

### Bullet Style
Use `-` for unordered lists. Nested bullets indent with a tab.

---
## When Editing Notes

- Match the opening sentence style: one sentence, no heading, states what and why.
- Preserve existing `where:` clause format after equations.
- Keep examples self-contained — define all variables used in the worked steps.
- Do not add a trailing summary section; the example and any discussion speak for themselves.
- Do not add comments or meta-commentary about the note itself inside the file.
