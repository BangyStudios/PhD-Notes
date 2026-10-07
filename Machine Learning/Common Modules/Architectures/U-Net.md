A U-Net is an encoder–decoder [CNN](CNN.md) architecture with symmetric skip connections that concatenate encoder feature maps directly into the corresponding decoder layers, preserving fine-grained spatial detail for dense prediction tasks such as image segmentation.

---
## Definition
Let the encoder produce feature maps $\{E_1, E_2, \ldots, E_N\}$ at successively lower resolutions, and let the decoder produce maps $\{D_N, \ldots, D_1\}$ at successively higher resolutions. Each decoder stage is:
$$
D_\ell = \mathrm{Conv}\!\left([\,\mathrm{Up}(D_{\ell+1})\,;\, E_\ell\,]\right)
$$
- **$\mathrm{Up}(\cdot)$:** Upsampling (transposed convolution or bilinear interpolation) that doubles spatial resolution.
- **$[\,\cdot\,;\,\cdot\,]$:** Channel-wise concatenation of the upsampled decoder map with the encoder skip map.
- **$\mathrm{Conv}$:** One or more $3 \times 3$ convolution + activation blocks.

The final layer applies a $1 \times 1$ convolution to map to the number of output classes.

---
## Example
Four-level U-Net with spatial sizes $64 \to 32 \to 16 \to 8$ (encoder) and $8 \to 16 \to 32 \to 64$ (decoder); channels double on each encoder step and halve on each decoder step.

| Stage | Operation | Spatial size | Channels |
|---|---|---|---|
| $E_1$ | Conv $\times 2$ | $64 \times 64$ | 64 |
| $E_2$ | MaxPool + Conv $\times 2$ | $32 \times 32$ | 128 |
| $E_3$ | MaxPool + Conv $\times 2$ | $16 \times 16$ | 256 |
| Bottleneck | MaxPool + Conv $\times 2$ | $8 \times 8$ | 512 |
| $D_3$ | Up + concat($E_3$) + Conv $\times 2$ | $16 \times 16$ | 256 |
| $D_2$ | Up + concat($E_2$) + Conv $\times 2$ | $32 \times 32$ | 128 |
| $D_1$ | Up + concat($E_1$) + Conv $\times 2$ | $64 \times 64$ | 64 |
| Output | Conv $1 \times 1$ | $64 \times 64$ | $C$ classes |

At $D_3$, the upsampled bottleneck ($8 \to 16$, 512 channels) is concatenated with $E_3$ (256 channels) to give a 768-channel tensor, then convolved down to 256 channels. The skip connection restores high-resolution detail that was lost during downsampling.
