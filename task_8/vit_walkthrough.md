# Vision Transformer (ViT) for MNIST — Implementation Walkthrough

> **Context**: This walkthrough explains the ViT implementation in
> [`vit_mnist.ipynb`](file:///home/thiru/qmlhep26-tasks/task_8/vit_mnist.ipynb).
> Written from a CS student perspective for GSoC 2026 (QMLHEP).

---

## 1. What is a Vision Transformer?

A **Vision Transformer** (ViT) applies the Transformer architecture — originally
designed for NLP — directly to images. Instead of processing words, it processes
**image patches** as tokens.

The key insight from Dosovitskiy et al. (2020):

> An image is just a sequence of patches. Treat each patch like a word,
> and let self-attention figure out the spatial relationships.

### ViT vs CNN — the fundamental difference

| | CNN | ViT |
|---|---|---|
| **Spatial bias** | Built-in (convolutions are local) | None — learned from data |
| **Receptive field** | Grows with depth | Global from layer 1 (attention) |
| **Data efficiency** | Good on small datasets | Needs large datasets |
| **Scalability** | Diminishing returns at scale | Scales extremely well |

---

## 2. Architecture Overview

```
Input Image (1×28×28)
        │
   ┌────▼─────┐
   │ Patch     │  Split into 4×4 patches → 49 patches
   │ Embedding │  Each patch: 16 pixels → projected to 64 dims
   └────┬──────┘
        │
   ┌────▼──────────────┐
   │ [CLS] + Pos Embed │  Prepend learnable CLS token → 50 tokens
   │                    │  Add learnable positional embeddings
   └────┬──────────────┘
        │
   ┌────▼──────────────┐
   │ Transformer       │ × 6 layers
   │  ├─ LayerNorm     │
   │  ├─ MHSA (8 heads)│
   │  ├─ LayerNorm     │
   │  └─ MLP (GELU)    │
   └────┬──────────────┘
        │
   ┌────▼─────┐
   │ LayerNorm│
   └────┬─────┘
        │
   ┌────▼─────┐
   │ CLS Head │  Linear(64 → 10) on CLS token
   └────┬─────┘
        │
     10 logits → class prediction
```

---

## 3. Patch Embedding — Step by Step

### 3.1 Splitting into patches

A 28×28 image with patch size 4 gives us:
- $(28/4)^2 = 49$ patches
- Each patch is $4 \times 4 \times 1 = 16$ pixels

### 3.2 Linear projection

Each 16-dim patch vector is projected to 64 dims via a linear layer.
In the code, we use `nn.Conv2d` with `kernel_size=stride=4` — this is
mathematically equivalent to extracting non-overlapping patches and
applying a linear layer, but faster on GPU.

```python
self.proj = nn.Conv2d(1, 64, kernel_size=4, stride=4)
# Output: (B, 64, 7, 7) → flatten → (B, 49, 64)
```

---

## 4. CLS Token & Positional Embeddings

### CLS token
A learnable parameter $\text{cls} \in \mathbb{R}^{1 \times D}$ prepended to the
patch sequence. After the transformer, the CLS token's output is used for
classification — it "attends" to all patches and aggregates global information.

### Positional embeddings
Learnable vectors $\text{pos} \in \mathbb{R}^{(N+1) \times D}$ added to the
token sequence. Without these, the model can't distinguish patch positions
(attention is permutation-equivariant by default).

```
tokens = [CLS, patch_1, patch_2, ..., patch_49]  # shape: (B, 50, 64)
tokens = tokens + pos_embed                        # element-wise add
```

---

## 5. Multi-Head Self-Attention (MHSA)

### 5.1 The attention formula

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- **Q** (queries), **K** (keys), **V** (values) are linear projections of input
- $\sqrt{d_k}$ scaling prevents dot products from growing too large
- Softmax produces attention weights — which tokens attend to which

### 5.2 Multi-head

Split embedding into $h=8$ heads, each operating on $d/h = 8$ dimensions:
- Allows different heads to learn different relationships
- Concat outputs and project back to $D$

### 5.3 Intuition

Each head learns to "look at" different aspects:
- Head 1 might attend to nearby patches (local texture)
- Head 2 might attend to distant patches (global shape)
- The model learns which attention patterns are useful

---

## 6. MLP Block

A simple two-layer feed-forward network applied to each token independently:

$$\text{MLP}(x) = W_2 \cdot \text{GELU}(W_1 x + b_1) + b_2$$

- Hidden dim = $D \times \text{ratio} = 64 \times 2 = 128$
- GELU activation (smoother than ReLU, standard in transformers)
- Dropout for regularization

---

## 7. Pre-LN Transformer Block

We use **Pre-LN** style (norm before each sub-layer), which is more stable
than the original Post-LN:

$$x = x + \text{MHSA}(\text{LayerNorm}(x))$$
$$x = x + \text{MLP}(\text{LayerNorm}(x))$$

Residual connections ensure gradients flow through deep networks.

---

## 8. Training Details

| Setting | Value | Why |
|---------|-------|-----|
| Optimizer | AdamW | Standard for transformers |
| LR | 3e-4 | Typical AdamW LR for ViTs |
| LR Schedule | Cosine Annealing | Smooth decay |
| Weight Decay | 1e-4 | Light regularization |
| Batch Size | 256 | GPU-efficient |
| Epochs | 15 | ViT needs more training than CNN on small data |
| Dropout | 0.1 | Prevents overfitting on 60k samples |

---

## 9. Why ViT Struggles on MNIST

**Expected:** ~97-98% accuracy (vs ~99%+ for simple CNN).

The gap exists because:
1. **No spatial inductive bias** — CNNs know that nearby pixels relate;
   ViTs must learn this from data alone.
2. **Small dataset** — ViTs were designed for ImageNet (1.3M images).
   MNIST (60k) doesn't provide enough signal for attention to discover
   optimal spatial patterns.
3. **Patch tokenization loses fine detail** — 4×4 patches discard
   sub-patch spatial structure that convolutions preserve.

Despite this, ViT is hugely valuable to study because:
- Attention is the foundation of modern ML
- ViTs dominate large-scale vision (ImageNet, detection, segmentation)
- The attention mechanism maps naturally to quantum inner products

---

## 10. Key Code Patterns

### Conv2d as patch embedding (efficient)
```python
nn.Conv2d(1, 64, kernel_size=4, stride=4)
# Equivalent to: extract patches → linear projection
# But runs as a single GPU-optimized op
```

### QKV in one linear layer
```python
self.qkv = nn.Linear(embed_dim, embed_dim * 3)
# Compute Q, K, V in one operation, then split
```

### Pre-LN residual
```python
x = x + self.attn(self.norm1(x))  # norm BEFORE attention
x = x + self.mlp(self.norm2(x))   # norm BEFORE MLP
```

---

## References

1. **ViT**: Dosovitskiy et al., "An Image is Worth 16×16 Words" (2020). [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
2. **Transformer**: Vaswani et al., "Attention Is All You Need" (2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
