# SineKAN for MNIST — Implementation Walkthrough

> **Context**: This walkthrough explains the SineKAN implementation in
> [`sinekan_mnist.py`](file:///home/thiru/qmlhep26-tasks/sinekan_mnist.py).
> Written from a CS student perspective for GSoC 2026 (QMLHEP).

---

## 1. What is a KAN?

A **Kolmogorov-Arnold Network** (KAN) is an alternative to the standard MLP, based on the
**Kolmogorov-Arnold Representation Theorem** (1957):

> Any continuous function $f: [0,1]^n \to \mathbb{R}$ can be written as:
>
> $$f(x_1, \ldots, x_n) = \sum_{j=0}^{2n} \Phi_j\left(\sum_{i=1}^{n} \varphi_{ij}(x_i)\right)$$

The key insight: **you only need compositions of univariate functions** to represent any
multivariate function.

### MLP vs KAN — the fundamental difference

| | MLP | KAN |
|---|---|---|
| **Where nonlinearity lives** | On nodes (ReLU, sigmoid) | On edges (learnable functions) |
| **Edge operation** | Fixed: $w \cdot x$ | Learnable: $\varphi(x)$ |
| **Node operation** | Apply activation $\sigma(\cdot)$ | Simple summation $\sum$ |

```
MLP:                              KAN:
  x₁ ──w₁──►[σ]──►                 x₁ ──φ₁₁(·)──►[Σ]──►
  x₂ ──w₂──►[σ]──►                 x₂ ──φ₂₁(·)──►[Σ]──►
        fixed σ                        learnable φ
```

---

## 2. Why SineKAN? (not B-splines)

The original KAN paper (Liu et al., 2024) used **B-splines** to parameterize $\varphi$.
**SineKAN** (Reinhardt et al., 2024) replaces them with sine/cosine series:

$$\varphi(x) = \sum_{k=1}^{K} \left[ a_k \sin(kx) + b_k \cos(kx) \right]$$

**Why this is a better choice for us:**

1. **Speed**: No grid management, no knot vectors — just matrix ops
2. **Smooth everywhere**: Infinitely differentiable (B-splines have limited smoothness)
3. **Fewer hyperparameters**: Only need to set $K$ (number of frequencies)
4. **Quantum-friendly**: $\sin$ and $\cos$ map directly to rotation gates ($R_Y$, $R_Z$)
5. **Mentor alignment**: One of our mentors co-authored the SineKAN paper

---

## 3. The SineKAN Layer — Step by Step

### 3.1 Setup

For a layer mapping $d_{in} \to d_{out}$:
- There are $d_{in} \times d_{out}$ edges
- Each edge has $2K$ learnable parameters: $a_1, \ldots, a_K$ and $b_1, \ldots, b_K$
- Total params in this layer: $d_{in} \times d_{out} \times 2K$

### 3.2 Forward Pass (for one sample)

Given input $\mathbf{x} \in \mathbb{R}^{d_{in}}$:

**Step 1** — Create frequency vector:
```
k = [1, 2, 3, ..., K]
```

**Step 2** — Compute products $k \cdot x_i$ for every input $i$ and frequency $k$:
```
kx[i, k] = k * x[i]     →  shape: (d_in, K)
```

**Step 3** — Apply trig functions:
```
sin_terms[i, k] = sin(kx[i, k])
cos_terms[i, k] = cos(kx[i, k])
```

**Step 4** — Weight and aggregate to get output $y_j$:

$$y_j = \sum_{i=1}^{d_{in}} \sum_{k=1}^{K} \left[ a_{k,i,j} \cdot \sin(k \cdot x_i) + b_{k,i,j} \cdot \cos(k \cdot x_i) \right]$$

In code, this is done efficiently with `torch.einsum`:
```python
y = einsum("oik,bik->bo", a_coeffs, sin_terms) + \
    einsum("oik,bik->bo", b_coeffs, cos_terms)
```

### 3.3 Intuition

Each edge learns **which frequencies matter** for its input-output relationship.
- High $a_k$ for large $k$ → edge captures high-frequency patterns
- Large $b_1$ with small other coefficients → edge is roughly a cosine (smooth, slow-varying)
- The network jointly learns all edges to minimize classification loss

---

## 4. Full Architecture

```
Input (784)          SineKAN Layer 1       LayerNorm       SineKAN Layer 2       Output
━━━━━━━━━━          ━━━━━━━━━━━━━━━       ━━━━━━━━━       ━━━━━━━━━━━━━━━       ━━━━━━
                     784×64 edges                          64×10 edges
 [28×28 img]  ──►   each has 2K=16    ──►  Normalize  ──►  each has 2K=16   ──►  [10 logits]
  flattened          sin/cos params                        sin/cos params         → argmax
                                                                                  = digit
```

**Parameter count** (K=8):
- Layer 1: $784 \times 64 \times 16 = 802,816$ params + 128 (LayerNorm)
- Layer 2: $64 \times 10 \times 16 = 10,240$ params
- **Total: ~813,184** params

Compare MLP with same shape:
- Layer 1: $784 \times 64 + 64 = 50,240$ (weights + bias) + 128 (LayerNorm)
- Layer 2: $64 \times 10 + 10 = 650$
- **Total: ~51,018** params

> **Honest note**: SineKAN has ~16× more parameters than MLP for the same layer
> dimensions. Each edge stores $2K$ values instead of 1 weight. This is the
> trade-off for having learnable nonlinear functions on every edge.

---

## 5. Training Details

| Setting | Value | Why |
|---------|-------|-----|
| Optimizer | AdamW | Good default, weight decay helps regularize |
| LR | 1e-3 | Standard for Adam-family optimizers |
| LR Schedule | Cosine Annealing | Smooth decay, avoids sharp drops |
| Weight Decay | 1e-4 | Light regularization |
| Batch Size | 256 | Balance between speed and gradient noise |
| Epochs | 10 | Enough for MNIST convergence |
| Normalization | LayerNorm | Stabilizes activations between KAN layers |
| Input Normalization | MNIST mean=0.1307, std=0.3081 | Standard preprocessing |

**Loss function**: `CrossEntropyLoss` (includes softmax internally)

$\text{Loss} = -\sum_{c=1}^{10} y_c \log(\hat{y}_c)$

where $y_c$ is the true label (one-hot) and $\hat{y}_c$ is the predicted probability.

---

## 6. Expected Results & Interpretation

**Typical results** (on Kaggle GPU):

| Model | Test Accuracy | Training Time | Parameters |
|-------|--------------|---------------|------------|
| SineKAN | ~96-97% | ~120-180s | ~813K |
| MLP | ~97-98% | ~20-40s | ~51K |

**What this tells us**:
- SineKAN achieves competitive accuracy but is **slower and more parameter-heavy**
- For MNIST (a relatively simple task), the extra expressiveness of KAN doesn't provide a
  significant accuracy boost over a well-tuned MLP
- The value of KANs lies in **different** domains: function approximation, interpretability,
  scientific computing — not necessarily raw classification performance

---

## 7. Key Code Patterns to Learn From

### Einsum for batched tensor operations
```python
# "oik,bik->bo" means:
#   o = output dimension
#   i = input dimension
#   k = frequency dimension
#   b = batch dimension
# Sum over i and k, keeping b and o → (batch, output) tensor
torch.einsum("oik,bik->bo", coefficients, basis_values)
```

### Xavier-like initialization for stability
```python
scale = 1.0 / (in_features * num_frequencies) ** 0.5
self.a_coeffs = nn.Parameter(torch.randn(...) * scale)
```
Without proper scaling, the sum over many sine terms would explode or vanish.

### LayerNorm between KAN layers
Essential because the output of SineKAN layers can have arbitrary scale
(sine outputs are bounded by [-1,1] but the weighted sum isn't).

---

## References

1. **KAN**: Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024). [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
2. **SineKAN**: Reinhardt et al., "SineKAN: Kolmogorov-Arnold Networks Using Sinusoidal Activation Functions" (2024). [arXiv:2407.04149](https://arxiv.org/abs/2407.04149)
3. **Kolmogorov (1957)**: "On the representation of continuous functions of several variables by superposition of continuous functions of one variable and addition"
