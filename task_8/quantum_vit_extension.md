# Quantum Vision Transformer (QViT) — Detailed Sketch

> **Context**: This proposes extending the classical ViT to a quantum circuit.
> Written for GSoC 2026 (QMLHEP) — research sketch, not full implementation.

---

## 1. Classical ViT Recap

```
Image → Patch Embed → [CLS]+Pos → (MHSA → MLP) × L → CLS Head → Class
```

**Components to quantize:**
1. Patch encoding
2. Attention mechanism (QKᵀ/√d → softmax → V)
3. MLP block (feed-forward network)
4. Positional encoding

---

## 2. Proposed Quantum ViT Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QUANTUM VISION TRANSFORMER                     │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │   QUANTUM     │   │  QUANTUM          │   │   MEASUREMENT    │    │
│  │   PATCH       │──►│  TRANSFORMER      │──►│   + CLS HEAD     │    │
│  │   ENCODING    │   │  LAYERS (× L)     │   │                  │    │
│  └──────────────┘   └──────────────────┘   └──────────────────┘    │
│                                                                     │
│  Classical pixels   Quantum attention      Expectation values       │
│  → qubit states     + quantum FFN          → class logits           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component-by-Component Quantum Mapping

### 3.1 Quantum Patch Encoding

**Classical**: Each 4×4 patch (16 pixels) → linear projection → D-dim vector.

**Quantum options**:

| Method | How | Qubits needed | Pros | Cons |
|--------|-----|--------------|------|------|
| **Angle encoding** | Each pixel → RY(xᵢ) rotation | 16 per patch | Simple, hardware-friendly | Linear qubit cost |
| **Amplitude encoding** | 16 pixels → amplitudes of 4 qubits | 4 per patch | Exponentially compact | Expensive state prep |
| **Hybrid** | Classical PCA/linear → 8 features → angle encode | 8 per patch | Practical compromise | Loses "fully quantum" claim |

**Recommended for GSoC**: Hybrid approach — classical dimensionality reduction
(16 → 8 features), then angle encoding onto 8 qubits per patch.

```
Patch (16 pixels) ──► Classical Linear(16→8) ──► Angle Encode onto 8 qubits
                                                  │
                                                  ▼
                                          q₀: ─ RY(z₀) ─
                                          q₁: ─ RY(z₁) ─
                                          q₂: ─ RY(z₂) ─
                                          ...
                                          q₇: ─ RY(z₇) ─
```

### 3.2 Quantum Positional Encoding

**Classical**: Learnable $\text{pos} \in \mathbb{R}^{N \times D}$ added to tokens.

**Quantum approach**: Add learnable rotation angles *before* data encoding:

$$|x_i, p_i\rangle = R_Y(\theta_i^{\text{pos}}) \cdot R_Y(x_i) |0\rangle$$

where $\theta_i^{\text{pos}}$ is a trainable parameter per position $i$.
This effectively "shifts" each qubit's encoding based on spatial position —
analogous to adding a positional vector in classical ViT.

### 3.3 Quantum Attention — The Core Challenge

**Classical attention:**
$$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V$$

This requires:
1. Computing inner products between all token pairs (QKᵀ)
2. A nonlinear normalization (softmax)
3. Weighted aggregation of values (× V)

**Quantum strategies:**

#### Option A: Swap Test for Inner Products
The **swap test** circuit estimates $|\langle \psi | \phi \rangle|^2$ —
the overlap between two quantum states. We can use this to compute
attention scores:

```
|0⟩ ── H ──●── H ── Measure
            │
|ψᵢ⟩ ──── SWAP
|ψⱼ⟩ ──── SWAP
```

P(measure 0) = ½(1 + |⟨ψᵢ|ψⱼ⟩|²) → gives us the attention score
between patches i and j.

**Problem**: Need O(N²) swap tests for N patches. For 49 patches, that's
~1200 circuits — expensive but parallelizable.

#### Option B: Parameterized Quantum Attention
Instead of explicitly computing QKᵀ, use a **parameterized quantum circuit**
that implicitly learns attention-like mixing:

```
For each pair of patch qubits (i, j):
    Apply CNOT(i, j)              ← entangle patches
    Apply RY(θ_ij) on j           ← learnable "attention weight"
    Apply CNOT(i, j)              ← disentangle

Then apply parameterized rotations (value transform)
```

This is more practical but less theoretically grounded as "attention."

#### Option C: Quantum Kernel Attention (Recommended)
Replace softmax(QKᵀ) with a **quantum kernel**:

$$K(x_i, x_j) = |\langle 0^n | U^\dagger(x_j) U(x_i) | 0^n \rangle|^2$$

This computes attention scores as quantum fidelities — a natural
generalization of dot-product attention to Hilbert space. The kernel
is computed on quantum hardware; aggregation on classical.

### 3.4 Quantum MLP Block

**Classical**: Linear → GELU → Linear

**Quantum**: Replace with a **Parameterized Quantum Circuit (PQC)**:

```
|ψ⟩ ──► [Layer of RY/RZ rotations] ──► [Entangling CNOTs] ──► 
     ──► [Layer of RY/RZ rotations] ──► [Entangling CNOTs] ──► Measure
```

The PQC acts as a universal function approximator on each token,
analogous to the classical MLP. Trainable parameters are rotation angles.

---

## 4. Full Quantum ViT Circuit (8-qubit example, 2 patches)

```
PATCH 1 ENCODING         POSITIONAL       QUANTUM ATTENTION        QUANTUM FFN         MEASURE
                          ENCODING         (entangle patches)       (PQC per token)

q₀: ─ RY(z₀) ─ RY(θ₀ᵖ) ──●──────────── RY(α₀)─RZ(β₀)───●─── ⟨Z₀⟩
q₁: ─ RY(z₁) ─ RY(θ₁ᵖ) ──●──────────── RY(α₁)─RZ(β₁)───●─── ⟨Z₁⟩
q₂: ─ RY(z₂) ─ RY(θ₂ᵖ) ──┼──●───────── RY(α₂)─RZ(β₂)───●─── ⟨Z₂⟩
q₃: ─ RY(z₃) ─ RY(θ₃ᵖ) ──┼──●───────── RY(α₃)─RZ(β₃)───●─── ⟨Z₃⟩
                            │  │
q₄: ─ RY(z₄) ─ RY(θ₄ᵖ) ──X──┼───────── RY(α₄)─RZ(β₄)───●─── ⟨Z₄⟩
q₅: ─ RY(z₅) ─ RY(θ₅ᵖ) ──X──┼───────── RY(α₅)─RZ(β₅)───●─── ⟨Z₅⟩
q₆: ─ RY(z₆) ─ RY(θ₆ᵖ) ─────X───────── RY(α₆)─RZ(β₆)───●─── ⟨Z₆⟩
q₇: ─ RY(z₇) ─ RY(θ₇ᵖ) ─────X───────── RY(α₇)─RZ(β₇)───●─── ⟨Z₇⟩

     ╰──── data ────╯  ╰─ pos ─╯  ╰── cross-patch ──╯  ╰─── PQC ───╯  ╰─ out ─╯
           (fixed)      (train)     entanglement          (trainable)

Patch 1: q₀-q₃    Patch 2: q₄-q₇
Cross-patch CNOTs create entanglement (≈ attention mixing)
```

---

## 5. What Makes This "Transformer-like" vs Just a VQC?

| Feature | Standard VQC | Quantum ViT (proposed) |
|---------|-------------|----------------------|
| Input structure | Flat vector | Patch-structured tokens |
| Cross-token interaction | Generic entanglement | Attention-like mixing between patches |
| Positional awareness | None | Learnable positional rotations |
| Token-wise processing | All qubits mixed | Per-token PQC (MLP analogue) |
| CLS aggregation | Not present | Designated output qubits |

**Genuinely transformer-like aspects:**
1. **Patch tokenization** is preserved — image is split, not flattened globally
2. **Cross-patch entanglement** mimics attention's all-pairs interaction
3. **Per-token PQC** mirrors the token-independent MLP block
4. **Positional encoding** as pre-data rotations
5. **Sequential layers** of attention + FFN, matching transformer depth

**What's different:**
- No explicit softmax — quantum fidelities replace it
- Entanglement creates correlations with no classical analogue
- Measurement collapse limits mid-circuit information extraction

---

## 6. Where Does Quantum Give Advantage (If Anywhere)?

### Potential advantages:
1. **Kernel computation**: Quantum kernels can compute attention-like
   similarity in exponentially large Hilbert space — potentially capturing
   richer feature relationships than classical dot products.
2. **Entanglement as attention**: Cross-patch entanglement creates
   all-pairs correlations in O(N) gates (vs O(N²) classical operations).
3. **Compact representation**: Amplitude encoding compresses N features
   into log₂(N) qubits — exponential compression.

### Honest caveats:
- **Readout bottleneck**: We can only extract O(n) classical bits from
  n qubits per measurement — the exponential state space collapses.
- **Barren plateaus**: Deep quantum circuits have vanishing gradients,
  making training difficult.
- **No proven advantage**: For MNIST-scale, classical ViT is more
  practical. Quantum advantage likely requires problems where the
  data naturally lives in Hilbert space.

---

## 7. Open Research Challenges

### 7.1 Scalable Quantum Attention
- Swap test costs O(N²) for N patches — same as classical.
- **Question**: Can quantum parallelism reduce attention from O(N²) to O(N)?
- Possible direction: quantum random features / linear attention analogues.

### 7.2 Barren Plateaus in Structured Circuits
- Does the patch + attention structure help avoid barren plateaus
  compared to unstructured VQCs?
- **Hypothesis**: The structured inductive bias (patch locality,
  attention sparsity) constrains the loss landscape beneficially.

### 7.3 Classical Encoding Bottleneck
- Angle encoding needs n qubits for n features — no compression.
- Amplitude encoding is exponentially efficient but hard to prepare.
- **Question**: Can we find a middle ground that's both efficient
  and hardware-practical?

---

## 8. GSoC Investigation Plan

**Phase 1 (Weeks 1-4): Foundation**
- Implement quantum patch encoding + positional rotations in PennyLane
- Build quantum kernel attention for 2-3 patches (toy problem)
- Verify that cross-patch entanglement produces meaningful attention patterns

**Phase 2 (Weeks 5-8): Hybrid QViT**
- Build full hybrid pipeline: classical patch embed → quantum attention → classical head
- Test on MNIST with 4-8 qubits per patch (simulator)
- Compare against: (a) classical ViT, (b) standard VQC, (c) quantum CNN

**Phase 3 (Weeks 9-12): Analysis**
- Gradient landscape analysis (barren plateau investigation)
- Noise simulation for hardware readiness
- Write up findings, document what worked and what didn't
- Identify which component benefits most from quantization

---

## 9. Honest Self-Assessment

**What excites me:**
- The patch → token → attention structure maps cleanly to quantum circuits
- Quantum kernels as attention scores is a mathematically elegant idea
- There's a real research gap — QViTs are largely unexplored

**What concerns me:**
- Current quantum hardware can't handle 49 patches × 8 qubits = 392 qubits
- The classical encoding bottleneck may eliminate any quantum advantage
- It's unclear if "quantum attention" is genuinely different from
  standard entanglement with extra steps
- Training deep quantum circuits remains an unsolved problem

**What I'd want to learn from mentors:**
- How to rigorously define when quantum attention differs from generic entanglement
- Whether structured Ansätze (attention-inspired) help with trainability
- Their perspective on which ViT component is most promising to quantize first

---

## References

1. Dosovitskiy et al., "An Image is Worth 16×16 Words" (2020). [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
2. Vaswani et al., "Attention Is All You Need" (2017). [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
3. Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier" (2020). Quantum 4, 226.
4. Di Sipio et al., "The Dawn of Quantum Natural Language Processing" (2022). ICASSP.
5. McClean et al., "Barren plateaus in quantum neural network training landscapes" (2018). Nature Comms 9, 4812.
