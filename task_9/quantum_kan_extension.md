# Quantum KAN Extension — Detailed Sketch

> **Context**: This document proposes how to extend the classical SineKAN
> (Reinhardt et al., 2024) to a quantum circuit implementation.
> Written for GSoC 2026 (QMLHEP) — this is a research sketch, not a full implementation.

---

## 1. Why SineKAN → Quantum is a Natural Fit

The core insight is surprisingly clean:

| Classical SineKAN | Quantum Circuit |
|---|---|
| $\sin(k \cdot x)$ | $R_Y(k \cdot x)$ rotation gate |
| $\cos(k \cdot x)$ | $R_Z(k \cdot x)$ rotation gate |
| Learnable coefficients $a_k, b_k$ | Learnable rotation angles in variational layers |
| Summation over edges | Measurement expectation values |

A qubit's state after a rotation gate $R_Y(\theta)$ is:

$$R_Y(\theta)|0\rangle = \cos(\theta/2)|0\rangle + \sin(\theta/2)|1\rangle$$

The **measurement probabilities** are literally $\cos^2(\theta/2)$ and $\sin^2(\theta/2)$.
So quantum circuits natively compute trigonometric functions of their parameters — exactly
what SineKAN needs.

This is **not** a coincidence. It's why SineKAN is the natural classical precursor to a
quantum KAN, and why it's a stronger starting point than B-spline KAN for quantum extension.

---

## 2. Proposed Quantum KAN Architecture

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QUANTUM KAN CIRCUIT                            │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐    │
│  │   ENCODING   │   │  VARIATIONAL KAN │   │   MEASUREMENT    │    │
│  │    LAYER     │──►│     LAYERS       │──►│     LAYER        │    │
│  │              │   │  (× L repeats)   │   │                  │    │
│  └──────────────┘   └──────────────────┘   └──────────────────┘    │
│                                                                     │
│  Classical input    Quantum KAN edges      Expectation values       │
│  x → angles         φ(x) on edges         → class probabilities    │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Detailed Circuit Diagram

For a small example: 4 input features, 2 variational KAN layers, 2 output classes.

```
q₀: ─ Rx(x₀) ─ Ry(x₀) ─┤ KAN Layer 1              ├─┤ KAN Layer 2              ├─ ⟨Z₀⟩
                          │                           │ │                           │
q₁: ─ Rx(x₁) ─ Ry(x₁) ─┤  ┌────────────────────┐  ├─┤  ┌────────────────────┐  ├─ ⟨Z₁⟩
                          │  │ Edge functions:    │  │ │  │ Edge functions:    │  │
q₂: ─ Rx(x₂) ─ Ry(x₂) ─┤  │ Ry(a₁·sin(k·θ))  │  ├─┤  │ Ry(a₁·sin(k·θ))  │  ├─
                          │  │ Rz(b₁·cos(k·θ))  │  │ │  │ Rz(b₁·cos(k·θ))  │  │
q₃: ─ Rx(x₃) ─ Ry(x₃) ─┤  │ + CNOT entangling │  ├─┤  │ + CNOT entangling │  ├─
                          │  └────────────────────┘  │ │  └────────────────────┘  │
                          └───────────────────────────┘ └───────────────────────────┘

         ENCODING              VARIATIONAL KAN              MEASUREMENT
       (angle embed)        (learnable edge functions)     (expectation values)
```

### 2.3 Expanded Single KAN Layer (4 qubits)

```
q₀: ──Ry(a₁₀·sin(θ₀))──Rz(b₁₀·cos(θ₀))──●─────────────────────────Ry(a₂₀·sin(2θ₀))──
q₁: ──Ry(a₁₁·sin(θ₁))──Rz(b₁₁·cos(θ₁))──X──●──────────────────────Ry(a₂₁·sin(2θ₁))──
q₂: ──Ry(a₁₂·sin(θ₂))──Rz(b₁₂·cos(θ₂))─────X──●───────────────────Ry(a₂₂·sin(2θ₂))──
q₃: ──Ry(a₁₃·sin(θ₃))──Rz(b₁₃·cos(θ₃))────────X───────────────────Ry(a₂₃·sin(2θ₃))──
      ╰──── edge functions (k=1) ──────╯  ╰─ entangling ─╯          ╰─ edge (k=2) ───╯
```

Each `θᵢ` comes from the encoding of classical input `xᵢ`.
The `aₖᵢ` and `bₖᵢ` are **trainable parameters** — these are the quantum analogues
of the SineKAN Fourier coefficients.

---

## 3. Component Details

### 3.1 Encoding Layer

**Purpose**: Map classical data $x \in \mathbb{R}^n$ into quantum states.

**Method**: Angle encoding — each feature becomes a rotation angle.

```python
# PennyLane pseudocode
for i in range(n_qubits):
    qml.RX(x[i], wires=i)
    qml.RY(x[i], wires=i)
```

For MNIST, we'd need dimensionality reduction first (784 → n_qubits).
Options:
- PCA to reduce 784 → 8-16 features
- A classical pre-processing layer (linear projection)
- Amplitude encoding (exponentially compact, but harder to implement)

### 3.2 Variational KAN Layers (the core innovation)

**What makes this KAN-like** (not just another VQC):

In a standard VQC, parameterized gates are:
```
Ry(θ)    ←  θ is a free parameter, independent of input
```

In a **Quantum KAN layer**, the gate angles are **functions of the qubit state**:
```
Ry(aₖ · sin(k · θ_in))    ←  angle depends on input through sin/cos
Rz(bₖ · cos(k · θ_in))    ←  aₖ, bₖ are learnable, k is the frequency
```

This is the key distinction: **each gate implements a learnable edge function**
$\varphi(x) = a \sin(kx) + b \cos(kx)$, matching the classical SineKAN structure.

The entangling CNOT gates between qubits serve as the **summation nodes** in the
KAN graph — they mix information between edges, analogous to the aggregation step
$y_j = \sum_i \varphi_{ij}(x_i)$ in the classical KAN.

```python
# PennyLane pseudocode for one KAN layer
def quantum_kan_layer(weights_a, weights_b, n_qubits, K):
    # Edge functions: parameterized sin/cos rotations
    for i in range(n_qubits):
        for k in range(1, K + 1):
            qml.RY(weights_a[i, k] * np.sin(k * qml.state(i)), wires=i)  # conceptual
            qml.RZ(weights_b[i, k] * np.cos(k * qml.state(i)), wires=i)

    # Entangling (summation analogue)
    for i in range(n_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
```

> **Important caveat**: In practice, we can't directly read qubit state mid-circuit.
> Instead, `θ_in` would come from the encoding layer's angles or from a
> **re-uploading** scheme where classical data is re-injected at each layer.

### 3.3 Data Re-uploading (practical approach)

Since we can't "read" intermediate qubit states, the practical approach is
**data re-uploading** (Pérez-Salinas et al., 2020):

```
Layer 1: Encode x → Apply variational gates f(aₖ, x)
Layer 2: Re-encode x → Apply variational gates g(bₖ, x)
...
```

Each layer re-injects the classical data, and the variational parameters
modify how that data interacts with the quantum state. This naturally
implements the SineKAN structure:

$$R_Y\big(a_k \cdot \sin(k \cdot x_i)\big) \cdot R_Z\big(b_k \cdot \cos(k \cdot x_i)\big)$$

The gate angles are explicit trigonometric functions of the classical input,
weighted by learnable parameters — exactly the SineKAN edge function.

### 3.4 Measurement Layer

```python
# Measure expectation values of Pauli-Z on designated output qubits
expectations = [qml.expval(qml.PauliZ(i)) for i in output_qubits]
# These ∈ [-1, 1], map to class logits via softmax
```

For 10-class MNIST, we'd need 10 expectation values. Options:
- Use 10 qubits, measure each → expensive
- Use fewer qubits, multiple Pauli observables (Z, X, Y) per qubit
- Post-process with a small classical layer (hybrid approach)

---

## 4. What Makes This "KAN-like" vs a Standard VQC?

This is the critical question. Here's the honest comparison:

| Feature | Standard VQC | Quantum KAN (proposed) |
|---------|-------------|----------------------|
| Gate angles | Free parameters $\theta$ | Functions of input: $a \sin(kx)$ |
| Input dependence | Only in encoding layer | Re-uploaded at every layer |
| Edge concept | No — gates are independent | Yes — each gate = learnable edge function |
| Frequency structure | No explicit frequencies | Explicit $k = 1, 2, \ldots, K$ frequencies |
| KA theorem connection | None | Each gate implements $\varphi_{ij}(x_i)$ |

**The genuinely KAN-like aspects**:
1. Gate angles are **explicit trigonometric functions of the data** (not free parameters)
2. There's a clear **edge → node** structure matching KAN's graph topology
3. The **frequency decomposition** ($k = 1, \ldots, K$) is preserved in the gate structure
4. Entanglement gates serve as **aggregation nodes** (summation in classical KAN)

**What's still different**:
- Quantum circuits operate in exponentially large Hilbert space (not just $\mathbb{R}^n$)
- Entanglement creates correlations that have no classical KAN analogue
- Measurement collapse introduces probabilistic behavior

---

## 5. Practical Architecture for MNIST

Given current hardware constraints (noisy qubits, limited connectivity):

```
┌─────────────────────────────────────────────────────────┐
│                Hybrid Quantum-Classical KAN             │
│                                                         │
│  ┌───────────┐    ┌──────────────┐    ┌──────────────┐ │
│  │ Classical  │    │   Quantum    │    │  Classical   │ │
│  │ Encoder   │───►│  KAN Core    │───►│  Decoder     │ │
│  │           │    │              │    │              │ │
│  │ 784 → 8  │    │  8 qubits    │    │  8 → 10     │ │
│  │ (Linear)  │    │  L=3 layers  │    │  (Linear)    │ │
│  └───────────┘    └──────────────┘    └──────────────┘ │
│                                                         │
│  Trainable          Trainable           Trainable       │
│  classically        hybrid              classically     │
└─────────────────────────────────────────────────────────┘
```

**Why hybrid**:
- 784 qubits is impossible on current hardware
- Classical encoder compresses to manageable qubit count
- Quantum core provides the KAN-specific learnable edge functions
- Classical decoder maps qubit measurements to 10 class logits

---

## 6. Open Research Questions (GSoC Investigation)

### 6.1 Expressibility
- Does the quantum KAN's Hilbert space structure give it an advantage in
  expressibility over the classical SineKAN?
- Can we prove or empirically show that fewer frequencies $K$ are needed in the
  quantum version compared to classical?

### 6.2 Trainability
- Do quantum KAN layers suffer from **barren plateaus** (vanishing gradients in
  high-dimensional parameter spaces)?
- Does the structured sine/cosine parameterization help avoid barren plateaus
  compared to arbitrary VQC rotations? (Hypothesis: yes, because the Fourier
  structure constrains the loss landscape)

### 6.3 Entanglement's Role
- In classical KAN, node aggregation is simple summation. In quantum KAN,
  entanglement gates create non-trivial correlations.
- **Question**: Does the entanglement structure need to mirror the KAN graph
  topology, or can arbitrary entanglement patterns work?
- What's the minimum entanglement needed for quantum advantage?

### 6.4 Scalability
- How does performance scale with number of qubits, layers, and frequencies?
- What's the trade-off between quantum circuit depth and accuracy?
- Can techniques like **circuit cutting** or **tensor network methods** help scale?

### 6.5 Classical Simulation Barrier
- At what qubit count does the quantum KAN become classically intractable?
- Can we identify a regime where quantum KAN provably outperforms classical KAN?

### 6.6 Noise Robustness
- How do gate errors affect the learned sine/cosine coefficients?
- Are SineKAN-style parameterizations more or less noise-robust than standard VQC?

---

## 7. What I'd Investigate in GSoC

**Phase 1 (Weeks 1-4): Foundation**
- Implement the quantum KAN layer in PennyLane
- Start with a toy problem (e.g., function approximation, not full MNIST)
- Compare expressibility vs classical SineKAN on 2-3 qubit systems
- Validate that the KAN structure is genuinely preserved in the quantum circuit

**Phase 2 (Weeks 5-8): MNIST Pilot**
- Build the full hybrid pipeline: classical encoder → quantum KAN → classical decoder
- Test on MNIST with 4-8 qubits (simulator)
- Benchmark against: (a) classical SineKAN, (b) standard VQC, (c) random baseline
- Study gradient landscapes (barren plateau analysis)

**Phase 3 (Weeks 9-12): Analysis & Writing**
- Investigate the open questions from Section 6
- Noise simulation to assess hardware readiness
- Write up findings, contribute to the QMLHEP codebase
- Document what worked, what didn't, and future directions

---

## 8. Honest Assessment

**What excites me**:
- The sin/cos → rotation gate mapping is elegant and mathematically natural
- SineKAN's Fourier structure could genuinely constrain quantum circuits in useful ways
- There's a real research gap here — quantum KANs are largely unexplored

**What concerns me**:
- Current quantum hardware is noisy with limited qubits — we're stuck in simulator territory
- The classical encoding bottleneck (784 → 8 qubits) may wash out any quantum advantage
- It's unclear if the "KAN-ness" survives the quantum translation, or if we end up
  with just another VQC with sine-parameterized angles
- Barren plateaus could make training intractable for deeper circuits

**What I'd want to learn from my mentors**:
- How to rigorously define "KAN-like" in the quantum context
- Their insights from the SineKAN paper on which properties are most important to preserve
- Whether the Fourier structure helps with trainability (their intuition would be invaluable)

---

## References

1. Liu et al., "KAN: Kolmogorov-Arnold Networks" (2024). [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
2. Reinhardt et al., "SineKAN: Kolmogorov-Arnold Networks Using Sinusoidal Activation Functions" (2024). [arXiv:2407.04149](https://arxiv.org/abs/2407.04149)
3. Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier" (2020). Quantum 4, 226.
4. McClean et al., "Barren plateaus in quantum neural network training landscapes" (2018). Nature Communications 9, 4812.
