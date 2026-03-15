# Quantum KAN Extension — Detailed Sketch

**Context**: 
This document proposes how to extend the classical SineKAN (Reinhardt et al., 2024) to a quantum circuit implementation, specifically targeted at High Energy Physics (HEP) applications at the CERN LHC.

Written for GSoC 2026 (QMLHEP) 

---

## 1. SineKAN → Quantum is a Natural Fit


| Classical SineKAN | Quantum Circuit |
|---|---|
| $\sin(k \cdot x)$ | $R_Y(k \cdot x)$ rotation gate |
| $\cos(k \cdot x)$ | $R_Z(k \cdot x)$ rotation gate |
| Learnable coefficients $a_k, b_k$ | Learnable rotation angles in variational layers |
| Summation over edges | Measurement expectation values |


Quantum circuits natively compute trigonometric functions of their parameters.     
Hence , SineKAN is a stronger starting point than B-spline KAN for quantum extension.

---

## 2. Proposed Quantum KAN Architecture

### 2.1 High-Level Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                     QUANTUM KAN CIRCUIT                             |
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────┐     │
│  │   ENCODING   │   │  VARIATIONAL KAN │   │   MEASUREMENT    │     │
│  │    LAYER     │──►│     LAYERS       │──►│     LAYER        │     │
│  │              │   │  (× L repeats)   │   │                  │     │
│  └──────────────┘   └──────────────────┘   └──────────────────┘     │
│                                                                     │
│  Classical input    Quantum KAN edges      Expectation values       │
│  x → angles         φ(x) on edges         → class probabilities     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 Quantum Circuit Structure

Instead of explicitly drawing the circuit, the architecture can be understood as a sequence of operations on $N$ qubits:

1. **Input Encoding**: Each classical feature $x_i$ is mapped to a rotation angle on qubit $i$ using $R_X$ and $R_Y$ gates.
2. **Edge Functions**: The trainable parameters $a_k$ and $b_k$ are applied as $R_Y$ and $R_Z$ rotations on the data-encoded states. A single edge with frequency $k$ looks like $R_Y(a_k \sin(\theta)) R_Z(b_k \cos(\theta))$.
3. **Entanglement (Optional)**: CNOT gates can optionally entangle adjacent qubits to mix information across different input features.
4. **Repetition**: Steps 1 and 2 are repeated for each frequency $k=1, \ldots, K$ to build up the full SineKAN Fourier series.

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

For HEP data (e.g., jet kinematics), we'd need dimensionality reduction first (e.g., $O(100)$ features $\to$ n_qubits).
Options:
- Classical Autoencoder or PCA to reduce feature space $\to$ 8-16 features
- Amplitude encoding (exponentially compact, but harder to implement on near-term hardware)

### 3.2 Variational KAN Layers (the core innovation)

**What makes this KAN-like** (not just another VQC):

In a standard VQC, parameterized gates are:
```
Ry(θ)    ←  θ is a free parameter, independent of input
```

In a **Quantum KAN layer**, the gate angles are **functions of the data**:
```
Ry(aₖ · sin(k · x_in))    ←  angle depends on input through sin/cos
Rz(bₖ · cos(k · x_in))    ←  aₖ, bₖ are learnable, k is the frequency
```

This is the key distinction: **each gate implements a learnable edge function**
$\varphi(x) = a \sin(kx) + b \cos(kx)$, matching the classical SineKAN structure.

**Node Aggregation (The Safe Route):**
To mimic the aggregation step $y_j = \sum_i \varphi_{ij}(x_i)$, the most robust near-term approach is **classical summation**: calculate the expectation value of each quantum edge independently, and sum them linearly in PyTorch. 

**Node Aggregation (The Advanced Route):**
Alternatively, entangling CNOT gates between qubits can serve as quantum summation nodes, mixing information between edges before measurement. This is more "quantum-native" but structurally harder to train.

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

## 5. Practical Architecture for HEP Applications

Given current hardware constraints (noisy qubits, limited connectivity), processing raw LHC detector data with a pure quantum KAN is impossible. 

We propose a **Hybrid Quantum-Classical architecture** for HEP event classification:
1. **Classical Encoder**: A standard neural network or PCA reduces the high-dimensional detector signals (e.g., track kinematics, tower energies) to a small number of latent features (e.g., 8-12 features).
2. **Quantum KAN Core**: An 8-to-12 qubit quantum KAN circuit processes these compressed features, applying the learnable SineKAN edge functions over $L=3$ layers to capture complex, non-linear correlations between the physical features.
3. **Classical Decoder**: The expectation values from the qubits are fed into a final classical linear-layer classifier to produce the event classification probabilities.

This isolates the quantum component to where it might genuinely provide an inductive bias (the KAN edge functions) while keeping the overall model trainable on real CERN datasets.


## Assessment

### Interesting Finds :
- The sin/cos to  rotation gate mapping is elegant and mathematically natural

- SineKAN's Fourier structure could constrain quantum circuits in useful ways

### Concerns :
- Current quantum hardware is noisy with limited qubits .

- It's unclear if KAN survives the quantum translation with efficiency .
- Barren plateaus could make training intractable for deeper circuits

### Learning Opportunity :
- How to rigorously define SineKAN in the quantum context.

- Your insights from the SineKAN paper on which properties are most important to preserve
- Whether the Fourier structure helps with trainability


## References

1. Liu et al., "KAN: Kolmogorov-Arnold Networks" . [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
2. Reinhardt et al., "SineKAN: Kolmogorov-Arnold Networks Using Sinusoidal Activation Functions" . [arXiv:2407.04149](https://arxiv.org/abs/2407.04149)
3. Pérez-Salinas et al., "Data re-uploading for a universal quantum classifier". Quantum 4, 226.
4. McClean et al., "Barren plateaus in quantum neural network training landscapes" . Nature Communications 9, 4812.
