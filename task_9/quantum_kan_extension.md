# Quantum KAN Extension 

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


## 2. Proposed Quantum KAN Architecture

###  High-Level Design

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


### 1. Input Encoding

Each input feature is mapped to a rotation angle.

xᵢ → θᵢ = k · xᵢ

Encoding is performed using parameterized rotation gates.

Each feature may be assigned its own qubit register to preserve the per-edge functional structure of KANs.

---

### 2. Quantum Edge Functions

Each classical KAN edge function becomes a data re-uploading quantum subcircuit.

Structure of a single edge circuit:

1. Encode input x
2. Apply trainable rotation gates
3. Re-upload x again
4. Repeat for L layers



The resulting expectation value:

⟨Z⟩ = Σ (a_k sin(kx) + b_k cos(kx))

This reproduces the Fourier edge function used in SineKAN.

---

### 3. Summation Nodes

KAN nodes sum outputs from incoming edges.

In the quantum architecture this corresponds to:

• measuring expectation values of qubits  
• classically summing measurement outputs  
• optionally introducing entanglement layers to represent interactions between features

Entanglement is inserted only at summation steps to preserve the structural identity of KANs.

---

### 4. Layered QKAN Structure

A full QKAN layer contains:

1. Feature encoding layer  
2. Edge-function circuits (data re-uploading blocks)  
3. Optional entanglement for feature aggregation  
4. Measurement to produce node outputs

These outputs are then passed as classical inputs to the next layer.


## Key Architectural Properties

**Fourier Basis Compatibility**  
Quantum rotations naturally generate the sine/cosine basis required by SineKAN.

**Parameter Efficiency**  
Fourier coefficients are encoded in a small number of trainable rotation parameters.

**Structural Interpretability**  
Each quantum subcircuit corresponds directly to a KAN edge function, preserving interpretability.

**Hardware Feasibility**  
The architecture relies on shallow circuits composed primarily of rotation gates and limited entanglement, making it compatible with near-term quantum hardware.


## Expected Research Contributions

**Formalizing the SineKAN–Quantum Correspondence**  
   Demonstrating that data re-uploading circuits compute truncated Fourier series equivalent to SineKAN edge functions.

**Designing a Native Quantum KAN Architecture**  
   Implementing a QKAN in PennyLane/Qiskit that preserves the structural principles of classical KANs.

**Benchmarking on HEP Tasks**  
   Evaluating QKAN performance against classical SineKAN models for signal–background classification in LHC datasets.

---
