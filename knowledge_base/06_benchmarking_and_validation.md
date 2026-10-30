# 06. Benchmarking, Testing & Verification Plan

## 1. The Verification Hierarchy

To guarantee numerical rigor and bug-free implementation across GPU hardware, validation is organized into a four-level hierarchical testing framework:

```
                  ┌────────────────────────────────────────┐
                  │ Level 4: Cross-Package Benchmarks      │
                  │ (vs OpenMolcas, BAGEL, CPU PySCF)      │
                  ├────────────────────────────────────────┤
                  │ Level 3: Finite-Difference Verification│
                  │ (Analytical forces/NACVs vs dE/dR)     │
                  ├────────────────────────────────────────┤
                  │ Level 2: State-Averaging & Invariance  │
                  │ (Model-space rotation, orbital grads)  │
                  ├────────────────────────────────────────┤
                  │ Level 1: Unit Component Tests          │
                  │ (GPU FCI Davidson, trans-RDMs, 3-RDMs) │
                  └────────────────────────────────────────┘
```

---

## 2. Level 1: Unit Tests for GPU FCI Subcomponents

Before testing the coupled SCF or PT2 iterations, every GPU kernel must be verified against CPU PySCF references:

1. **Multi-Root Davidson Solver:**
   - Solve for 3 roots of $\text{CAS}(4,4)$ and $\text{CAS}(6,6)$ in `gpu4pyscf.fci.direct_spin1`.
   - Verify: $|E_k^{\text{GPU}} - E_k^{\text{CPU}}| < 10^{-12}\text{ Hartree}$.
2. **Transition 1-RDM (`trans_rdm1`):**
   - Compute $\gamma_{tu}^{IJ} = \langle \Psi_I | \hat{E}_{tu} | \Psi_J \rangle$ on GPU for all pairs $(I, J)$.
   - Verify: $\max_{tu} |\gamma_{tu}^{IJ, \text{GPU}} - \gamma_{tu}^{IJ, \text{CPU}}| < 10^{-12}$.
3. **Transition 2-RDM (`trans_rdm12`):**
   - Compute $\Gamma_{tuvw}^{IJ} = \langle \Psi_I | \hat{e}_{tuvw} | \Psi_J \rangle$ on GPU.
   - Verify: $\max_{tuvw} |\Gamma_{tuvw}^{IJ, \text{GPU}} - \Gamma_{tuvw}^{IJ, \text{CPU}}| < 10^{-12}$.
4. **Active 3-RDM (`make_rdm3`):**
   - Verify contractions of 3-RDM against CPU reference.

---

## 3. Level 2: Invariance & Self-Consistency Tests

1. **State-Averaged Orbital Gradient:**
   - At converged SA-CASSCF, the norm of the orbital gradient must satisfy:
     $$\|\mathbf{g}^{\text{SA}}\| = \max_{p>q} 2 |F_{pq} - F_{qp}| < 10^{-7}\text{ Hartree}$$
2. **Model-Space Invariance of XMS-CASPT2:**
   - Apply an arbitrary random unitary rotation $\mathbf{U} \in U(N_{\text{states}})$ to the converged SA-CASSCF CI vectors:
     $$\mathbf{C}' = \mathbf{C} \mathbf{U}$$
   - Run `DFXMSCASPT2` using both $\mathbf{C}$ and $\mathbf{C}'$.
   - **Verification:** The computed XMS-CASPT2 adiabatic energies, nuclear gradients, and NACVs must be **identical to within machine precision ($< 10^{-10}$)**, confirming exact model-space invariance!

---

## 4. Level 3: Finite-Difference Validation

### 4.1 Numerical vs Analytical Gradients
For each state $I$, verify the analytical nuclear gradient $\nabla_{\mathbf{R}} E_I$:
$$\left[ \nabla_{\mathbf{R}_A} E_I \right]_\alpha \approx \frac{E_I(\mathbf{R} + \delta \hat{\mathbf{e}}_{A\alpha}) - E_I(\mathbf{R} - \delta \hat{\mathbf{e}}_{A\alpha})}{2\delta}$$
- Recommended step size: $\delta = 1.0 \times 10^{-3}\text{ Bohr}$.
- Criterion: Max difference between analytical and numerical gradient $< 10^{-6}\text{ Hartree/Bohr}$.

### 4.2 Numerical vs Analytical Non-Adiabatic Couplings
The non-adiabatic coupling vector can be verified numerically via two-point finite differences of the wave functions:
$$\mathbf{d}_{IJ, A\alpha} = \langle \Psi_I | \frac{\partial \Psi_J}{\partial R_{A\alpha}} \rangle \approx \frac{\langle \Psi_I(\mathbf{R}) | \Psi_J(\mathbf{R} + \delta \hat{\mathbf{e}}_{A\alpha}) \rangle - \langle \Psi_I(\mathbf{R}) | \Psi_J(\mathbf{R} - \delta \hat{\mathbf{e}}_{A\alpha}) \rangle}{2\delta}$$
where the overlap $\langle \Psi_I(\mathbf{R}) | \Psi_J(\mathbf{R} \pm \delta) \rangle$ is evaluated including the non-orthogonality of the displaced AO basis sets:
$$S_{\mu\nu}(\mathbf{R}, \mathbf{R}') = \langle \chi_\mu(\mathbf{R}) | \chi_\nu(\mathbf{R}') \rangle$$
- Verify both with and without the CSF term (`use_etfs=True` vs `use_etfs=False`).
- Criterion: Max absolute deviation $< 10^{-5}\text{ a.u.}^{-1}$.

---

## 5. Benchmark Systems for Photochemistry

### System 1: Ethylene ($\text{C}_2\text{H}_4$) - The Minimal $\pi$-System
- **Active Space:** $\text{CAS}(2,2)$ ($\pi, \pi^*$) and $\text{CAS}(4,4)$ ($\pi, \pi^*, \sigma, \sigma^*$).
- **Basis Sets:** `6-31G*`, `cc-pVDZ`, `def2-TZVP`.
- **Target Physics:**
  - Planar Franck-Condon vertical excitation ($S_0 \to S_1$).
  - $90^\circ$ twisted-pyramidalized conical intersection.
  - Verifies orbital optimization and non-adiabatic coupling at strong coupling geometry.

### System 2: Fulvene ($\text{C}_6\text{H}_6$) - The Classic Avoided Crossing vs CoIn
- **Active Space:** $\text{CAS}(6,6)$, $6\pi$ electrons in 6 active orbitals.
- **Basis Sets:** `cc-pVDZ`, `def2-TZVP`.
- **Target Physics:**
  - Fulvene exhibits a low-lying conical intersection between $S_0$ and $S_1$ along the exocyclic C=C stretch and twisting coordinate.
  - Standard MS-CASPT2 produces an unphysical barrier and asymmetric Hamiltonian; **XMS-CASPT2 reproduces the true conical topology**.

### System 3: Retinal Protonated Schiff Base Minimal Model (PSB3)
- **Active Space:** $\text{CAS}(6,6)$, 3 conjugated double bonds:
  $$\text{CH}_2=\text{CH}-\text{CH}=\text{CH}-\text{CH}=\text{NH}_2^+$$
- **Target Physics:**
  - Ultrafast photoisomerization ($cis \to trans$) in rhodopsin.
  - Strong charge-transfer character in $S_1$ requires accurate dynamic correlation (CASPT2).
  - Trajectory simulation benchmark: 200 fs NAMD trajectory test.

---

## 6. Performance & Speedup Profiling Metrics

Target wall-clock timing and speedup benchmarks (tested on NVIDIA A100 / H100 vs 32-core Intel Xeon):

| Method | System & Basis | CPU PySCF | GPU4PySCF Target | Expected Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **SA(3)-CASSCF(6,6) Energy** | Ethylene / def2-TZVP | 4.2 s | **0.12 s** | **35x** |
| **SA(3)-CASSCF(6,6) Gradient + NAC** | Ethylene / def2-TZVP | 18.5 s | **0.45 s** | **41x** |
| **DF-XMS-CASPT2(6,6) Energy** | PSB3 / def2-TZVP (350 AOs) | 145 s | **2.8 s** | **52x** |
| **DF-XMS-CASPT2(6,6) Force + NAC** | PSB3 / def2-TZVP (350 AOs) | 980 s | **12.5 s** | **78x** |
| **1000-Step NAMD Trajectory** | PSB3 / def2-TZVP | ~11 days | **~3.5 hours** | **~75x** |
