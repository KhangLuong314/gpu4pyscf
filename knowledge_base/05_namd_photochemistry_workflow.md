# 05. Photochemistry & Non-Adiabatic Molecular Dynamics (NAMD)

## 1. Photochemical Reaction Pathways & Conical Intersections

Upon photon absorption, a molecule undergoes vertical excitation to an excited electronic state surface ($S_1, S_2, \dots$), initiating nuclear motion away from the Franck-Condon region:

```
Energy
  ▲                   / (S1)
  │   Franck-Condon  /
  │      ┌───►      X Conical Intersection (CoIn)
  │     /│         / \  (Non-radiative funnel: τ < 100 fs)
  │    / │        /   \
  │   /  │       /     ▼
  │  /   ▼      /   Photoproduct
  │ /   (S0)   /
  └────────────────────────► Nuclear Coordinate
```

At **Conical Intersections (CoIns)**, two adiabatic potential energy surfaces become strictly degenerate:
$$\Delta E_{IJ}(\mathbf{R}_{\text{CoIn}}) = E_J(\mathbf{R}_{\text{CoIn}}) - E_I(\mathbf{R}_{\text{CoIn}}) = 0$$
The Born-Oppenheimer approximation breaks down completely, and non-adiabatic transitions occur on the sub-100 femtosecond timescale.

### The 2D Branching Space
Near an $N$-dimensional conical intersection, the electronic degeneracy is lifted linearly only along a **two-dimensional branching space**:
1. **Tuning Vector ($\mathbf{g}_{IJ}$):** Energy difference gradient vector:
   $$\mathbf{g}_{IJ} = \frac{1}{2} \nabla_{\mathbf{R}} (E_J - E_I)$$
2. **Coupling Vector ($\mathbf{h}_{IJ}$):** Off-diagonal Hamiltonian gradient (scaled NACV):
   $$\mathbf{h}_{IJ} = \langle \Psi_I | \nabla_{\mathbf{R}} \hat{H} | \Psi_J \rangle = (E_J - E_I) \mathbf{d}_{IJ}$$

In all other $(3N_{\text{atom}} - 8)$ nuclear degrees of freedom (the **intersection seam**), the electronic states remain degenerate to first order.

---

## 2. Tully's Fewest Switches Surface Hopping (FSSH)

FSSH (Tully, *J. Chem. Phys.* 1990, 93, 1061) is the standard method for non-adiabatic dynamics. Classical nuclei evolve on a single "active" adiabatic potential energy surface $k$, while electronic amplitudes evolve quantum-mechanically along the trajectory $\mathbf{R}(t)$.

### 2.1 The Electronic Equation of Motion
The electronic wave function is expanded in the instantaneous adiabatic basis:
$$|\Psi(t)\rangle = \sum_j c_j(t) |\Psi_j(\mathbf{R}(t))\rangle$$
The time-dependent Schrödinger equation yields:
$$i\hbar \dot{c}_j(t) = E_j(\mathbf{R}(t)) c_j(t) - i\hbar \sum_l \left( \dot{\mathbf{R}}(t) \cdot \mathbf{d}_{jl}(\mathbf{R}(t)) \right) c_l(t)$$
where $\sigma_{jl} = \dot{\mathbf{R}} \cdot \mathbf{d}_{jl}$ is the scalar **time-derivative non-adiabatic coupling**.

In terms of the electronic density matrix $\rho_{jk} = c_j c_k^*$:
$$\dot{\rho}_{jk} = -\frac{i}{\hbar} (E_j - E_k) \rho_{jk} - \sum_l \left( \dot{\mathbf{R}} \cdot \mathbf{d}_{jl} \rho_{lk} - \rho_{jl} \dot{\mathbf{R}} \cdot \mathbf{d}_{lk} \right)$$

### 2.2 Hopping Probability
In a time step $\Delta t$, the probability of switching from the currently occupied state $k$ to state $j$ ($j \neq k$) is:
$$g_{k \to j} = \max\left( 0, \frac{2 \Delta t}{\rho_{kk}} \text{Re}\left( \rho_{jk}^* \left[ \dot{\mathbf{R}} \cdot \mathbf{d}_{jk} \right] \right) \right)$$
A random number $\zeta \in [0, 1)$ determines if a hop occurs:
$$\sum_{l=1}^{j-1} g_{k \to l} \le \zeta < \sum_{l=1}^j g_{k \to l}$$

---

## 3. Critical Semiclassical Nuances in NAMD

### 3.1 Electronic Wave Function Phase Tracking
At each time step $\mathbf{R}(t)$, the quantum chemistry solver computes adiabatic eigenfunctions $|\Psi_j\rangle$. Because eigenvectors are determined up to an arbitrary global phase ($\pm 1$), random sign changes can occur between consecutive time steps:
$$|\Psi_j(\mathbf{R}(t + \Delta t))\rangle \approx - |\Psi_j(\mathbf{R}(t))\rangle$$
If uncorrected, this sign flip inverts the sign of the NACV $\mathbf{d}_{jk}$, causing severe numerical errors in $\dot{\rho}$ and false hops!

**Algorithm (Phase Matching via State Overlap):**
1. Evaluate the overlap matrix between states at step $t$ and step $t + \Delta t$:
   $$S_{ij}(t, t + \Delta t) = \langle \Psi_i(t) | \Psi_j(t + \Delta t) \rangle$$
2. Inspect the diagonal element $S_{ii}$:
   $$\text{sign}_i = \text{sgn}(S_{ii}(t, t + \Delta t))$$
3. If $\text{sign}_i < 0$:
   $$|\Psi_i(t + \Delta t)\rangle \longleftarrow - |\Psi_i(t + \Delta t)\rangle$$
   $$\mathbf{d}_{ij}(t + \Delta t) \longleftarrow - \mathbf{d}_{ij}(t + \Delta t) \quad \forall j$$
This logic is implemented in `gpu4pyscf.md.fssh` and must be preserved when evaluating SA-CASSCF and XMS-CASPT2 states.

---

### 3.2 Direction of Velocity Adjustment
When a switch occurs from state $k$ to state $j$, nuclear velocities must be adjusted to conserve total energy:
$$E_{\text{tot}} = \frac{1}{2} \sum_A M_A |\mathbf{v}_A|^2 + E_k = \frac{1}{2} \sum_A M_A |\mathbf{v}_A'|^2 + E_j$$

**Which direction in nuclear configuration space should receive the momentum kick?**
- Rigorous semiclassical theory (Pechukas 1969, Herman 1984) demonstrates that momentum must be rescaled along the **Non-Adiabatic Coupling Vector $\mathbf{d}_{kj}$**:
  $$\mathbf{P}_A' = \mathbf{P}_A - \gamma \frac{\mathbf{d}_{kj, A}}{M_A}$$
- Substituting into energy conservation yields a quadratic equation for the scalar $\gamma$:
  $$\frac{1}{2} \sum_A \frac{|\mathbf{d}_{kj, A}|^2}{M_A} \gamma^2 - \left( \sum_A \mathbf{v}_A \cdot \mathbf{d}_{kj, A} \right) \gamma + (E_j - E_k) = 0$$
- **Frustrated Hops:** If the discriminant is negative ($b^2 - 4ac < 0$), an upward hop ($E_j > E_k$) is forbidden because the nuclear kinetic energy along the coupling vector is insufficient. The hop is rejected, and the trajectory remains on state $k$.

---

### 3.3 Decoherence Corrections
In FSSH, the electronic wave packet does not split physically; classical trajectories carry coherent electronic amplitudes indefinitely. The **Energy-based Decoherence Correction (EDC)** (Granucci & Persico, *J. Chem. Phys.* 2007, 126, 134114) damps off-diagonal density matrix elements towards zero:
$$\rho_{jk}(t + \Delta t) = \rho_{jk}(t) \exp\left( - \frac{\Delta t}{\tau_{jk}} \right)$$
with decoherence time:
$$\tau_{jk} = \frac{\hbar}{|E_j - E_k|} \left( 1 + \frac{C}{E_{\text{kin}}} \right)$$
This correction is natively supported in `gpu4pyscf.md.fssh`.

---

## 4. Minimum Energy Conical Intersection (MECI) Optimization

Locating the lowest-energy point on the conical intersection seam is essential for identifying photochemical decay channels.

At a MECI, we minimize the higher state energy $E_J$ subject to the degeneracy constraint $E_J - E_I = 0$:
$$\min_{\mathbf{R}} E_J(\mathbf{R}) \quad \text{subject to} \quad E_J(\mathbf{R}) - E_I(\mathbf{R}) = 0$$

### The Bearpark-Robb-Schlegel Projected Gradient Method
The gradient of state $J$ is projected onto the $(3N_{\text{atom}} - 8)$-dimensional intersection seam:
$$\mathbf{P}_{\text{seam}} = \mathbf{I} - \hat{\mathbf{g}} \hat{\mathbf{g}}^T - \hat{\mathbf{h}} \hat{\mathbf{h}}^T$$
where $\hat{\mathbf{g}}$ and $\hat{\mathbf{h}}$ are normalized orthogonalized vectors spanning the branching space.

The effective optimization gradient is:
$$\mathbf{F}_{\text{MECI}} = \mathbf{P}_{\text{seam}} \nabla_{\mathbf{R}} E_J + 2 (E_J - E_I) \hat{\mathbf{g}}$$
The first term minimizes energy along the seam, while the second term drives the system towards exact electronic degeneracy.

**GPU Impact:** MECI optimizations typically require $100 - 300$ geometry iterations. With GPU-accelerated SA-CASSCF or XMS-CASPT2, each gradient and NAC evaluation drops from minutes to under a second, turning hours-long searches into single-minute optimizations!


---

## 5. Key Literature References & Citations

1. **Foundations of Fewest Switches Surface Hopping (FSSH):**
   - J. C. Tully, *"Molecular dynamics with electronic transitions"*, *J. Chem. Phys.* **93**, 1061–1071 (1990). DOI: [`10.1063/1.459170`](https://doi.org/10.1063/1.459170). [BibTeX: `Tully1990`](references.bib#L27)
2. **Momentum Adjustment along NACVs & Electron Translation Factors:**
   - S. Fatehi and J. E. Subotnik, *"Derivative Couplings with Built-In Electron-Translation Factors: Application to Benzene"*, *J. Phys. Chem. Lett.* **3**, 2039–2043 (2012). DOI: [`10.1021/jz3006173`](https://doi.org/10.1021/jz3006173). [BibTeX: `Fatehi2012`](references.bib#L41)
3. **Decoherence Corrections in Surface Hopping:**
   - G. Granucci and M. Persico, *"Critical appraisal of the fewest switches algorithm for surface hopping"*, *J. Chem. Phys.* **126**, 134114 (2007). DOI: [`10.1063/1.2715585`](https://doi.org/10.1063/1.2715585). [BibTeX: `Granucci2007`](references.bib#L29)
4. **Minimum Energy Conical Intersection (MECI) Optimization:**
   - M. J. Bearpark, M. A. Robb, and H. B. Schlegel, *"A direct method for the location of the lowest energy point on a potential surface crossing"*, *Chem. Phys. Lett.* **223**, 269–274 (1994). DOI: [`10.1016/0009-2614(94)00433-1`](https://doi.org/10.1016/0009-2614(94)00433-1). [BibTeX: `Bearpark1994`](references.bib#L31)
5. **GPU-Accelerated Multireference Photodynamics Trajectories:**
   - J. W. Snyder Jr., B. F. E. Curchod, and T. J. Martínez, *"GPU-Accelerated State-Averaged Complete Active Space Self-Consistent Field Interfaced with Ab Initio Multiple Spawning Unravels the Photodynamics of Provitamin D3"*, *J. Phys. Chem. Lett.* **7**, 2444–2449 (2016). DOI: [`10.1021/acs.jpclett.6b00970`](https://doi.org/10.1021/acs.jpclett.6b00970). [BibTeX: `Snyder2016`](references.bib#L5)
