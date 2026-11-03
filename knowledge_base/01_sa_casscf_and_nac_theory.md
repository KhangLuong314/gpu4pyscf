# 01. Theory: SA-CASSCF Gradients & Non-Adiabatic Couplings (NACVs)

## 1. Introduction and Foundations

In computational photochemistry and non-adiabatic molecular dynamics (NAMD), electronically excited states frequently interact through conical intersections (CoIns) and avoided crossings. Single-reference methods (such as standard DFT or MP2) break down near these regions due to strong static correlation, multireference character, and electronic degeneracies. 

The **State-Averaged Complete Active Space Self-Consistent Field (SA-CASSCF)** method provides a balanced zeroth-order description of multiple electronic states by optimizing a single set of molecular orbitals (MOs) for an ensemble average of states.

---

## 2. The SA-CASSCF Energy Functional

Let $\{|\Psi_I\rangle\}_{I=1}^{N_{\text{states}}}$ be a set of orthogonal electronic states expanded in an active space configuration interaction (CI) basis:
$$|\Psi_I\rangle = \sum_{\mu \in \text{FCI}} C_{\mu I} |\Phi_\mu\rangle$$
where $|\Phi_\mu\rangle$ are Configuration State Functions (CSFs) or Slater determinants constructed from orthonormal molecular orbitals:
$$\phi_p(\mathbf{r}) = \sum_{\mu=1}^{N_{\text{AO}}} C_{\mu p} \chi_\mu(\mathbf{r})$$

The orbital space is partitioned into three canonical subspaces:
1. **Core (Inactive) Orbitals ($i, j, k, \dots$):** Doubly occupied in all configurations ($n_{\text{core}}$).
2. **Active Orbitals ($t, u, v, w, x, y, \dots$):** Variable occupation ($0, 1, 2$) across the active electron count ($n_{\text{cas}}, n_{\text{elec}}$).
3. **Virtual (Secondary) Orbitals ($a, b, c, \dots$):** Unoccupied in all configurations.

The state-averaged energy functional is defined as a weighted sum of state energies:
$$E_{\text{SA}} = \sum_{I=1}^{N_{\text{states}}} w_I E_I = \sum_{I=1}^{N_{\text{states}}} w_I \langle \Psi_I | \hat{H} | \Psi_I \rangle$$
subject to the constraints:
$$\sum_{I=1}^{N_{\text{states}}} w_I = 1, \quad w_I > 0, \quad \langle \Psi_I | \Psi_J \rangle = \delta_{IJ}$$

### 2.1 State-Averaged Density Matrices

The total state-averaged energy can be written compactly using the state-averaged 1- and 2-particle reduced density matrices (1-RDM and 2-RDM):
$$E_{\text{SA}} = \sum_{pq} h_{pq} D_{pq}^{\text{SA}} + \frac{1}{2} \sum_{pqrs} (pq|rs) d_{pqrs}^{\text{SA}} + V_{\text{nuc}}$$
where:
$$D_{pq}^{\text{SA}} = \sum_{I=1}^{N_{\text{states}}} w_I D_{pq}^I, \quad D_{pq}^I = \langle \Psi_I | \hat{E}_{pq} | \Psi_I \rangle$$
$$d_{pqrs}^{\text{SA}} = \sum_{I=1}^{N_{\text{states}}} w_I d_{pqrs}^I, \quad d_{pqrs}^I = \langle \Psi_I | \hat{e}_{pqrs} | \Psi_I \rangle$$
and the spin-traced excitation operators are:
$$\hat{E}_{pq} = a_{p\alpha}^\dagger a_{q\alpha} + a_{p\beta}^\dagger a_{q\beta}$$
$$\hat{e}_{pqrs} = \hat{E}_{pq}\hat{E}_{rs} - \delta_{qr}\hat{E}_{ps} = \sum_{\sigma, \tau} a_{p\sigma}^\dagger a_{r\tau}^\dagger a_{s\tau} a_{q\sigma}$$

Because core orbitals are doubly occupied, their density matrices are scalar:
$$D_{ij}^{\text{core}} = 2\delta_{ij}, \quad d_{ijkl}^{\text{core}} = 4\delta_{ij}\delta_{kl} - 2\delta_{il}\delta_{jk}$$
Only active-space elements $D_{tu}^I$ and $d_{tuvw}^I$ depend on the CI coefficients $C_{\mu I}$.

---

## 3. Stationarity Conditions & Orbital Optimization

Orbital rotations are parameterized via an anti-Hermitian operator $\hat{\kappa}$:
$$\hat{U} = \exp(-\hat{\kappa}), \quad \hat{\kappa} = \sum_{p > q} \kappa_{pq} (\hat{E}_{pq} - \hat{E}_{qp})$$
Orbital rotations between redundant pairs (core-core, active-active, virtual-virtual) do not change the total state-averaged energy, leaving non-redundant rotations:
$$\{pq\} \in \{\text{core-active}, \text{core-virtual}, \text{active-virtual}\}$$

The first-order orbital gradient with respect to $\kappa_{pq}$ is:
$$g_{pq} = \left. \frac{\partial E_{\text{SA}}}{\partial \kappa_{pq}} \right|_{\kappa=0} = 2 (F_{pq} - F_{qp})$$
where the Generalized Fock Matrix $F_{pq}$ is:
$$F_{pq} = \sum_{r} h_{pr} D_{qr}^{\text{SA}} + \sum_{rst} (pr|st) d_{qrst}^{\text{SA}}$$

At convergence:
$$g_{pq} = 0 \quad \forall p > q \quad \Longleftrightarrow \quad F_{pq} = F_{qp}$$
$$(\mathbf{H}_{\text{act}} - E_I \mathbf{I}) \mathbf{C}_I = 0 \quad \forall I \in \{1, \dots, N_{\text{states}}\}$$

---

## 4. Coupled-Perturbed MCSCF (CP-MCSCF) for State-Specific Gradients

### 4.1 The Need for Orbital & CI Response
In single-state CASSCF, the energy is fully stationary with respect to both orbital rotations ($\partial E / \partial \kappa = 0$) and CI coefficients ($\partial E / \partial C = 0$). Hence, by the Hellmann-Feynman theorem, no orbital or CI response derivatives are required for the gradient.

**However, in SA-CASSCF, individual state energies $E_I$ are NOT stationary!**
$$g_{pq}^I = \left. \frac{\partial E_I}{\partial \kappa_{pq}} \right|_{\kappa=0} = 2(F_{pq}^I - F_{qp}^I) \neq 0$$
When calculating the nuclear gradient of state $I$ ($\nabla_{\mathbf{R}} E_I$), one must account for how the state-averaged orbitals and state CI vectors respond to nuclear perturbations $\mathbf{R}$:
$$\frac{d E_I}{d \mathbf{R}} = \left( \frac{\partial E_I}{\partial \mathbf{R}} \right)_{\text{explicit}} + \sum_{p>q} \frac{\partial E_I}{\partial \kappa_{pq}} \frac{\partial \kappa_{pq}}{\partial \mathbf{R}} + \sum_{\mu} \frac{\partial E_I}{\partial C_{\mu I}} \frac{\partial C_{\mu I}}{\partial \mathbf{R}}$$

### 4.2 The Z-Vector (Lagrangian) Formulation
Solving the coupled-perturbed equations for every nuclear coordinate ($3N_{\text{atoms}}$ times) is computationally prohibitive. Instead, Handy and Schaefer's **Z-vector method** solves a single set of perturbation-independent linear equations.

We construct the variational Lagrangian:
$$\mathcal{L}_I = E_I + \sum_{p>q} z_{pq} g_{pq}^{\text{SA}} + \sum_{J=1}^{N_{\text{states}}} \sum_{\mu} z_{\mu J}^{\text{ci}} \left[ \sum_{\nu} H_{\mu\nu} C_{\nu J} - E_J C_{\mu J} \right]$$
Setting $\partial \mathcal{L}_I / \partial \kappa_{pq} = 0$ and $\partial \mathcal{L}_I / \partial C_{\mu J} = 0$ yields the CP-MCSCF linear system:
$$\begin{pmatrix} \mathbf{A}^{\text{orb-orb}} & \mathbf{A}^{\text{orb-ci}} \\ \mathbf{A}^{\text{ci-orb}} & \mathbf{A}^{\text{ci-ci}} \end{pmatrix} \begin{pmatrix} \mathbf{z}^{\text{orb}} \\ \mathbf{z}^{\text{ci}} \end{pmatrix} = - \begin{pmatrix} \mathbf{g}^I \\ \mathbf{0} \end{pmatrix}$$
where $\mathbf{A}$ is the SA-CASSCF electronic Hessian:
$$\mathbf{A}_{pq, rs}^{\text{orb-orb}} = \frac{\partial^2 E_{\text{SA}}}{\partial \kappa_{pq} \partial \kappa_{rs}}$$

Once the Lagrange multipliers $\mathbf{z}^{\text{orb}}$ and $\mathbf{z}^{\text{ci}}$ are determined, the gradient is evaluated without any nuclear derivatives of wave function parameters:
$$\nabla_{\mathbf{R}} E_I = \sum_{\mu\nu} P_{\mu\nu}^{\text{eff}} \nabla_{\mathbf{R}} h_{\mu\nu} + \sum_{\mu\nu\rho\sigma} \Gamma_{\mu\nu\rho\sigma}^{\text{eff}} \nabla_{\mathbf{R}} (\mu\nu|\rho\sigma) + \sum_{\mu\nu} W_{\mu\nu}^{\text{eff}} \nabla_{\mathbf{R}} S_{\mu\nu}$$
where $P^{\text{eff}}$, $\Gamma^{\text{eff}}$, and $W^{\text{eff}}$ are the relaxed 1-particle, 2-particle, and energy-weighted density matrices.

---

## 5. Non-Adiabatic Coupling Vectors (NACVs)

### 5.1 Definition & Off-Diagonal Hellmann-Feynman Relation
The first-order Non-Adiabatic Coupling Vector (NACV, or derivative coupling) between adiabatic states $I$ and $J$ ($I \neq J$) is:
$$\mathbf{d}_{IJ}(\mathbf{R}) = \langle \Psi_I | \nabla_{\mathbf{R}} \Psi_J \rangle = - \langle \nabla_{\mathbf{R}} \Psi_I | \Psi_J \rangle$$

Differentiating the Schrödinger equation $\hat{H}|\Psi_J\rangle = E_J|\Psi_J\rangle$ and projecting with $\langle \Psi_I|$ yields the off-diagonal Hellmann-Feynman relation:
$$\mathbf{d}_{IJ}(\mathbf{R}) = \frac{\langle \Psi_I | (\nabla_{\mathbf{R}} \hat{H}) | \Psi_J \rangle}{E_J(\mathbf{R}) - E_I(\mathbf{R})}$$
Near a conical intersection where $E_J \to E_I$, $\mathbf{d}_{IJ}$ diverges as $1/\Delta E_{IJ}$. The scaled coupling vector:
$$\mathbf{h}_{IJ} = (E_J - E_I) \mathbf{d}_{IJ}$$
remains smooth and non-zero, defining the non-adiabatic coupling direction in the 2D branching space $(g, h)$.

### 5.2 Transition Reduced Density Matrices (TRDMs)
Evaluating $\langle \Psi_I | \hat{H} | \Psi_J \rangle$ requires state-to-state **transition density matrices**:
$$\gamma_{tu}^{IJ} = \langle \Psi_I | \hat{E}_{tu} | \Psi_J \rangle$$
$$\Gamma_{tuvw}^{IJ} = \langle \Psi_I | \hat{e}_{tuvw} | \Psi_J \rangle$$

Key symmetry properties:
- $\gamma_{tu}^{IJ} = \gamma_{ut}^{JI}$
- For $I \neq J$, $\gamma^{IJ}$ is generally **not symmetric** ($\gamma_{tu}^{IJ} \neq \gamma_{ut}^{IJ}$).
- Symmetrized and anti-symmetrized parts play distinct roles:
  $$\gamma_{(tu)}^{IJ} = \frac{1}{2}(\gamma_{tu}^{IJ} + \gamma_{ut}^{IJ}), \quad \gamma_{[tu]}^{IJ} = \frac{1}{2}(\gamma_{tu}^{IJ} - \gamma_{ut}^{IJ})$$

### 5.3 Complete Three-Component Decomposition of the NACV
In SA-CASSCF, the total analytical NACV is partitioned into three distinct contributions:
$$\mathbf{d}_{IJ} = \mathbf{d}_{IJ}^{\text{active}} + \mathbf{d}_{IJ}^{\text{CSF}} + \mathbf{d}_{IJ}^{\text{response}}$$

#### 1. Active-Orbital Hellmann-Feynman Term ($\mathbf{d}_{IJ}^{\text{active}}$)
Arises from the derivative of the Hamiltonian contracted with active transition density matrices:
$$\mathbf{d}_{IJ}^{\text{active}} = \frac{1}{E_J - E_I} \left[ \sum_{tu} \gamma_{tu}^{IJ} \nabla h_{tu} + \frac{1}{2} \sum_{tuvw} \Gamma_{tuvw}^{IJ} \nabla (tu|vw) + \sum_{i, tu} \gamma_{tu}^{IJ} [2\nabla(ii|tu) - \nabla(iu|ti)] \right]$$
In atomic orbital (AO) space, this maps to contractions with $\nabla h_{\mu\nu}$ and $\nabla (\mu\nu|\rho\sigma)$.

#### 2. CSF Derivative / Basis Function Derivative Term ($\mathbf{d}_{IJ}^{\text{CSF}}$)
Because the MOs are expanded in atom-centered AOs $\chi_\mu(\mathbf{r} - \mathbf{R}_A)$, moving nucleus $A$ differentiates the basis functions. Orthogonality of the CSFs gives rise to an overlap derivative term:
$$\mathbf{d}_{IJ}^{\text{CSF}} = \frac{1}{2} \sum_{\mu\nu} (\nabla_{\mathbf{R}} S_{\mu\nu}) \left[ \sum_{t,u \in \text{act}} C_{\mu t} C_{\nu u} (\gamma_{ut}^{IJ} - \gamma_{tu}^{IJ}) \right]$$
Notice this term is proportional to the **anti-symmetric** transition density matrix $\gamma_{[tu]}^{IJ}$.

#### 3. Orbital & CI Response Term ($\mathbf{d}_{IJ}^{\text{response}}$)
Similar to state gradients, the off-diagonal transition Hamiltonian is not stationary with respect to state-averaged orbital rotations. The transition orbital gradient is:
$$L_{pq}^{IJ} = \langle \Psi_I | [\hat{E}_{pq} - \hat{E}_{qp}, \hat{H}] | \Psi_J \rangle = 2 (F_{pq}^{IJ} - F_{qp}^{IJ})$$
We solve the Z-vector CP-MCSCF equation for the transition Lagrange multipliers $\mathbf{z}^{IJ}$:
$$\mathbf{A}^{\text{SA}} \mathbf{z}^{IJ} = -\mathbf{L}^{IJ}$$
and contract $\mathbf{z}^{IJ}$ with the standard derivative integral operators.

### 5.4 The Transition Generalized Fock Matrix ($F^{IJ}$) vs Single-State Fock ($F^I$)
A critical distinction in multireference response theory:
- **Single-State / State-Averaged Fock Matrix ($F^I, F^{\text{SA}}$):**
  Evaluates state expectation values $\langle \Psi_I | \hat{H} | \Psi_I \rangle$. The core electrons are doubly occupied ($D_{ii} = 2$), contributing non-zero diagonal core Fock elements $F_{ii} = 2 \epsilon_i$ and core-core Coulomb/exchange repulsion.
- **Transition Generalized Fock Matrix ($F^{IJ}$, $I \ne J$):**
  Evaluates off-diagonal matrix elements $\langle \Psi_I | \hat{H} | \Psi_J \rangle$. Because states are orthogonal ($\langle \Psi_I | \Psi_J \rangle = 0$), the core transition density is identically zero ($D_{\text{core}}^{IJ} = 0$).
  All terms in $F^{IJ}$ are strictly linear in active transition RDMs ($\gamma^{IJ}, \Gamma^{IJ}$):
  $$F_{pi}^{IJ} = 2 \left( \mathbf{C}^T [2 J(D_{\text{trans}}) - K(D_{\text{trans}})] \mathbf{C}_{\text{core}} \right)_{pi}$$
  $$F_{pt}^{IJ} = \left( \mathbf{C}^T F^{\text{core}} \mathbf{C}_{\text{act}} \gamma^{IJ} \right)_{pt} + \sum_{uvw} (pv|uw) \Gamma_{tuvw}^{IJ}$$
  Consequently, if $\gamma^{IJ} = 0$ and $\Gamma^{IJ} = 0$, $F^{IJ} \equiv 0$ and $L^{IJ} \equiv 0$.

### 5.5 Spin Selection Rules in Non-Adiabatic Couplings
The nuclear derivative operator $\nabla_{\mathbf{R}}$ does not act on spin coordinates:
$$[\nabla_{\mathbf{R}}, \hat{S}^2] = 0, \quad [\nabla_{\mathbf{R}}, \hat{S}_z] = 0$$
Therefore, in non-relativistic quantum mechanics without spin-orbit coupling:
$$\langle \Psi_{S} | \nabla_{\mathbf{R}} | \Psi_{T} \rangle \equiv \mathbf{0} \quad (\Delta S \ne 0 \text{ selection rule})$$
Transitions between Singlet and Triplet manifolds (such as $S_0 \to T_1$ and $S_0 \to T_2$) have identically zero transition RDMs ($\gamma^{ST} = 0$, $\Gamma^{ST} = 0$). Their non-adiabatic coupling vectors vanish identically ($\mathbf{d}_{ST} = \mathbf{0}$). Inter-system crossings in NAMD require spin-orbit coupling (SOC) matrix elements rather than non-adiabatic derivative couplings.


---

## 6. Electron Translation Factors (ETFs) in Dynamics

A critical physical nuance in non-adiabatic dynamics:
Standard atom-centered basis functions without velocity-dependent phase factors violate Galilean invariance. Fatehi and Subotnik (*J. Phys. Chem. Lett.* 2012, 3, 2039) showed that:
- The raw $\mathbf{d}_{IJ}^{\text{CSF}}$ term introduces spurious origin dependence and non-conservation of total electron + nuclear momentum.
- Rigorous incorporation of **Electron Translation Factors (ETFs)** effectively **cancels the CSF derivative term** in the non-adiabatic coupling:
$$\mathbf{d}_{IJ}^{\text{ETF}} = \mathbf{d}_{IJ}^{\text{active}} + \mathbf{d}_{IJ}^{\text{response}} \quad (\text{CSF term omitted})$$
- In `pyscf.nac.sacasscf`, setting `use_etfs=True` enables this physically correct behavior for NAMD. This should be a configurable default in `gpu4pyscf`.


---

## 7. Key Literature References & Citations

1. **GPU SA-CASSCF Analytical Gradients & NACVs:**
   - J. W. Snyder Jr., E. G. Hohenstein, N. Luehr, and T. J. Martínez, *"An atomic orbital-based formulation of analytical gradients and nonadiabatic coupling vector elements for the state-averaged complete active space self-consistent field method on graphical processing units"*, *J. Chem. Phys.* **143**, 154107 (2015). DOI: [`10.1063/1.4932613`](https://doi.org/10.1063/1.4932613). [BibTeX: `Snyder2015`](references.bib#L1)
2. **GPU Density Fitting for Multireference Gradients:**
   - R. Wang, Y. Wang, L. Lu, D. Hait, and T. J. Martínez, *"Complete Active Space Self-Consistent Field with GPU-Accelerated Density Fitting"*, *J. Chem. Theory Comput.* **22**, 3398–3414 (2026). DOI: [`10.1021/acs.jctc.5c02079`](https://doi.org/10.1021/acs.jctc.5c02079). [BibTeX: `Wang2026`](references.bib#L3)
3. **Electron Translation Factors & Momentum Conservation:**
   - S. Fatehi and J. E. Subotnik, *"Derivative Couplings with Built-In Electron-Translation Factors: Application to Benzene"*, *J. Phys. Chem. Lett.* **3**, 2039–2043 (2012). DOI: [`10.1021/jz3006173`](https://doi.org/10.1021/jz3006173). [BibTeX: `Fatehi2012`](references.bib#L41)
   - S. Fatehi, E. Alguire, Y. Shao, and J. E. Subotnik, *"Analytic derivative couplings between configuration-interaction-singles states with built-in electron-translation factors for translational invariance"*, *J. Chem. Phys.* **135**, 234105 (2011). DOI: [`10.1063/1.3665031`](https://doi.org/10.1063/1.3665031). [BibTeX: `Fatehi2011`](references.bib#L25)
4. **PySCF CP-MCSCF & SA-CASSCF NAC Implementation:**
   - Q. Sun et al., *"Recent developments in the PySCF program package"*, *J. Chem. Phys.* **153**, 024109 (2020). DOI: [`10.1063/5.0006074`](https://doi.org/10.1063/5.0006074). [BibTeX: `Sun2020`](references.bib#L35)
