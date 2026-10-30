# 02. Theory: XMS-CASPT2 Energy, Gradients & Derivative Couplings

## 1. Motivation: Why SA-CASSCF Alone Fails for Photochemistry

While SA-CASSCF correctly captures **static (non-dynamic) correlation** and handles near-degenerate frontier orbitals, it completely omits **dynamic electron correlation** (dispersion, instantaneous core-valence, and valence-valence electron repulsion). 

Consequences in photochemical reaction modeling:
1. **Vertical Excitation Errors:** SA-CASSCF consistently overestimates vertical excitation energies by $0.5 - 1.5\text{ eV}$.
2. **State Inversions:** States with strong ionic character (such as $V$-states in polyenes) require substantial dynamical $\sigma$-polarization and are severely destabilized relative to covalent states (such as $2^1A_g^-$ states), leading to incorrect state orderings.
3. **Distorted Conical Intersections:** Relative stabilization of states by dynamic correlation varies drastically between planar Franck-Condon geometries and twisted/pyramidalized conical intersection geometries.

**Second-Order Multireference Perturbation Theory (CASPT2)** introduces dynamic correlation on top of the CASSCF reference. However, standard Multi-State CASPT2 (MS-CASPT2) introduces severe pathologies near conical intersections, necessitating **Extended Multi-State CASPT2 (XMS-CASPT2)**.

---

## 2. The Multi-State Pathology & The XMS-CASPT2 Solution

### 2.1 The Invariance Breakdown in Standard MS-CASPT2
In standard MS-CASPT2 (Finley et al. 1998), an effective Hamiltonian $H_{\alpha\beta}^{\text{eff}}$ is constructed over a set of reference states $\{|\Psi_\alpha\rangle\}_{\alpha=1}^{N_{\text{states}}}$. The zeroth-order Hamiltonian is defined as:
$$\hat{H}_0^{\text{MS}} = \sum_{\alpha=1}^{N_{\text{states}}} |\Psi_\alpha\rangle \hat{f}_\alpha \langle \Psi_\alpha| + \hat{Q} \hat{F} \hat{Q}$$
where $\hat{f}_\alpha$ is the state-specific generalized Fock operator constructed using the density matrix of state $\alpha$:
$$\hat{f}_\alpha = \sum_{pq} \left( h_{pq} + \sum_{rs} D_{rs}^\alpha \left[ (pq|rs) - \frac{1}{2}(ps|rq) \right] \right) \hat{E}_{pq}$$
and $\hat{Q} = 1 - \sum_\alpha |\Psi_\alpha\rangle\langle \Psi_\alpha|$ projects onto the first-order interacting space.

**The Fatal Flaw:** Because each state $\alpha$ has its own distinct Fock operator $\hat{f}_\alpha$, the operator $\hat{H}_0^{\text{MS}}$ **is NOT invariant under unitary rotations within the reference model space**:
$$|\tilde{\Psi}_\alpha\rangle = \sum_\beta U_{\beta\alpha} |\Psi_\beta\rangle \quad \Longrightarrow \quad \hat{H}_0^{\text{MS}}[|\tilde{\Psi}\rangle] \neq \hat{H}_0^{\text{MS}}[|\Psi\rangle]$$
As a direct consequence:
- The effective Hamiltonian $H_{\alpha\beta}^{\text{eff}}$ is non-Hermitian and basis-dependent.
- Near conical intersections, state mixing causes artificial avoided crossings, distorted seams, or non-physical cusps in adiabatic potential energy surfaces.
- Analytical gradients and derivative couplings suffer from unphysical singularities.

---

### 2.2 The XMS-CASPT2 Invariant Formulation
To restore exact unitary invariance, Granovsky (2011, XMC-QDPT) and Shiozaki et al. (2011, XMS-CASPT2) redefined the zeroth-order Hamiltonian using an **average Fock operator** and an **orthogonalized model space basis**.

#### Step 1: State-Averaged Active Space Fock Matrix
We define the state-averaged Fock operator over the $N_{\text{states}}$ target states:
$$\bar{F} = \frac{1}{N_{\text{states}}} \sum_{\alpha=1}^{N_{\text{states}}} \hat{f}_\alpha$$
In the CASSCF reference basis $\{|\Psi_\alpha\rangle\}$, we construct the model-space Fock matrix:
$$F_{\alpha\beta}^{(0)} = \langle \Psi_\alpha | \bar{F} | \Psi_\beta \rangle = \sum_{pq} \bar{F}_{pq} \langle \Psi_\alpha | \hat{E}_{pq} | \Psi_\beta \rangle = \sum_{t, u \in \text{act}} \bar{F}_{tu} \gamma_{tu}^{\alpha\beta} + \delta_{\alpha\beta} \sum_{i \in \text{core}} 2\bar{F}_{ii}$$
where $\gamma_{tu}^{\alpha\beta} = \langle \Psi_\alpha | \hat{E}_{tu} | \Psi_\beta \rangle$ is the active-space transition 1-RDM between states $\alpha$ and $\beta$.

#### Step 2: Diagonalization of the Model-Space Fock Matrix
We diagonalize $\mathbf{F}^{(0)}$ via a unitary rotation matrix $\mathbf{U}$:
$$\mathbf{U}^\dagger \mathbf{F}^{(0)} \mathbf{U} = \mathbf{f}^{\text{diag}}, \quad f_\alpha \delta_{\alpha\beta} = \sum_{\gamma,\delta} U_{\gamma\alpha} F_{\gamma\delta}^{(0)} U_{\delta\beta}$$
This defines the rotated reference states $\{|\tilde{\Psi}_\alpha\rangle\}$:
$$|\tilde{\Psi}_\alpha\rangle = \sum_{\beta=1}^{N_{\text{states}}} U_{\beta\alpha} |\Psi_\beta\rangle$$

#### Step 3: Strictly Invariant Zeroth-Order Hamiltonian
In this rotated basis, the XMS zeroth-order Hamiltonian is defined as:
$$\hat{H}_0^{\text{XMS}} = \sum_{\alpha=1}^{N_{\text{states}}} |\tilde{\Psi}_\alpha\rangle \bar{F} \langle \tilde{\Psi}_\alpha| + \hat{Q} \bar{F} \hat{Q}$$
Because $\mathbf{F}^{(0)}$ is diagonal in the basis $\{|\tilde{\Psi}_\alpha\rangle\}$, we have:
$$\langle \tilde{\Psi}_\alpha | \hat{H}_0^{\text{XMS}} | \tilde{\Psi}_\beta \rangle = f_\alpha \delta_{\alpha\beta}$$
**Theorem:** $\hat{H}_0^{\text{XMS}}$ is strictly invariant under any unitary transformation applied to the original reference states $\{|\Psi_\alpha\rangle\}$. 

#### Step 4: The Effective Second-Order Hamiltonian
The first-order perturbation functions $|\Psi_\alpha^{(1)}\rangle$ are solved independently for each rotated reference state $|\tilde{\Psi}_\alpha\rangle$:
$$(\hat{H}_0^{\text{XMS}} - f_\alpha) |\Psi_\alpha^{(1)}\rangle = - \hat{Q} \hat{H} |\tilde{\Psi}_\alpha\rangle$$
The second-order effective Hamiltonian is constructed:
$$H_{\alpha\beta}^{\text{eff}} = \langle \tilde{\Psi}_\alpha | \hat{H} | \tilde{\Psi}_\beta \rangle + \frac{1}{2} \left[ \langle \tilde{\Psi}_\alpha | \hat{H} | \Psi_\beta^{(1)} \rangle + \langle \Psi_\alpha^{(1)} | \hat{H} | \tilde{\Psi}_\beta \rangle \right]$$
Notice $H_{\alpha\beta}^{\text{eff}}$ is **manifestly Hermitian by construction**.

Finally, diagonalizing $\mathbf{H}^{\text{eff}}$ yields the final XMS-CASPT2 energies $E_K$ and adiabatic states $|K\rangle$:
$$\mathbf{V}^\dagger \mathbf{H}^{\text{eff}} \mathbf{V} = \mathbf{E}_{\text{XMS-CASPT2}}$$
$$|K\rangle = \sum_{\alpha=1}^{N_{\text{states}}} V_{\alpha K} |\tilde{\Psi}_\alpha\rangle$$

---

## 3. Internally Contracted Perturbation Equations

In **internally contracted CASPT2 (ic-CASPT2)**, the first-order wave function $|\Psi_\alpha^{(1)}\rangle$ is expanded by applying two-electron excitation operators to the reference state $|\tilde{\Psi}_\alpha\rangle$:
$$|\Psi_\alpha^{(1)}\rangle = \sum_{\mu} T_\mu^\alpha |\mu\rangle, \quad |\mu\rangle = \hat{E}_{pq,rs} |\tilde{\Psi}_\alpha\rangle$$
where $\hat{E}_{pq,rs} = \hat{E}_{pq}\hat{E}_{rs} - \delta_{qr}\hat{E}_{ps}$.

### 3.1 The 8 Excitation Manifolds
The configurations are classified into 8 orthogonal excitation classes depending on indices:
1. **$A$ (Core $\to$ Virtual):** $\hat{E}_{ai,bj}$
2. **$B$ (Active $\to$ Virtual):** $\hat{E}_{at,bu}$
3. **$C$ (Core $\to$ Active):** $\hat{E}_{ti,uj}$
4. **$D$ (Core, Core $\to$ Virtual, Active):** $\hat{E}_{ai,tj}$
5. **$E$ (Core, Active $\to$ Virtual, Virtual):** $\hat{E}_{ai,bt}$
6. **$F$ (Core, Core $\to$ Virtual, Virtual):** $\hat{E}_{ai,bj}$ (pure MP2-like pairs)
7. **$G$ (Active, Active $\to$ Virtual, Virtual):** $\hat{E}_{at,bu}$
8. **$H$ (Core, Core $\to$ Active, Active):** $\hat{E}_{ti,uj}$

### 3.2 Overlap Metric & Active-Space RDMs
Internally contracted states $|\mu\rangle$ are not orthogonal; their overlap metric is:
$$S_{\mu\nu} = \langle \mu | \nu \rangle = \langle \tilde{\Psi}_\alpha | \hat{E}_{pq,rs}^\dagger \hat{E}_{p'q',r's'} | \tilde{\Psi}_\alpha \rangle$$
Evaluating $S_{\mu\nu}$ contracts up to the **3-particle reduced density matrix (3-RDM)** of the active space:
$$\Gamma_{tuvwxy} = \langle \tilde{\Psi}_\alpha | \hat{E}_{tu} \hat{E}_{vw} \hat{E}_{xy} | \tilde{\Psi}_\alpha \rangle + \dots$$
and for certain uncontracted variants, the 4-RDM.
In Celani-Werner partially contracted schemes, projector approximations eliminate explicit 4-RDMs, leaving active-space requirements at **1-RDM, 2-RDM, and 3-RDM**.

Metric singularities ($S_{\mu\nu} v_k = \sigma_k v_k$ with $\sigma_k < 10^{-7}$) are diagonalized and eliminated (canonical orthogonalization) before solving the linear system for amplitudes $T_\mu^\alpha$:
$$(\mathbf{F}_0 - f_\alpha \mathbf{S}) \mathbf{T}^\alpha = -\mathbf{V}^\alpha$$

---

## 4. Intruder States & Level Shifts

When an energy denominator $(f_i + f_j - f_a - f_b)$ approaches zero, the perturbation expansion diverges (the **intruder state problem**).

### 4.1 Real Level Shift (Forsberg & Malmqvist 1997)
Adds a constant real shift $\epsilon$ to the zeroth-order Hamiltonian denominator:
$$\frac{1}{\Delta E} \longrightarrow \frac{1}{\Delta E + \epsilon}$$
and applies an energetic correction: $\Delta E_{\text{corr}} = \epsilon \sum_\mu (T_\mu^\alpha)^2$.

### 4.2 Imaginary Level Shift (Park & Shiozaki 2019)
For gradients and derivative couplings, real level shifts can still produce derivative discontinuities. Park and Shiozaki introduced an **imaginary shift** $i\epsilon$:
$$\frac{1}{\Delta E + i\epsilon} = \frac{\Delta E}{\Delta E^2 + \epsilon^2} - i \frac{\epsilon}{\Delta E^2 + \epsilon^2}$$
Taking the real part of the resulting energy gives:
$$E^{(2)}(\epsilon) = \sum_\mu \frac{\Delta E_\mu |V_\mu|^2}{\Delta E_\mu^2 + \epsilon^2}$$
**Advantages for Photodynamics:**
- $E^{(2)}(\epsilon)$ is strictly real, smooth, and singularity-free everywhere in nuclear coordinate space.
- Analytical gradients and derivative couplings have continuous, well-behaved derivatives across conical intersection seams.

---

## 5. Analytical Gradients & Derivative Couplings for XMS-CASPT2

To compute the non-adiabatic derivative coupling between XMS-CASPT2 states $K$ and $L$:
$$\mathbf{d}_{KL}^{\text{XMS-CASPT2}} = \langle K | \nabla_{\mathbf{R}} L \rangle$$
we construct the fully variational Lagrangian (Park & Shiozaki, *J. Chem. Theory Comput.* 2017, 13, 2561):
$$\mathcal{L}_{KL} = H_{KL}^{\text{eff}} + \sum_{\alpha, \mu} \lambda_{\mu}^{\alpha} R_{\mu}^\alpha + \sum_{\alpha > \beta} \mu_{\alpha\beta} F_{\alpha\beta}^{(0)} + \sum_{p>q} z_{pq} g_{pq}^{\text{SA}} + \sum_{\alpha=1}^{N_{\text{states}}} \sum_{\mu} z_{\mu\alpha}^{\text{ci}} [(\mathbf{H}_{\text{act}} - E_\alpha) \mathbf{C}_\alpha]_\mu$$

The Lagrangian enforces:
1. **PT2 Amplitude Conditions ($R_\mu^\alpha = 0$):** Stationarity of first-order wave functions with multipliers $\boldsymbol{\lambda}$.
2. **Model Space Diagonalization ($F_{\alpha\beta}^{(0)} = 0$ for $\alpha \neq \beta$):** Multipliers $\boldsymbol{\mu}$.
3. **Orbital Stationarity ($g_{pq}^{\text{SA}} = 0$):** Z-vector orbital multipliers $\mathbf{z}^{\text{orb}}$.
4. **CI Eigenvalue Condition:** Z-vector CI multipliers $\mathbf{z}^{\text{ci}}$.

### Back-Propagation / Z-Vector Sequence:
1. **Solve for $\boldsymbol{\lambda}$:** Solve the adjoint CASPT2 linear equations with source term $\partial H_{KL}^{\text{eff}} / \partial T$.
2. **Solve for $\boldsymbol{\mu}$:** Determine multipliers for the model-space rotation condition.
3. **Solve for $\mathbf{z}^{\text{ci}}$ and $\mathbf{z}^{\text{orb}}$:** Solve the coupled CP-MCSCF Z-vector equations on the GPU.
4. **Assemble Effective Density Matrices ($P^{\text{eff}}, \Gamma^{\text{eff}}, W^{\text{eff}}$):**
   $$\mathbf{d}_{KL} = \sum_{\mu\nu} P_{\mu\nu}^{\text{eff}} \nabla h_{\mu\nu} + \sum_{\mu\nu\rho\sigma} \Gamma_{\mu\nu\rho\sigma}^{\text{eff}} \nabla (\mu\nu|\rho\sigma) + \sum_{\mu\nu} W_{\mu\nu}^{\text{eff}} \nabla S_{\mu\nu}$$
Under Density Fitting, the 4-center term simplifies directly into GPU-evaluated 3-center contractions:
$$\sum_{P\mu\nu} \gamma_{P, \mu\nu}^{\text{eff}} \nabla (P|\mu\nu) + \sum_{PQ} \gamma_{PQ}^{\text{eff}} \nabla (P|Q)$$
matching `gpu4pyscf`'s native gradient infrastructure!


---

## 6. Key Literature References & Citations

1. **Origin of Extended Model-Space Invariant Perturbation Theory:**
   - A. A. Granovsky, *"Extended multi-configuration quasi-degenerate perturbation theory: The new approach to multi-state multi-reference perturbation theory"*, *J. Chem. Phys.* **134**, 214113 (2011). DOI: [`10.1063/1.3596699`](https://doi.org/10.1063/1.3596699). [BibTeX: `Granovsky2011`](references.bib#L7)
2. **Definition of Internally Contracted XMS-CASPT2:**
   - T. Shiozaki, W. Győrffy, P. Celani, and H.-J. Werner, *"Communication: Extended multi-state complete active space second-order perturbation theory: Energy and nuclear gradients"*, *J. Chem. Phys.* **135**, 081106 (2011). DOI: [`10.1063/1.3633329`](https://doi.org/10.1063/1.3633329). [BibTeX: `Shiozaki2011`](references.bib#L9)
3. **Analytical Derivative Couplings for XMS-CASPT2:**
   - J. W. Park and T. Shiozaki, *"Analytical Derivative Coupling for Multistate CASPT2 Theory"*, *J. Chem. Theory Comput.* **13**, 2561–2570 (2017). DOI: [`10.1021/acs.jctc.7b00018`](https://doi.org/10.1021/acs.jctc.7b00018). [BibTeX: `Park2017`](references.bib#L11)
4. **Imaginary Shift Formulation for Non-Singular Derivatives:**
   - J. W. Park, R. Al-Saadon, N. E. Strand, and T. Shiozaki, *"Imaginary Shift in CASPT2 Nuclear Gradient and Derivative Coupling Theory"*, *J. Chem. Theory Comput.* **15**, 4088–4098 (2019). DOI: [`10.1021/acs.jctc.9b00368`](https://doi.org/10.1021/acs.jctc.9b00368). [BibTeX: `Park2019`](references.bib#L13)
5. **Nuclear Gradients & Lagrangian Z-Vector Pipeline:**
   - B. Vlaisavljevich and T. Shiozaki, *"Nuclear Energy Gradients for Internally Contracted Complete Active Space Second-Order Perturbation Theory: Multistate Extensions"*, *J. Chem. Theory Comput.* **12**, 3781–3787 (2016). DOI: [`10.1021/acs.jctc.6b00572`](https://doi.org/10.1021/acs.jctc.6b00572). [BibTeX: `Vlaisavljevich2016`](references.bib#L15)
6. **Internally Contracted Excitation Manifolds (Classes A-H):**
   - P. Celani and H.-J. Werner, *"Multireference perturbation theory for large restricted and selected active space reference wave functions"*, *J. Chem. Phys.* **112**, 5546–5557 (2000). DOI: [`10.1063/1.481132`](https://doi.org/10.1063/1.481132). [BibTeX: `Celani2000`](references.bib#L19)
7. **Original Multi-State CASPT2 Formulation:**
   - J. Finley, P.-Å. Malmqvist, B. O. Roos, and L. Serrano-Andrés, *"The multi-state CASPT2 method"*, *Chem. Phys. Lett.* **288**, 299–306 (1998). DOI: [`10.1016/S0009-2614(98)00252-8`](https://doi.org/10.1016/S0009-2614(98)00252-8). [BibTeX: `Finley1998`](references.bib#L21)
