# 07. Annotated Literature Bibliography & Reference Database

This document provides a comprehensive annotated bibliography of the foundational peer-reviewed literature underpinning the GPU-accelerated SA-CASSCF, XMS-CASPT2, and Non-Adiabatic Coupling (NAC) project. 

The accompanying machine-readable BibTeX database is located at [`references.bib`](references.bib).

---

## 1. SA-CASSCF Gradients & GPU Acceleration

### [`Snyder2015`](references.bib#L1)
* **Full Citation:** J. W. Snyder Jr., E. G. Hohenstein, N. Luehr, and T. J. Martínez, *"An atomic orbital-based formulation of analytical gradients and nonadiabatic coupling vector elements for the state-averaged complete active space self-consistent field method on graphical processing units"*, **The Journal of Chemical Physics** 143, 154107 (2015).
* **DOI:** [`10.1063/1.4932613`](https://doi.org/10.1063/1.4932613)
* **Key Implementation Takeaways:**
  - Formulates the nuclear gradient and NACVs directly in the atomic orbital (AO) basis, exploiting AO sparsity to drop formal scaling from $O(N^4)$ to $O(N^2)$.
  - Maps the Coupled-Perturbed MCSCF (CP-MCSCF) linear equations onto GPU-accelerated iterative matrix-vector multiplications via generalized Coulomb and exchange builders.
  - Demonstrates that GPU acceleration enables on-the-fly dynamics and conical intersection searches for molecules exceeding 1,000 atoms.

### [`Wang2026`](references.bib#L3)
* **Full Citation:** R. Wang, Y. Wang, L. Lu, D. Hait, and T. J. Martínez, *"Complete Active Space Self-Consistent Field with GPU-Accelerated Density Fitting"*, **Journal of Chemical Theory and Computation** 22(7), 3398–3414 (2026).
* **DOI:** [`10.1021/acs.jctc.5c02079`](https://doi.org/10.1021/acs.jctc.5c02079)
* **Key Implementation Takeaways:**
  - Introduces GPU-accelerated Density Fitting (DF) for CASSCF energy and analytical gradient evaluations in TeraChem.
  - Replaces 4-center ERIs with 3-center tensors $(P|\mu\nu)$, providing a $3\times - 30\times$ speedup per time step in nonadiabatic dynamics with negligible energy deviations ($< 10\ \mu\text{Hartree}$).

### [`Snyder2016`](references.bib#L5)
* **Full Citation:** J. W. Snyder Jr., B. F. E. Curchod, and T. J. Martínez, *"GPU-Accelerated State-Averaged Complete Active Space Self-Consistent Field Interfaced with Ab Initio Multiple Spawning Unravels the Photodynamics of Provitamin D3"*, **The Journal of Physical Chemistry Letters** 7(13), 2444–2449 (2016).
* **DOI:** [`10.1021/acs.jpclett.6b00970`](https://doi.org/10.1021/acs.jpclett.6b00970)
* **Key Implementation Takeaways:**
  - Validates GPU-accelerated multireference on-the-fly dynamics for complex photochemical ring-opening and internal conversion.
  - Demonstrates robust convergence of GPU Krylov solvers near conical intersections.

---

## 2. Extended Multi-State CASPT2 (XMS-CASPT2) Theory & Invariance

### [`Granovsky2011`](references.bib#L7)
* **Full Citation:** A. A. Granovsky, *"Extended multi-configuration quasi-degenerate perturbation theory: The new approach to multi-state multi-reference perturbation theory"*, **The Journal of Chemical Physics** 134(21), 214113 (2011).
* **DOI:** [`10.1063/1.3596699`](https://doi.org/10.1063/1.3596699)
* **Key Implementation Takeaways:**
  - Proves the fundamental invariance failure of standard MS-CASPT2 / MC-QDPT under unitary rotations of the reference state space.
  - Proposes diagonalizing the state-averaged active-space generalized Fock operator in the reference space to create an invariant zeroth-order Hamiltonian.

### [`Shiozaki2011`](references.bib#L9)
* **Full Citation:** T. Shiozaki, W. Győrffy, P. Celani, and H.-J. Werner, *"Communication: Extended multi-state complete active space second-order perturbation theory: Energy and nuclear gradients"*, **The Journal of Chemical Physics** 135(8), 081106 (2011).
* **DOI:** [`10.1063/1.3633329`](https://doi.org/10.1063/1.3633329)
* **Key Implementation Takeaways:**
  - Adapts Granovsky's invariant scheme to internally contracted CASPT2, defining the **XMS-CASPT2** method.
  - Establishes that the effective second-order Hamiltonian $H_{\alpha\beta}^{\text{eff}}$ is Hermitian by construction, eliminating heuristic symmetrizations and yielding smooth potential energy surfaces across conical intersections.

### [`Finley1998`](references.bib#L21)
* **Full Citation:** J. Finley, P.-Å. Malmqvist, B. O. Roos, and L. Serrano-Andrés, *"The multi-state CASPT2 method"*, **Chemical Physics Letters** 288(2-4), 299–306 (1998).
* **DOI:** [`10.1016/S0009-2614(98)00252-8`](https://doi.org/10.1016/S0009-2614(98)00252-8)
* **Key Implementation Takeaways:**
  - Foundational paper defining the original multi-state CASPT2 formulation; provides the reference against which the XMS-CASPT2 invariant formulation is evaluated.

---

## 3. Analytical Gradients & Derivative Couplings for XMS-CASPT2

### [`Park2017`](references.bib#L11)
* **Full Citation:** J. W. Park and T. Shiozaki, *"Analytical Derivative Coupling for Multistate CASPT2 Theory"*, **Journal of Chemical Theory and Computation** 13(6), 2561–2570 (2017).
* **DOI:** [`10.1021/acs.jctc.7b00018`](https://doi.org/10.1021/acs.jctc.7b00018)
* **Key Implementation Takeaways:**
  - Formulates the exact Lagrangian and Z-vector linear response equations for analytical derivative couplings (NACVs) in MS-CASPT2 and XMS-CASPT2.
  - Proves that the computational cost of evaluating analytical derivative couplings is essentially identical to calculating standard nuclear gradients.
  - Implemented in BAGEL; demonstrated for MECI optimizations of stilbene and GFP chromophore models.

### [`Park2019`](references.bib#L13)
* **Full Citation:** J. W. Park, R. Al-Saadon, N. E. Strand, and T. Shiozaki, *"Imaginary Shift in CASPT2 Nuclear Gradient and Derivative Coupling Theory"*, **Journal of Chemical Theory and Computation** 15(7), 4088–4098 (2019).
* **DOI:** [`10.1021/acs.jctc.9b00368`](https://doi.org/10.1021/acs.jctc.9b00368)
* **Key Implementation Takeaways:**
  - Replaces conventional real level shifts with an imaginary level shift ($i\epsilon$) in the CASPT2 energy denominators.
  - Proves that the real part of the resulting perturbation energy is strictly smooth and non-divergent across electronic degeneracies, completely eliminating intruder-state derivative singularities.

### [`Vlaisavljevich2016`](references.bib#L15)
* **Full Citation:** B. Vlaisavljevich and T. Shiozaki, *"Nuclear Energy Gradients for Internally Contracted Complete Active Space Second-Order Perturbation Theory: Multistate Extensions"*, **Journal of Chemical Theory and Computation** 12(8), 3781–3787 (2016).
* **DOI:** [`10.1021/acs.jctc.6b00572`](https://doi.org/10.1021/acs.jctc.6b00572)
* **Key Implementation Takeaways:**
  - Formulates nuclear gradients for internally contracted multi-state CASPT2.
  - Details the back-propagation chain from CASPT2 first-order amplitude multipliers to reference CI and orbital response Z-vector equations.

---

## 4. Density Fitting & Internal Contraction

### [`Celani2000`](references.bib#L19)
* **Full Citation:** P. Celani and H.-J. Werner, *"Multireference perturbation theory for large restricted and selected active space reference wave functions"*, **The Journal of Chemical Physics** 112(13), 5546–5557 (2000).
* **DOI:** [`10.1063/1.481132`](https://doi.org/10.1063/1.481132)
* **Key Implementation Takeaways:**
  - Defines the 8 excitation classes ($A$ through $H$) for internally contracted CASPT2.
  - Introduces projector approximations that eliminate the explicit active-space 4-particle RDM, requiring only 1-, 2-, and 3-particle active RDMs.

### [`Gyorffy2013`](references.bib#L17)
* **Full Citation:** W. Győrffy, T. Shiozaki, G. Knizia, and H.-J. Werner, *"Analytical energy gradients for second-order multireference perturbation theory using density fitting"*, **The Journal of Chemical Physics** 138(10), 104104 (2013).
* **DOI:** [`10.1063/1.4793737`](https://doi.org/10.1063/1.4793737)
* **Key Implementation Takeaways:**
  - Formulates density fitting for CASPT2 analytical gradients, mapping 4-center electron repulsion integrals into 3-center contractions.

### [`Aquilante2007`](references.bib#L39)
* **Full Citation:** F. Aquilante, T. B. Pedersen, and R. Lindh, *"Unbiased auxiliary basis sets for accurate two-electron integral approximations"*, **The Journal of Chemical Physics** 126(19), 194106 (2007).
* **DOI:** [`10.1063/1.2777146`](https://doi.org/10.1063/1.2777146)
* **Key Implementation Takeaways:**
  - Evaluates auxiliary basis sets and Cholesky decomposition approximations for multiconfigurational perturbation theory.

### [`Forsberg1997`](references.bib#L23)
* **Full Citation:** N. Forsberg and P.-Å. Malmqvist, *"Multiconfiguration perturbation theory with imaginary level shift"*, **Chemical Physics Letters** 274(1-3), 196–204 (1997).
* **DOI:** [`10.1016/S0009-2614(97)00669-6`](https://doi.org/10.1016/S0009-2614(97)00669-6)
* **Key Implementation Takeaways:**
  - Foundational formulation of level shift techniques for avoiding intruder state divergences in CASPT2.

---

## 5. Non-Adiabatic Dynamics, Surface Hopping & ETFs

### [`Tully1990`](references.bib#L27)
* **Full Citation:** J. C. Tully, *"Molecular dynamics with electronic transitions"*, **The Journal of Chemical Physics** 93(2), 1061–1071 (1990).
* **DOI:** [`10.1063/1.459170`](https://doi.org/10.1063/1.459170)
* **Key Implementation Takeaways:**
  - Foundational paper introducing Tully's Fewest Switches Surface Hopping (FSSH).
  - Establishes the relationship between non-adiabatic coupling vectors $\mathbf{d}_{ij}$, hopping probabilities, and nuclear momentum adjustment.

### [`Fatehi2011`](references.bib#L25) & [`Fatehi2012`](references.bib#L41)
* **Full Citations:**
  - S. Fatehi, E. Alguire, Y. Shao, and J. E. Subotnik, *"Analytic derivative couplings between configuration-interaction-singles states with built-in electron-translation factors for translational invariance"*, **The Journal of Chemical Physics** 135(23), 234105 (2011). DOI: [`10.1063/1.3665031`](https://doi.org/10.1063/1.3665031).
  - S. Fatehi and J. E. Subotnik, *"Derivative Couplings with Built-In Electron-Translation Factors: Application to Benzene"*, **The Journal of Physical Chemistry Letters** 3(15), 2039–2043 (2012). DOI: [`10.1021/jz3006173`](https://doi.org/10.1021/jz3006173).
* **Key Implementation Takeaways:**
  - Proves that the atomic-orbital overlap derivative (CSF) term $\frac{1}{2} \sum \nabla S_{\mu\nu} (\gamma_{ut} - \gamma_{tu})$ violates Galilean invariance and breaks total momentum conservation in moving frames.
  - Demonstrates that rigorous inclusion of Electron Translation Factors (ETFs) in non-adiabatic dynamics is equivalent to **omitting the CSF derivative contribution**, providing origin-independent and momentum-conserving forces for surface hopping.

### [`Granucci2007`](references.bib#L29)
* **Full Citation:** G. Granucci and M. Persico, *"Critical appraisal of the fewest switches algorithm for surface hopping"*, **The Journal of Chemical Physics** 126(13), 134114 (2007).
* **DOI:** [`10.1063/1.2715585`](https://doi.org/10.1063/1.2715585)
* **Key Implementation Takeaways:**
  - Formulates the Energy-based Decoherence Correction (EDC) to damp unphysical electronic wave packet coherence along exiting trajectories.

### [`Bearpark1994`](references.bib#L31)
* **Full Citation:** M. J. Bearpark, M. A. Robb, and H. B. Schlegel, *"A direct method for the location of the lowest energy point on a potential surface crossing"*, **Chemical Physics Letters** 223(3), 269–274 (1994).
* **DOI:** [`10.1016/0009-2614(94)00433-1`](https://doi.org/10.1016/0009-2614(94)00433-1)
* **Key Implementation Takeaways:**
  - Introduces the projected gradient algorithm to optimize Minimum Energy Conical Intersections (MECIs) by separating the 2D branching space from the $(3N_{\text{atom}}-8)$ intersection seam.

---

## 6. PySCF Core Framework References

### [`Sun2018`](references.bib#L33) & [`Sun2020`](references.bib#L35)
* **Full Citations:**
  - Q. Sun et al., *"PySCF: the Python-based simulations of chemistry framework"*, **WIREs Computational Molecular Science** 8(1), e1340 (2018). DOI: [`10.1002/wcms.1340`](https://doi.org/10.1002/wcms.1340).
  - Q. Sun et al., *"Recent developments in the PySCF program package"*, **The Journal of Chemical Physics** 153(2), 024109 (2020). DOI: [`10.1063/5.0006074`](https://doi.org/10.1063/5.0006074).
* **Key Implementation Takeaways:**
  - Establishes the core modular software patterns, integral engines (libcint), CI determinant address mappings, and the base CPU implementation of `pyscf.nac.sacasscf`.
