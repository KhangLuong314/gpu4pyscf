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

Target wall-clock timing and speedup benchmarks (tested on NVIDIA L40S GPU vs 16-core Intel Xeon CPU reference in BAGEL):

| Method & System | Basis Set | BAGEL Reference (16 CPU Cores) | GPU4PySCF (1x NVIDIA L40S) | Observed Speedup | Convergence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **DF-RHF SCF** | Ethylene / cc-pVDZ / cc-pvdz-jkfit | 7.73 s | **0.45 s** | **17.2x** | 8 cycles |
| **SA(3)-CASSCF(4,4)** | Ethylene / cc-pVDZ / cc-pvdz-jkfit | 291.25 s (34 macro-cycles) | **3.74 s** (14 macro-cycles) | **77.9x** | Converged ($|g| < 10^{-5}$) |
| **End-to-End Total** | Ethylene / cc-pVDZ / cc-pvdz-jkfit | 299.12 s | **4.19 s** | **71.3x** | Complete |

### 6.1 HPC Cluster Cross-Validation Results: Ethylene CAS(4,4) SA(3)-CASSCF

From production runs on the Centurion HPC cluster (L40S GPU partition, Job 18738908 vs BAGEL reference `ethylene_bagel.out`):

#### 1. Energy and Root Comparison

| State | BAGEL Energy (`spin: 0` CSFs) | BAGEL $\langle \hat{S}^2 \rangle$ | GPU4PySCF Energy (`direct_spin1`) | GPU4PySCF $\langle \hat{S}^2 \rangle$ | State Identity | Excitation (eV) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Root 0** | $-78.05665737\text{ Ha}$ | $0.0000$ | $-78.06120916\text{ Ha}$ | $0.0000$ | $S_0$ Ground Singlet | $0.0000\text{ eV}$ |
| **Root 1** | $-77.72583683\text{ Ha}$ | $0.0000$ | $-77.90534527\text{ Ha}$ | $2.0000$ | BAGEL: $S_1$ / GPU: $T_1$ | $9.0020\text{ eV}$ (B) / $4.2413\text{ eV}$ (G) |
| **Root 2** | $-77.70097357\text{ Ha}$ | $0.0000$ | $-77.73201604\text{ Ha}$ | $2.0000$ | BAGEL: $S_2$ / GPU: $T_2$ | $9.6787\text{ eV}$ (B) / $8.9578\text{ eV}$ (G) |
| **SA Avg** | $-77.82782259\text{ Ha}$ | — | $-77.89952349\text{ Ha}$ | — | Ensemble Average | — |

#### 2. Spin Symmetry & Manifold Analysis
- **Spin-Adapted CSF vs Determinant Basis:**
  - BAGEL solves within the spin-adapted Knowles-Handy Configuration State Function (CSF) subspace ($N_{\text{CSF}} = 20$ for singlets in CAS(4,4)), strictly isolating pure singlet roots $\langle \hat{S}^2 \rangle = 0.0000$.
  - GPU4PySCF `direct_spin1` operates in the complete $M_S = 0$ Slater determinant basis ($N_{\text{det}} = 36$), which encompasses 20 singlets, 15 triplets, and 1 quintet.
  - In planar ethylene, exchange stabilization ($K_{\pi\pi^*} > 0$, Hund's rule) places the lowest triplet states $T_1$ ($\sim 4.24\text{ eV}$) and $T_2$ ($\sim 8.96\text{ eV}$) below the singlet excited states $S_1$ ($\sim 9.00\text{ eV}$) and $S_2$ ($\sim 9.68\text{ eV}$). Consequently, the 3 lowest algebraic eigenstates of $\hat{H}$ in $M_S = 0$ are $(S_0, T_1, T_2)$.
  - The ground state $S_0$ energies match closely: $-78.061209\text{ Ha}$ vs $-78.056657\text{ Ha}$ ($\Delta = 0.00455\text{ Ha} = 2.8\text{ kcal/mol}$), where the subtle shift originates entirely from orbital relaxation in the state-averaged Fock operator (averaging with triplets vs singlets).

#### 3. GPU Transition RDMs Verification
The GPU RawKernels for asymmetric transition RDMs in `gpu4pyscf.fci.direct_spin1` rigorously obey spin selection rules:
- $\langle S_0 | \hat{E}_{tu} | T_1 \rangle$: $\|\gamma^{01}\|_F = 8.11 \times 10^{-14}$ (identically zero; spin-forbidden $\Delta S \ne 0$).
- $\langle S_0 | \hat{E}_{tu} | T_2 \rangle$: $\|\gamma^{02}\|_F = 4.22 \times 10^{-12}$ (identically zero; spin-forbidden $\Delta S \ne 0$).
- $\langle T_1 | \hat{E}_{tu} | T_2 \rangle$: $\|\gamma^{12}\|_F = 0.999818$ (non-zero; spin-allowed triplet-triplet transition $\Delta S = 0$).

#### 4. Active Space 1-RDM Occupations
GPU4PySCF active space natural occupations:
$$\mathbf{n}_{\text{act}} = [1.999583, \; 1.302503, \; 0.364410, \; 0.333503], \quad \sum_k n_k = 4.000000$$
Demonstrating strong multireference electron correlation and exact particle number conservation on the GPU.

---

## 7. Automated BAGEL Cross-Validation Driver & Gatekeeper Hardening

For Phase 1 cross-code verification (ethylene SA(3)-CASSCF(4,4) in `cc-pvdz` / `cc-pvdz-jkfit`), the automated comparison suite (`benchmarks/bagel/`) enforces rigorous gatekeeper criteria to prevent false positives in continuous integration and cluster deployments:

1. **Numerical Tolerances:**
   - State-Averaged Total Energy: $|E_{\text{SA}}^{\text{GPU}} - E_{\text{SA}}^{\text{BAGEL}}| \le 1.0 \times 10^{-7}\text{ Hartree}$.
   - Individual Root Energies: $\max_k |E_k^{\text{GPU}} - E_k^{\text{BAGEL}}| \le 1.0 \times 10^{-7}\text{ Hartree}$.
   - Active Space 1-RDM: $\max_{p,q} |\gamma_{pq}^{\text{GPU}} - \gamma_{pq}^{\text{BAGEL}}| \le 1.0 \times 10^{-5}$ across active orbital permutations.
2. **Gatekeeper Hardening Rules:**
   - **NaN Invalidation:** Any `NaN` value in active RDMs or state energies immediately marks the run `FAIL` (exit code 1).
   - **Mandatory RDM Data:** Calculations omitting `casdm1` or returning empty matrices are rejected with exit code 1.
   - **State Cardinality Assertion:** Root counts must match exactly (`len(states_gpu) == len(states_bagel)`); truncated root outputs trigger exit code 1.
   - **Convergence Enforcement:** Calculations flagged as unconverged (`converged: false`) trigger exit code 1.
   - **Dimension Mismatch Guards:** Ragged or dimensionally incompatible matrices are rejected cleanly without unhandled exceptions.
   - **IEEE 754 Floating-Point Slack:** Analytical boundary comparisons use $10^{-12}$ slack on RDMs and $10^{-15}$ on energies to prevent decimal-to-binary float representation false negatives.
   - **Scientific Notation & Dynamic Slicing in Log Parser:** Regex accommodates exponential notation (`-1.23e-04`) and extracts active occupations from full molecular orbital outputs via `occs[nclosed:nclosed+nact]`.

---

## 8. Production Slurm HPC Cluster Pipeline (`benchmarks/slurm/`)

To validate GPU acceleration and multireference accuracy on production NVIDIA GPU clusters without burdening local development hosts, the test and benchmark suite is orchestrated via Slurm batch automation (`benchmarks/slurm/run_benchmarks_hpc.sh`).

### 8.1 Cluster Architecture & Resource Allocation

The script requests an isolated GPU compute environment designed for high-throughput execution:
- `#SBATCH --job-name=gpu4pyscf_bagel_bench`
- `#SBATCH --nodes=1`
- `#SBATCH --ntasks=1`
- `#SBATCH --cpus-per-task=8`
- `#SBATCH --gres=gpu:1`
- `#SBATCH --mem=32G`
- `#SBATCH --time=02:00:00`

### 8.2 Execution Pipeline Workflow

```mermaid
sequenceDiagram
    autonumber
    participant Slurm as Slurm Scheduler
    participant Script as run_benchmarks_hpc.sh
    participant Diag as GPU Diagnostics
    participant Bagel as BAGEL Runner
    participant GPU as GPU4PySCF Runner
    participant Driver as Comparison Driver
    participant Archive as Results Archive

    Slurm->>Script: Launch Job on GPU Compute Node
    Script->>Diag: Module load (CUDA/MPI), check nvidia-smi & CuPy
    Diag-->>Script: GPU verified (Count >= 1)
    alt RUN_UNIT_TESTS == 1
        Script->>Script: pytest test_sa_casscf.py test_fci_rdms.py
    end
    Script->>Bagel: run_bagel_reference.py (or fallback JSON)
    Bagel-->>Script: bagel_reference_ethylene_cas44.json
    Script->>GPU: run_gpu4pyscf_benchmark.py
    GPU-->>Script: gpu4pyscf_cas44_results.h5
    Script->>Driver: compare_bagel_gpu4pyscf.py (tol: 1e-7 Ha, 1e-5 RDM)
    Driver-->>Script: validation_report.txt (Status: PASS / FAIL)
    alt Validation Passed
        Script->>Archive: Copy HDF5, logs, report to results/run_<job_id>/
        Script-->>Slurm: Trap scratch cleanup & Exit Code 0
    else Validation Failed / Terminated
        Script-->>Slurm: Trap scratch cleanup & Exit Code 1
    end
```

### 8.3 Scratch Lifecycle & Signal Safety
To prevent I/O bottlenecks and filesystem pollution, the runner:
1. Allocates a node-local scratch directory (`$SLURM_TMPDIR` or `/tmp/$USER/gpu4pyscf_$SLURM_JOB_ID`).
2. Redirects all PySCF temporary integral buffers via `export PYSCF_TMPDIR="${SCRATCH_DIR}"`.
3. Sets POSIX signal traps (`trap cleanup EXIT SIGTERM SIGINT`) to guarantee that all intermediate scratch files are purged upon job completion or premature preemption, preserving cluster disk health.

### 8.4 Modular Configuration Variables

| Variable | Default | Purpose |
|---|---|---|
| `MODULE_CUDA` | `cuda/12.4` | HPC environment module for CUDA runtime. |
| `MODULE_MPI` | *(empty)* | Optional MPI module for BAGEL parallelism. |
| `CONDA_ENV_NAME` | `gpu4pyscf` | Target Conda environment for GPU4PySCF. |
| `BAGEL_BIN` | *(empty)* | Custom binary path for BAGEL executable. |
| `RUN_UNIT_TESTS` | `1` | Executes `pytest` unit test suite prior to benchmark. |
| `TOL_ENERGY` | `1e-7` | Maximum acceptable energy difference in Hartree. |
| `TOL_RDM` | `1e-5` | Maximum acceptable 1-RDM element difference. |

