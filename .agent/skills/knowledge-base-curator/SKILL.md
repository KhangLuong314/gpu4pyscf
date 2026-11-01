---
name: knowledge-base-curator
description: Synchronizes, updates, and curates the project knowledge base in knowledge_base/ whenever theoretical derivations, mathematical formulations, GPU algorithmic acceleration strategies, software architecture, or implementation plans are discussed, clarified, or refined.
allowed-tools: Read Write Edit Bash
metadata:
  version: "1.0"
  last-reviewed: "2026-10-30"
  author: "Antigravity Pair Programmer"
---

# Knowledge Base Curator Skill

## Purpose

This skill maintains the `knowledge_base/` documentation system as a living, authoritative, publication-grade repository of theory, algorithms, software architecture, and implementation status for GPU-accelerated SA-CASSCF, XMS-CASPT2, NACVs, and NAMD photodynamics in `gpu4pyscf`.

Whenever theoretical concepts, mathematical formulations, GPU kernel optimizations, or architectural designs are discussed or clarified in conversation, this skill guides the systematic update of the corresponding knowledge base modules.

---

## Knowledge Base Map

All documentation resides in `/knowledge_base/`:

| Module | Primary Scope | Update Triggers |
| :--- | :--- | :--- |
| **`INDEX.md`** | Table of contents, architecture overview, mermaid flowcharts | New modules added, roadmap phase transitions |
| **`01_sa_casscf_and_nac_theory.md`** | SA-CASSCF functional, orbital stationarity, CP-MCSCF / Z-vector, analytical NACVs, ETF/CSF terms | Discussion of MCSCF gradients, response equations, coupling vectors, branching space |
| **`02_xms_caspt2_theory.md`** | Invariance breakdown, average Fock operator, model space rotation, 8 excitation classes, imaginary shifts | Discussions of CASPT2, multi-state dynamic correlation, intruder states, dynamic coupling |
| **`03_gpu_acceleration_strategies.md`** | GPU DF-AO2MO, CUDA raw kernels for FCI (`_build_t1`), multi-root Davidson, GPU Krylov solvers, VRAM tiling | CUDA kernel design, CuPy operations, memory management, stream pipelining |
| **`04_software_architecture_gpu4pyscf.md`** | Class hierarchy, module layout, API signatures, scanner contracts, CPU/GPU interop | Class design changes, new module structures, scanner interface definitions |
| **`05_namd_photochemistry_workflow.md`** | Tully FSSH algorithm, wave function phase tracking, momentum rescaling, MECI optimization | Non-adiabatic dynamics, trajectory algorithms, crossing point optimization |
| **`06_benchmarking_and_validation.md`** | Finite-difference protocols, cross-code comparisons (OpenMolcas/BAGEL), benchmark molecules | New validation tests, tolerance thresholds, benchmark systems |
| **`07_annotated_bibliography.md`** & **`references.bib`** | Literature citations, abstracts, algorithmic insights, BibTeX database | New literature cited, reference additions |
| **`08_codebase_gap_analysis_and_phase1_plan.md`** | Repository capability audit, gap analysis, milestone execution blueprints | Progress updates, newly implemented features, milestone completions |

---

## Operating Protocol: When Theory is Discussed

Whenever an interaction with the user involves theoretical explanations, mathematical derivations, or architectural decisions:

### Step 1: Identify What Was Clarified or Derived
Extract the key conceptual additions, for example:
- A mathematical derivation or identity (e.g. electronic Hessian blocks, Z-vector Lagrangian).
- A physical insight (e.g. why MS-CASPT2 breaks at conical intersections, momentum adjustment direction).
- A GPU implementation tactic (e.g. tiling auxiliary basis indices, avoiding device-host synchronization).
- A design contract (e.g. `NACGradients.kernel()` signature, scanner methods).

### Step 2: Locate the Relevant Knowledge Base Module(s)
Determine which file(s) in `knowledge_base/` should house the new information using the Knowledge Base Map above. If the topic spans multiple areas (e.g., theory + GPU algorithm), update both the theoretical file (e.g. `01_` or `02_`) and the implementation/algorithmic file (e.g. `03_` or `04_`).

### Step 3: Apply the Content Update
Follow these strict documentation standards:
1. **KaTeX Formatting:** Use inline `$ ... $` and display `$$ ... $$` math. Always escape literal dollar signs (`\$`).
2. **Mermaid Diagrams:** When workflows or state transitions are discussed, provide or update mermaid diagrams.
3. **Alert Callouts:** Emphasize critical pitfalls using GitHub alerts (`> [!IMPORTANT]`, `> [!NOTE]`, `> [!WARNING]`).
4. **Code Contracts:** Show exact Python/CuPy or CUDA kernel signatures where applicable.
5. **Cross-References:** Ensure internal markdown links between modules stay valid.

### Step 4: Keep `INDEX.md` and Status Synchronized
- If a new module is added, register it in `INDEX.md`.
- If an implementation milestone is reached or reprioritized, update `08_codebase_gap_analysis_and_phase1_plan.md`.
