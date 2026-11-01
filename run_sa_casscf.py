#!/usr/bin/env python3
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Standalone test script for GPU-accelerated State-Averaged CASSCF (SA-CASSCF).

Performs density-fitted SA(3)-CASSCF(4,4) on ethylene (C2H4) in cc-pVDZ /
cc-pvdz-jkfit matching the BAGEL benchmark specification:
- Active space: CAS(4,4) (4 active electrons in 4 active orbitals)
- Closed-shell core: 6 orbitals (12 electrons)
- Roots: 3 singlet states (weights: 1/3, 1/3, 1/3)
- Spherical Gaussians (mol.cart = False)
"""

import argparse
import faulthandler
import json
import time
import numpy as np

faulthandler.enable()


ETHYLENE_GEOMETRY = """
C   0.00000000   0.00000000   0.66750000
C   0.00000000   0.00000000  -0.66750000
H   0.00000000   0.92300000   1.23250000
H   0.00000000  -0.92300000   1.23250000
H   0.00000000   0.92300000  -1.23250000
H   0.00000000  -0.92300000  -1.23250000
"""


def to_numpy(arr):
    if arr is None:
        return np.array([])
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def run_sa_casscf(
    basis="cc-pvdz",
    auxbasis="cc-pvdz-jkfit",
    ncas=4,
    nelecas=(2, 2),
    nstates=3,
    weights=None,
    conv_tol=1e-8,
    conv_tol_grad=1e-5,
    max_cycle=100,
    output_json="pyscf_sa_casscf_results.json",
):
    import pyscf
    from pyscf import scf
    from gpu4pyscf import mcscf

    if weights is None:
        weights = [1.0 / nstates] * nstates

    print("=" * 72)
    print("GPU4PySCF State-Averaged CASSCF (SA-CASSCF) Test")
    print("=" * 72)
    print(f"Molecule      : Ethylene (C2H4)")
    print(f"Basis Set     : {basis}")
    print(f"DF Auxiliary  : {auxbasis}")
    print(f"Active Space  : CAS({ncas}, {nelecas})")
    print(f"Target States : {nstates} roots, weights = {weights}")
    print(f"Convergence   : tol = {conv_tol}, tol_grad = {conv_tol_grad}")
    print("=" * 72)

    # 1. Molecule definition (spherical 5d to match BAGEL)
    mol = pyscf.M(
        atom=ETHYLENE_GEOMETRY,
        basis=basis,
        verbose=4,
    )
    mol.cart = False

    # 2. Density-fitted RHF on GPU
    t0_rhf = time.time()
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis).run()
    mf_gpu = mf.to_gpu()
    t_rhf = time.time() - t0_rhf
    print(f"\n>>> [1/3] GPU DF-RHF converged in {t_rhf:.2f} s")
    print(f"    RHF Total Energy: {mf_gpu.e_tot:.10f} Hartree")

    # 3. State-Averaged CASSCF on GPU
    t0_mc = time.time()
    mc = mcscf.DFCASSCF(mf_gpu, ncas=ncas, nelecas=nelecas, auxbasis=auxbasis)
    mc.state_average_(weights=weights)
    mc.conv_tol = conv_tol
    mc.conv_tol_grad = conv_tol_grad
    mc.max_cycle_macro = max_cycle

    mc.kernel()
    t_mc = time.time() - t0_mc
    print(f"\n>>> [2/3] GPU DF-SA-CASSCF completed in {t_mc:.2f} s")
    print(f"    Converged: {mc.converged}")
    print(f"    State-Averaged Energy: {mc.e_tot:.10f} Hartree")

    # 4. Extract individual state energies
    state_energies = [float(e) for e in mc.e_states]
    e_ref = state_energies[0]
    print("\n>>> [3/3] Electronic State Energies:")
    print("-" * 52)
    print(f"{'State':<10} {'Energy (Hartree)':<24} {'Excitation (eV)':<15}")
    print("-" * 52)
    for i, e in enumerate(state_energies):
        d_ev = (e - e_ref) * 27.211386245988
        print(f"State {i:<4} {e:<24.10f} {d_ev:<15.4f}")
    print("-" * 52)

    # 5. State-averaged active 1-RDM
    if hasattr(mc, "casdm1") and mc.casdm1 is not None:
        casdm1 = to_numpy(mc.casdm1)
    else:
        casdm1 = to_numpy(mc.fcisolver.make_rdm1(mc.ci, ncas, nelecas))
    print("\nState-Averaged Active 1-RDM Diagonal (Occupations):")
    for k in range(ncas):
        print(f"  Active orbital {k}: {casdm1[k, k]:.6f}")

    # 6. GPU Transition RDMs (using native trans_rdm1 on GPU)
    print("\nTransition 1-RDMs between roots (Frobenius norm):")
    trans_rdm_summary = {}
    for i in range(len(mc.ci)):
        for j in range(i + 1, len(mc.ci)):
            tdm1 = mc.fcisolver.trans_rdm1(mc.ci[i], mc.ci[j], ncas, nelecas)
            tdm1_np = to_numpy(tdm1)
            norm = float(np.linalg.norm(tdm1_np))
            trans_rdm_summary[f"{i}->{j}"] = norm
            print(f"  <S{i}| E_tu |S{j}> Frobenius Norm = {norm:.8f}")

    # 7. Save results to JSON for comparison with BAGEL
    results = {
        "molecule": "ethylene",
        "basis": basis,
        "auxbasis": auxbasis,
        "ncas": ncas,
        "nelecas": list(nelecas),
        "nstates": nstates,
        "weights": weights,
        "e_tot_sa": float(mc.e_tot),
        "state_energies": state_energies,
        "excitation_energies_ev": [(e - e_ref) * 27.211386245988 for e in state_energies],
        "casdm1_diag": [float(casdm1[k, k]) for k in range(ncas)],
        "casdm1": casdm1.tolist(),
        "trans_rdm1_norms": trans_rdm_summary,
        "timings": {
            "rhf_seconds": t_rhf,
            "mcscf_seconds": t_mc,
            "total_seconds": t_rhf + t_mc,
        },
        "converged": bool(mc.converged),
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to: {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run GPU4PySCF State-Averaged CASSCF test on ethylene."
    )
    parser.add_argument("--basis", default="cc-pvdz", help="AO basis (default: cc-pvdz)")
    parser.add_argument("--auxbasis", default="cc-pvdz-jkfit", help="DF auxbasis (default: cc-pvdz-jkfit)")
    parser.add_argument("--ncas", type=int, default=4, help="Active orbitals (default: 4)")
    parser.add_argument("--nstates", type=int, default=3, help="Number of states (default: 3)")
    parser.add_argument("--output", default="pyscf_sa_casscf_results.json", help="Output JSON filename")
    args = parser.parse_args()

    run_sa_casscf(
        basis=args.basis,
        auxbasis=args.auxbasis,
        ncas=args.ncas,
        nstates=args.nstates,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
