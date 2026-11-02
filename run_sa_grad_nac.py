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
Standalone benchmark and test runner for Phase 2:
GPU SA-CASSCF Analytical Nuclear Gradients and Non-Adiabatic Coupling Vectors (NACVs).

System: Ethylene (C2H4)
Basis : cc-pVDZ / cc-pvdz-jkfit
Active: CAS(4,4), 3 states (weights: 1/3, 1/3, 1/3)

Evaluates:
1. State-averaged energy analytical gradient.
2. State-specific analytical gradients for Roots 0, 1, 2 via CP-MCSCF Z-vector.
3. Analytical Non-Adiabatic Coupling Vectors (NACVs) d_IJ and h_IJ across all root pairs.
4. Timings and performance metrics on GPU.
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


def run_phase2_benchmark(
    basis="cc-pvdz",
    auxbasis="cc-pvdz-jkfit",
    ncas=4,
    nelecas=(2, 2),
    nstates=3,
    output_json="pyscf_sa_grad_nac_results.json",
):
    import pyscf
    from pyscf import scf
    from gpu4pyscf import mcscf
    from gpu4pyscf.grad import sacasscf as sacasscf_grad
    from gpu4pyscf.nac import sacasscf as sacasscf_nac

    weights = [1.0 / nstates] * nstates

    print("=" * 76)
    print("GPU4PySCF Phase 2: SA-CASSCF Analytical Gradients & NACVs Benchmark")
    print("=" * 76)
    print(f"Molecule      : Ethylene (C2H4)")
    print(f"Basis Set     : {basis}")
    print(f"DF Auxiliary  : {auxbasis}")
    print(f"Active Space  : CAS({ncas}, {nelecas})")
    print(f"Target States : {nstates} roots, weights = {weights}")
    print("=" * 76)

    # 1. Molecule definition
    mol = pyscf.M(
        atom=ETHYLENE_GEOMETRY,
        basis=basis,
        verbose=3,
    )
    mol.cart = False

    # 2. Density-fitted RHF on GPU
    t0_rhf = time.time()
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis).run()
    mf_gpu = mf.to_gpu()
    t_rhf = time.time() - t0_rhf
    print(f"\n[1/4] GPU DF-RHF converged in {t_rhf:.2f} s (E = {mf_gpu.e_tot:.10f} Ha)")

    # 3. State-Averaged CASSCF on GPU
    t0_mc = time.time()
    mc = mcscf.DFCASSCF(mf_gpu, ncas=ncas, nelecas=nelecas, auxbasis=auxbasis)
    mc.state_average_(weights=weights)
    mc.conv_tol = 1e-8
    mc.conv_tol_grad = 1e-5
    mc.kernel()
    t_mc = time.time() - t0_mc
    print(f"\n[2/4] GPU DF-SA-CASSCF converged in {t_mc:.2f} s")
    print(f"      SA Total Energy = {mc.e_tot:.10f} Ha")
    for i, e in enumerate(mc.e_states):
        print(f"      Root {i}: {float(e):.10f} Ha (ΔE = {(float(e) - float(mc.e_states[0])) * 27.211386:.4f} eV)")

    # 4. Analytical Nuclear Gradients (State-Averaged & State-Specific)
    print("\n[3/4] Evaluating Analytical Nuclear Gradients on GPU:")
    print("-" * 76)

    grad_timings = {}
    gradients_dict = {}

    # State-averaged gradient
    t0_gsa = time.time()
    g_sa_scanner = mc.nuc_grad_method(state=None)
    g_sa = g_sa_scanner.kernel()
    t_gsa = time.time() - t0_gsa
    grad_timings['state_averaged'] = t_gsa
    gradients_dict['state_averaged'] = to_numpy(g_sa).tolist()
    norm_gsa = float(np.linalg.norm(g_sa))
    print(f"  ∇ E_SA (State-Averaged) : Computed in {t_gsa:.3f} s, ||∇ E||_F = {norm_gsa:.8f} Ha/Bohr")

    # State-specific gradients for each root
    for k in range(nstates):
        t0_gk = time.time()
        gk_scanner = mc.nuc_grad_method(state=k)
        gk = gk_scanner.kernel()
        t_gk = time.time() - t0_gk
        grad_timings[f'state_{k}'] = t_gk
        gradients_dict[f'state_{k}'] = to_numpy(gk).tolist()
        norm_gk = float(np.linalg.norm(gk))
        print(f"  ∇ E_{k}  (Root {k})          : Computed in {t_gk:.3f} s, ||∇ E||_F = {norm_gk:.8f} Ha/Bohr")

    # 5. Non-Adiabatic Coupling Vectors (NACVs) between all pairs
    print("\n[4/4] Evaluating Non-Adiabatic Coupling Vectors (NACVs) on GPU:")
    print("-" * 76)

    nac_timings = {}
    nacs_dict = {}

    for i in range(nstates):
        for j in range(i + 1, nstates):
            pair_key = f"{i}->{j}"

            # Standard NACV d_IJ (with ETFs for NAMD)
            t0_dij = time.time()
            nac_obj = mc.NACGradients(state=(i, j), use_etfs=True, mult_ediff=False)
            d_ij = nac_obj.kernel()
            t_dij = time.time() - t0_dij
            norm_dij = float(np.linalg.norm(d_ij))

            # Scaled coupling h_IJ = ΔE * d_IJ
            h_obj = mc.NACGradients(state=(i, j), use_etfs=True, mult_ediff=True)
            h_ij = h_obj.kernel()
            norm_hij = float(np.linalg.norm(h_ij))

            # Raw basis coupling (without ETFs)
            raw_obj = mc.NACGradients(state=(i, j), use_etfs=False, mult_ediff=False)
            raw_dij = raw_obj.kernel()
            norm_raw = float(np.linalg.norm(raw_dij))

            nac_timings[pair_key] = t_dij
            nacs_dict[pair_key] = {
                "d_ij_norm": norm_dij,
                "h_ij_norm": norm_hij,
                "raw_d_ij_norm": norm_raw,
                "d_ij": to_numpy(d_ij).tolist(),
                "h_ij": to_numpy(h_ij).tolist(),
            }

            print(f"  NAC Pair ({i}, {j}):")
            print(f"    d_{i}{j} (with ETFs) : Time = {t_dij:.3f} s, ||d||_F = {norm_dij:.8f} Bohr^-1")
            print(f"    h_{i}{j} (ΔE * d)    : ||h||_F = {norm_hij:.8f} Ha/Bohr")
            print(f"    Raw (no ETFs)    : ||d_raw||_F = {norm_raw:.8f} Bohr^-1")

    # 6. Save comprehensive results
    total_time = t_rhf + t_mc + sum(grad_timings.values()) + sum(nac_timings.values())
    results = {
        "molecule": "ethylene",
        "basis": basis,
        "auxbasis": auxbasis,
        "ncas": ncas,
        "nelecas": list(nelecas),
        "nstates": nstates,
        "e_tot_sa": float(mc.e_tot),
        "state_energies": [float(e) for e in mc.e_states],
        "grad_timings_seconds": grad_timings,
        "gradients": gradients_dict,
        "nac_timings_seconds": nac_timings,
        "nacs": nacs_dict,
        "timings": {
            "rhf_seconds": t_rhf,
            "mcscf_seconds": t_mc,
            "gradients_total_seconds": sum(grad_timings.values()),
            "nac_total_seconds": sum(nac_timings.values()),
            "total_benchmark_seconds": total_time,
        },
        "success": True,
    }

    if output_json:
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Phase 2 Benchmark Results saved to: {output_json}")
        print(f"     Total End-to-End Runtime: {total_time:.2f} s on GPU")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run GPU4PySCF Phase 2 Gradients & NACVs Benchmark.")
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--auxbasis", default="cc-pvdz-jkfit")
    parser.add_argument("--ncas", type=int, default=4)
    parser.add_argument("--nstates", type=int, default=3)
    parser.add_argument("--output", default="pyscf_sa_grad_nac_results.json")
    args = parser.parse_args()

    run_phase2_benchmark(
        basis=args.basis,
        auxbasis=args.auxbasis,
        ncas=args.ncas,
        nstates=args.nstates,
        output_json=args.output,
    )


if __name__ == "__main__":
    main()
