# Copyright 2021-2026 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy as np
import pyscf
from pyscf import scf
import cupy as cp

try:
    import gpu4pyscf
    from gpu4pyscf import mcscf
    from gpu4pyscf.grad import sacasscf as sacasscf_grad
    has_gpu4pyscf = True
except (ImportError, Exception):
    has_gpu4pyscf = False


def _finite_difference_grad(mol, ncas=2, nelecas=(1, 1), weights=(0.5, 0.5), state=None, delta=1e-3):
    """Compute numerical nuclear gradient via centered finite differences."""
    coords = mol.atom_coords().copy()
    natm = mol.natm
    g_num = np.zeros((natm, 3))

    for ia in range(natm):
        for ix in range(3):
            # + delta
            c_p = coords.copy()
            c_p[ia, ix] += delta
            mol_p = mol.copy()
            mol_p.set_geom_(c_p, unit='Bohr')
            mol_p.build()
            mf_p = scf.RHF(mol_p).density_fit(auxbasis='def2-svp-jkfit').run(verbose=0)
            mc_p = mcscf.DFCASSCF(mf_p.to_gpu(), ncas=ncas, nelecas=nelecas)
            mc_p.state_average_(weights=weights)
            mc_p.run(verbose=0)
            if state is None:
                e_p = float(mc_p.e_tot)
            else:
                e_p = float(mc_p.e_states[int(state)])

            # - delta
            c_m = coords.copy()
            c_m[ia, ix] -= delta
            mol_m = mol.copy()
            mol_m.set_geom_(c_m, unit='Bohr')
            mol_m.build()
            mf_m = scf.RHF(mol_m).density_fit(auxbasis='def2-svp-jkfit').run(verbose=0)
            mc_m = mcscf.DFCASSCF(mf_m.to_gpu(), ncas=ncas, nelecas=nelecas)
            mc_m.state_average_(weights=weights)
            mc_m.run(verbose=0)
            if state is None:
                e_m = float(mc_m.e_tot)
            else:
                e_m = float(mc_m.e_states[int(state)])

            g_num[ia, ix] = (e_p - e_m) / (2.0 * delta)

    return g_num


@unittest.skipIf(not has_gpu4pyscf, "gpu4pyscf is required for GPU gradient tests")
class TestSACASSCFGradients(unittest.TestCase):
    def setUp(self):
        self.mol = pyscf.M(
            atom="""
            H  0.0  0.0  0.0
            H  0.0  0.0  0.9
            """,
            basis='def2-svp',
            verbose=0
        )
        self.mf = scf.RHF(self.mol).density_fit(auxbasis='def2-svp-jkfit').run()
        self.mf_gpu = self.mf.to_gpu()

    def test_state_averaged_gradient(self):
        """Verify analytical state-averaged gradient matches finite difference."""
        mc = mcscf.DFCASSCF(self.mf_gpu, ncas=2, nelecas=(1, 1))
        mc.state_average_(weights=[0.5, 0.5])
        mc.conv_tol = 1e-10
        mc.conv_tol_grad = 1e-6
        mc.run()

        # Analytical gradient (state=None)
        g_scanner = mc.nuc_grad_method(state=None)
        g_ana = g_scanner.kernel()

        # Numerical gradient
        g_num = _finite_difference_grad(self.mol, ncas=2, nelecas=(1, 1),
                                       weights=[0.5, 0.5], state=None, delta=1e-3)

        diff = np.max(np.abs(g_ana - g_num))
        self.assertLess(diff, 5e-4, f"Max gradient deviation too large: {diff}")

    def test_state_specific_gradient(self):
        """Verify analytical state-specific gradient with CP-MCSCF Z-vector response."""
        mc = mcscf.DFCASSCF(self.mf_gpu, ncas=2, nelecas=(1, 1))
        mc.state_average_(weights=[0.6, 0.4])
        mc.conv_tol = 1e-10
        mc.conv_tol_grad = 1e-6
        mc.run()

        # Analytical gradient of ground root (state=0)
        g0_scanner = mc.nuc_grad_method(state=0)
        g0_ana = g0_scanner.kernel()

        g0_num = _finite_difference_grad(self.mol, ncas=2, nelecas=(1, 1),
                                         weights=[0.6, 0.4], state=0, delta=1e-3)

        diff0 = np.max(np.abs(g0_ana - g0_num))
        self.assertLess(diff0, 1e-3, f"Root 0 gradient deviation too large: {diff0}")

        # Analytical gradient of excited root (state=1)
        g1_scanner = mc.nuc_grad_method(state=1)
        g1_ana = g1_scanner.kernel()

        g1_num = _finite_difference_grad(self.mol, ncas=2, nelecas=(1, 1),
                                         weights=[0.6, 0.4], state=1, delta=1e-3)

        diff1 = np.max(np.abs(g1_ana - g1_num))
        self.assertLess(diff1, 1e-3, f"Root 1 gradient deviation too large: {diff1}")


if __name__ == '__main__':
    unittest.main()
