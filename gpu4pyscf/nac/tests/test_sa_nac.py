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

try:
    import gpu4pyscf
    from gpu4pyscf import mcscf
    from gpu4pyscf.nac import sacasscf as sacasscf_nac
    has_gpu4pyscf = True
except (ImportError, Exception):
    has_gpu4pyscf = False


@unittest.skipIf(not has_gpu4pyscf, "gpu4pyscf is required for GPU NAC tests")
class TestSACASSCFNAC(unittest.TestCase):
    def setUp(self):
        self.mol = pyscf.M(
            atom="""
            H  0.0  0.0  0.0
            H  0.0  0.0  1.0
            """,
            basis='def2-svp',
            verbose=0
        )
        self.mf = scf.RHF(self.mol).density_fit(auxbasis='def2-svp-jkfit').run()
        self.mf_gpu = self.mf.to_gpu()

        self.mc = mcscf.DFCASSCF(self.mf_gpu, ncas=2, nelecas=(1, 1))
        self.mc.state_average_(weights=[0.5, 0.5])
        self.mc.conv_tol = 1e-10
        self.mc.conv_tol_grad = 1e-6
        self.mc.run()

    def test_diagonal_coupling_is_zero(self):
        """Verify that d_{00} is identically zero by anti-symmetry."""
        nac_obj = self.mc.NACGradients(state=(0, 0))
        d_00 = nac_obj.kernel()
        norm_00 = np.linalg.norm(d_00)
        self.assertEqual(norm_00, 0.0)

    def test_antisymmetry(self):
        """Verify that d_{10} = - d_{01}."""
        nac_01 = self.mc.NACGradients(state=(0, 1), use_etfs=True).kernel()
        nac_10 = self.mc.NACGradients(state=(1, 0), use_etfs=True).kernel()

        diff = np.max(np.abs(nac_01 + nac_10))
        self.assertLess(diff, 1e-10, f"Anti-symmetry violation: {diff}")

    def test_mult_ediff_scaling(self):
        """Verify that mult_ediff=True scales by ΔE: h_{01} = (E_1 - E_0) * d_{01}."""
        e_diff = self.mc.e_states[1] - self.mc.e_states[0]

        d_01 = self.mc.NACGradients(state=(0, 1), mult_ediff=False).kernel()
        h_01 = self.mc.NACGradients(state=(0, 1), mult_ediff=True).kernel()

        expected_h01 = d_01 * e_diff
        diff = np.max(np.abs(h_01 - expected_h01))
        self.assertLess(diff, 1e-10, f"Energy difference scaling violation: {diff}")

    def test_etf_option(self):
        """Verify that use_etfs can be toggled and executes cleanly."""
        nac_etf = self.mc.NACGradients(state=(0, 1), use_etfs=True).kernel()
        nac_raw = self.mc.NACGradients(state=(0, 1), use_etfs=False).kernel()

        self.assertEqual(nac_etf.shape, (self.mol.natm, 3))
        self.assertEqual(nac_raw.shape, (self.mol.natm, 3))


if __name__ == '__main__':
    unittest.main()
