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

import unittest
import numpy as np

import pyscf
from pyscf import lib
from pyscf.fci import direct_spin1 as cpu_direct_spin1
from pyscf.fci import cistring

try:
    import cupy as cp
    cuda_available = cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    cuda_available = False

try:
    from gpu4pyscf.fci import direct_spin1 as gpu_direct_spin1
    from gpu4pyscf.fci.direct_spin1 import FCISolver as GPUFCISolver
except Exception:
    import types, sys
    gpu_direct_spin1 = None
    GPUFCISolver = None
    if 'gpu4pyscf' not in sys.modules:
        sys.modules['gpu4pyscf'] = types.ModuleType('gpu4pyscf')
    if 'gpu4pyscf.fci' not in sys.modules:
        sys.modules['gpu4pyscf.fci'] = types.ModuleType('gpu4pyscf.fci')
    if 'gpu4pyscf.fci.direct_spin1' not in sys.modules:
        sys.modules['gpu4pyscf.fci.direct_spin1'] = types.ModuleType('gpu4pyscf.fci.direct_spin1')


def _make_orthogonal_roots(norb, nelec, nroots=3, seed=42):
    '''Generate orthogonal multi-root CI vectors by diagonalizing a symmetric Hamiltonian.'''
    rng = np.random.RandomState(seed)
    neleca, nelecb = cpu_direct_spin1._unpack_nelec(nelec)
    na = lib.comb(norb, neleca)
    nb = lib.comb(norb, nelecb)
    h1e = rng.randn(norb, norb)
    h1e = (h1e + h1e.T) * 0.5
    eri = rng.randn(norb, norb, norb, norb) * 0.1
    eri = (eri + eri.transpose(1, 0, 3, 2)) * 0.5
    eri = (eri + eri.transpose(2, 3, 0, 1)) * 0.5

    solver = cpu_direct_spin1.FCISolver()
    _, ci_roots = solver.kernel(h1e, eri, norb, nelec, nroots=nroots)
    if not isinstance(ci_roots, (list, tuple)):
        ci_roots = [ci_roots]
    # Normalize each root
    ci_roots = [c / np.linalg.norm(c) for c in ci_roots]
    return ci_roots


class TestFCIRDMHostSanity(unittest.TestCase):
    '''Host-executable tests verifying reference symmetries and API definitions.'''

    def test_cpu_reference_symmetries_cas44(self):
        norb = 4
        nelec = (2, 2)
        ci_roots = _make_orthogonal_roots(norb, nelec, nroots=2, seed=123)
        ci0, ci1 = ci_roots[0], ci_roots[1]

        dm1_00 = cpu_direct_spin1.trans_rdm1(ci0, ci0, norb, nelec)
        dm1_11 = cpu_direct_spin1.trans_rdm1(ci1, ci1, norb, nelec)
        dm1_01 = cpu_direct_spin1.trans_rdm1(ci0, ci1, norb, nelec)
        dm1_10 = cpu_direct_spin1.trans_rdm1(ci1, ci0, norb, nelec)

        # Trace relations: Tr(gamma^IJ) = Ne * delta_IJ
        ne_tot = sum(nelec)
        self.assertAlmostEqual(np.trace(dm1_00), ne_tot, places=10)
        self.assertAlmostEqual(np.trace(dm1_11), ne_tot, places=10)
        self.assertAlmostEqual(np.trace(dm1_01), 0.0, places=10)
        self.assertAlmostEqual(np.trace(dm1_10), 0.0, places=10)

        # Hermiticity / bra-ket exchange: gamma_pq^01 = gamma_qp^10
        np.testing.assert_allclose(dm1_01, dm1_10.T, atol=1e-12)

        dm1_01_, dm2_01 = cpu_direct_spin1.trans_rdm12(ci0, ci1, norb, nelec, reorder=True)
        _, dm2_10 = cpu_direct_spin1.trans_rdm12(ci1, ci0, norb, nelec, reorder=True)
        np.testing.assert_allclose(dm1_01, dm1_01_, atol=1e-12)

        # Particle permutation symmetry: Gamma_pqrs^IJ = Gamma_rspq^IJ
        np.testing.assert_allclose(dm2_01, dm2_01.transpose(2, 3, 0, 1), atol=1e-12)

        # Bra-ket exchange symmetry: Gamma_pqrs^01 = Gamma_qpsr^10
        np.testing.assert_allclose(dm2_01, dm2_10.transpose(1, 0, 3, 2), atol=1e-12)

        # Contraction relation: sum_r Gamma_pqrr^IJ = (Ne - 1) * gamma_pq^IJ
        contracted_2rdm = np.einsum('pqrr->pq', dm2_01)
        np.testing.assert_allclose(contracted_2rdm, (ne_tot - 1) * dm1_01, atol=1e-12)

    def test_algebraic_simulation_cas44_and_asymmetric(self):
        '''Simulate the tile loop and dot products on CPU to verify algebraic equivalence to <1e-14.'''
        test_cases = [
            (4, (2, 2)),
            (5, (3, 1)),
        ]
        TILE = 32
        for norb, nelec in test_cases:
            with self.subTest(norb=norb, nelec=nelec):
                neleca, nelecb = nelec
                na = lib.comb(norb, neleca)
                nb = lib.comb(norb, nelecb)
                nnorb = norb * norb

                link_indexa = cistring.gen_linkstr_index(range(norb), neleca)
                link_indexb = cistring.gen_linkstr_index(range(norb), nelecb)

                def convert(link):
                    full_link = link.copy()
                    full_link[:, :, 0] = link[:, :, 1] * norb + link[:, :, 0]
                    nlink = full_link.shape[1]
                    addr = np.zeros((nnorb, full_link.shape[0]), dtype=np.uint32)
                    sign = np.zeros((nnorb, full_link.shape[0]), dtype=np.int8)
                    for i in range(full_link.shape[0]):
                        for k in range(nlink):
                            pair = full_link[i, k, 0]
                            addr[pair, i] = full_link[i, k, 2]
                            sign[pair, i] = full_link[i, k, 3]
                    return addr, sign

                addra, signa = convert(link_indexa)
                addrb, signb = convert(link_indexb)

                ci_roots = _make_orthogonal_roots(norb, nelec, nroots=2, seed=777)
                ci0, ci1 = ci_roots[0], ci_roots[1]

                def sim_build_t1(ci, strb0):
                    t1 = np.zeros((nnorb, na * TILE))
                    for stra in range(na):
                        for tx in range(TILE):
                            strb = strb0 + tx
                            if strb < nb:
                                for pair in range(nnorb):
                                    val = 0.0
                                    sa = signa[pair, stra]
                                    if sa != 0:
                                        a1 = addra[pair, stra]
                                        val += sa * ci[a1, strb]
                                    sb = signb[pair, strb]
                                    if sb != 0:
                                        b1 = addrb[pair, strb]
                                        val += sb * ci[stra, b1]
                                    t1[pair, stra * TILE + tx] = val
                    return t1

                dm1 = np.zeros(nnorb)
                dm2 = np.zeros((nnorb, nnorb))
                ci_tile_bra = np.zeros((na, TILE))

                for strb0 in range(0, nb, TILE):
                    t1_ket = sim_build_t1(ci1, strb0)
                    t1_bra = sim_build_t1(ci0, strb0)
                    blen = min(TILE, nb - strb0)
                    ci_tile_bra.fill(0.)
                    ci_tile_bra[:, :blen] = ci0[:, strb0:strb0 + blen]
                    dm1 += np.dot(t1_ket, ci_tile_bra.ravel())
                    dm2 += np.dot(t1_bra, t1_ket.T)

                dm1 = dm1.reshape(norb, norb).T
                dm2 = dm2.reshape(norb, norb, norb, norb).transpose(1, 0, 2, 3)
                for k in range(norb):
                    dm2[:, k, k, :] -= dm1.T
                dm2 = dm2.reshape(nnorb, nnorb)
                dm2 = (dm2 + dm2.T) * 0.5
                dm2 = dm2.reshape(norb, norb, norb, norb)

                ref_dm1, ref_dm2 = cpu_direct_spin1.trans_rdm12(ci0, ci1, norb, nelec, reorder=True)
                np.testing.assert_allclose(dm1, ref_dm1, atol=1e-12)
                np.testing.assert_allclose(dm2, ref_dm2, atol=1e-12)


@unittest.skipUnless(cuda_available and gpu_direct_spin1 is not None,
                     "CUDA and CuPy required for GPU FCI RDM tests")
class TestFCITransitionRDMsGPU(unittest.TestCase):
    '''Comprehensive test suite for GPU transition RDMs in gpu4pyscf.fci.direct_spin1.'''

    def setUp(self):
        cp.random.seed(42)

    def test_cas44_symmetric_multi_root_rdm1_rdm12(self):
        '''Test CAS(4,4) multi-root orthogonal CI vectors against CPU PySCF reference.'''
        norb = 4
        nelec = (2, 2)
        ci_roots_cpu = _make_orthogonal_roots(norb, nelec, nroots=3, seed=101)
        ci_roots_gpu = [cp.asarray(c) for c in ci_roots_cpu]

        # Test all bra-ket pairs among roots 0, 1, 2
        test_pairs = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (2, 2)]
        for i, j in test_pairs:
            with self.subTest(root_bra=i, root_ket=j):
                cibra_cpu = ci_roots_cpu[i]
                ciket_cpu = ci_roots_cpu[j]
                cibra_gpu = ci_roots_gpu[i]
                ciket_gpu = ci_roots_gpu[j]

                # 1. trans_rdm1
                ref_dm1 = cpu_direct_spin1.trans_rdm1(cibra_cpu, ciket_cpu, norb, nelec)
                gpu_dm1 = gpu_direct_spin1.trans_rdm1(cibra_gpu, ciket_gpu, norb, nelec)
                self.assertIsInstance(gpu_dm1, cp.ndarray)
                np.testing.assert_allclose(gpu_dm1.get(), ref_dm1, atol=1e-10)

                # 2. trans_rdm12 with reorder=True
                ref_dm1_, ref_dm2 = cpu_direct_spin1.trans_rdm12(
                    cibra_cpu, ciket_cpu, norb, nelec, reorder=True)
                gpu_dm1_, gpu_dm2 = gpu_direct_spin1.trans_rdm12(
                    cibra_gpu, ciket_gpu, norb, nelec, reorder=True)
                self.assertIsInstance(gpu_dm1_, cp.ndarray)
                self.assertIsInstance(gpu_dm2, cp.ndarray)
                np.testing.assert_allclose(gpu_dm1_.get(), ref_dm1_, atol=1e-10)
                np.testing.assert_allclose(gpu_dm2.get(), ref_dm2, atol=1e-10)

                # 3. trans_rdm12 with reorder=False
                ref_dm1_raw, ref_dm2_raw = cpu_direct_spin1.trans_rdm12(
                    cibra_cpu, ciket_cpu, norb, nelec, reorder=False)
                gpu_dm1_raw, gpu_dm2_raw = gpu_direct_spin1.trans_rdm12(
                    cibra_gpu, ciket_gpu, norb, nelec, reorder=False)
                np.testing.assert_allclose(gpu_dm1_raw.get(), ref_dm1_raw, atol=1e-10)
                np.testing.assert_allclose(gpu_dm2_raw.get(), ref_dm2_raw, atol=1e-10)

    def test_asymmetric_electron_configurations(self):
        '''Test systems with neleca != nelecb (open-shell / spin-polarized sectors).'''
        test_configs = [
            (5, (3, 1)),  # norb=5, neleca=3, nelecb=1 (Ne=4)
            (4, (2, 1)),  # norb=4, neleca=2, nelecb=1 (Ne=3)
            (4, (3, 0)),  # norb=4, neleca=3, nelecb=0 (high-spin quartet)
        ]

        for norb, nelec in test_configs:
            with self.subTest(norb=norb, nelec=nelec):
                ci_roots_cpu = _make_orthogonal_roots(norb, nelec, nroots=2, seed=202)
                ci_bra_cpu, ci_ket_cpu = ci_roots_cpu[0], ci_roots_cpu[1]
                ci_bra_gpu = cp.asarray(ci_bra_cpu)
                ci_ket_gpu = cp.asarray(ci_ket_cpu)

                # Transition RDMs between distinct roots
                ref_dm1 = cpu_direct_spin1.trans_rdm1(ci_bra_cpu, ci_ket_cpu, norb, nelec)
                gpu_dm1 = gpu_direct_spin1.trans_rdm1(ci_bra_gpu, ci_ket_gpu, norb, nelec)
                np.testing.assert_allclose(gpu_dm1.get(), ref_dm1, atol=1e-10)

                ref_dm1_, ref_dm2 = cpu_direct_spin1.trans_rdm12(
                    ci_bra_cpu, ci_ket_cpu, norb, nelec, reorder=True)
                gpu_dm1_, gpu_dm2 = gpu_direct_spin1.trans_rdm12(
                    ci_bra_gpu, ci_ket_gpu, norb, nelec, reorder=True)
                np.testing.assert_allclose(gpu_dm1_.get(), ref_dm1_, atol=1e-10)
                np.testing.assert_allclose(gpu_dm2.get(), ref_dm2, atol=1e-10)

                # State RDMs on diagonal
                ref_diag_dm1, ref_diag_dm2 = cpu_direct_spin1.make_rdm12(
                    ci_bra_cpu, norb, nelec, reorder=True)
                gpu_diag_dm1, gpu_diag_dm2 = gpu_direct_spin1.make_rdm12(
                    ci_bra_gpu, norb, nelec, reorder=True)
                np.testing.assert_allclose(gpu_diag_dm1.get(), ref_diag_dm1, atol=1e-10)
                np.testing.assert_allclose(gpu_diag_dm2.get(), ref_diag_dm2, atol=1e-10)

    def test_mathematical_symmetries_and_trace_relations(self):
        '''Verify exact quantum-mechanical symmetry relations on GPU.'''
        norb = 4
        nelec = (2, 2)
        ne_tot = sum(nelec)
        ci_roots_cpu = _make_orthogonal_roots(norb, nelec, nroots=2, seed=303)
        ci0_gpu = cp.asarray(ci_roots_cpu[0])
        ci1_gpu = cp.asarray(ci_roots_cpu[1])

        dm1_00 = gpu_direct_spin1.trans_rdm1(ci0_gpu, ci0_gpu, norb, nelec).get()
        dm1_11 = gpu_direct_spin1.trans_rdm1(ci1_gpu, ci1_gpu, norb, nelec).get()
        dm1_01 = gpu_direct_spin1.trans_rdm1(ci0_gpu, ci1_gpu, norb, nelec).get()
        dm1_10 = gpu_direct_spin1.trans_rdm1(ci1_gpu, ci0_gpu, norb, nelec).get()

        # 1. Trace relation: Tr(gamma^IJ) = Ne * delta_IJ
        self.assertAlmostEqual(float(np.trace(dm1_00)), ne_tot, places=10)
        self.assertAlmostEqual(float(np.trace(dm1_11)), ne_tot, places=10)
        self.assertAlmostEqual(float(np.trace(dm1_01)), 0.0, places=10)
        self.assertAlmostEqual(float(np.trace(dm1_10)), 0.0, places=10)

        # 2. 1-RDM Hermiticity / bra-ket exchange: gamma_pq^01 = gamma_qp^10
        np.testing.assert_allclose(dm1_01, dm1_10.T, atol=1e-12)

        _, dm2_00 = gpu_direct_spin1.trans_rdm12(ci0_gpu, ci0_gpu, norb, nelec, reorder=True)
        _, dm2_01 = gpu_direct_spin1.trans_rdm12(ci0_gpu, ci1_gpu, norb, nelec, reorder=True)
        _, dm2_10 = gpu_direct_spin1.trans_rdm12(ci1_gpu, ci0_gpu, norb, nelec, reorder=True)
        dm2_00 = dm2_00.get()
        dm2_01 = dm2_01.get()
        dm2_10 = dm2_10.get()

        # 3. Particle permutation symmetry: Gamma_pqrs^IJ = Gamma_rspq^IJ
        np.testing.assert_allclose(dm2_00, dm2_00.transpose(2, 3, 0, 1), atol=1e-12)
        np.testing.assert_allclose(dm2_01, dm2_01.transpose(2, 3, 0, 1), atol=1e-12)

        # 4. Bra-ket exchange symmetry: Gamma_pqrs^01 = Gamma_qpsr^10
        np.testing.assert_allclose(dm2_01, dm2_10.transpose(1, 0, 3, 2), atol=1e-12)

        # 5. Contraction relation: sum_r Gamma_pqrr^IJ = (Ne - 1) * gamma_pq^IJ
        np.testing.assert_allclose(np.einsum('pqrr->pq', dm2_00), (ne_tot - 1) * dm1_00, atol=1e-12)
        np.testing.assert_allclose(np.einsum('pqrr->pq', dm2_01), (ne_tot - 1) * dm1_01, atol=1e-12)

        # 6. Total trace of 2-RDM: sum_{pr} Gamma_pprr^IJ = Ne * (Ne - 1) * delta_IJ
        self.assertAlmostEqual(float(np.einsum('pprr->', dm2_00)), ne_tot * (ne_tot - 1), places=10)
        self.assertAlmostEqual(float(np.einsum('pprr->', dm2_01)), 0.0, places=10)

    def test_diagonal_consistency(self):
        '''Verify that trans_rdm12(ci, ci) matches make_rdm12(ci) to exact precision.'''
        norb = 4
        nelec = (2, 2)
        ci_cpu = _make_orthogonal_roots(norb, nelec, nroots=1, seed=404)[0]
        ci_gpu = cp.asarray(ci_cpu)

        t_dm1 = gpu_direct_spin1.trans_rdm1(ci_gpu, ci_gpu, norb, nelec)
        m_dm1 = gpu_direct_spin1.make_rdm1(ci_gpu, norb, nelec)
        np.testing.assert_allclose(t_dm1.get(), m_dm1.get(), atol=1e-14)

        t_dm1_, t_dm2 = gpu_direct_spin1.trans_rdm12(ci_gpu, ci_gpu, norb, nelec)
        m_dm1_, m_dm2 = gpu_direct_spin1.make_rdm12(ci_gpu, norb, nelec)
        np.testing.assert_allclose(t_dm1_.get(), m_dm1_.get(), atol=1e-14)
        np.testing.assert_allclose(t_dm2.get(), m_dm2.get(), atol=1e-14)

    def test_fcisolver_class_interface(self):
        '''Verify FCISolver class methods return NumPy arrays and match module functions.'''
        norb = 4
        nelec = (2, 2)
        ci_roots_cpu = _make_orthogonal_roots(norb, nelec, nroots=2, seed=505)
        ci0_gpu = cp.asarray(ci_roots_cpu[0])
        ci1_gpu = cp.asarray(ci_roots_cpu[1])

        solver = GPUFCISolver()
        solver.norb = norb
        solver.nelec = nelec

        # Test solver methods with default norb/nelec
        dm1 = solver.trans_rdm1(ci0_gpu, ci1_gpu)
        self.assertIsInstance(dm1, np.ndarray)
        gpu_dm1 = gpu_direct_spin1.trans_rdm1(ci0_gpu, ci1_gpu, norb, nelec).get()
        np.testing.assert_allclose(dm1, gpu_dm1, atol=1e-14)

        dm1_, dm2 = solver.trans_rdm12(ci0_gpu, ci1_gpu)
        self.assertIsInstance(dm1_, np.ndarray)
        self.assertIsInstance(dm2, np.ndarray)
        gpu_dm1_, gpu_dm2 = gpu_direct_spin1.trans_rdm12(ci0_gpu, ci1_gpu, norb, nelec)
        np.testing.assert_allclose(dm1_, gpu_dm1_.get(), atol=1e-14)
        np.testing.assert_allclose(dm2, gpu_dm2.get(), atol=1e-14)

        # Test make_rdm1 and make_rdm12 on solver
        s_dm1 = solver.make_rdm1(ci0_gpu)
        self.assertIsInstance(s_dm1, np.ndarray)
        np.testing.assert_allclose(s_dm1, gpu_direct_spin1.make_rdm1(ci0_gpu, norb, nelec).get(), atol=1e-14)

        s_dm1_, s_dm2 = solver.make_rdm12(ci0_gpu)
        self.assertIsInstance(s_dm1_, np.ndarray)
        self.assertIsInstance(s_dm2, np.ndarray)
        m_dm1, m_dm2 = gpu_direct_spin1.make_rdm12(ci0_gpu, norb, nelec)
        np.testing.assert_allclose(s_dm1_, m_dm1.get(), atol=1e-14)
        np.testing.assert_allclose(s_dm2, m_dm2.get(), atol=1e-14)

        # Test solver methods with explicit norb and nelec
        dm1_exp = solver.trans_rdm1(ci0_gpu, ci1_gpu, norb=norb, nelec=nelec)
        np.testing.assert_allclose(dm1_exp, dm1, atol=1e-14)

    def test_precomputed_gpu_link_index(self):
        '''Verify passing precomputed 4-tuple CuPy link_index executes identically.'''
        norb = 4
        nelec = (2, 2)
        ci_roots_cpu = _make_orthogonal_roots(norb, nelec, nroots=2, seed=606)
        ci0_gpu = cp.asarray(ci_roots_cpu[0])
        ci1_gpu = cp.asarray(ci_roots_cpu[1])

        gpu_link_index = gpu_direct_spin1._prepare_rdm_link_index(None, norb, nelec)
        self.assertIsInstance(gpu_link_index, tuple)
        self.assertEqual(len(gpu_link_index), 4)
        self.assertIsInstance(gpu_link_index[0], cp.ndarray)

        # Reuse link_index
        dm1_cached = gpu_direct_spin1.trans_rdm1(
            ci0_gpu, ci1_gpu, norb, nelec, link_index=gpu_link_index)
        dm1_default = gpu_direct_spin1.trans_rdm1(
            ci0_gpu, ci1_gpu, norb, nelec, link_index=None)
        np.testing.assert_allclose(dm1_cached.get(), dm1_default.get(), atol=1e-14)

        dm12_cached = gpu_direct_spin1.trans_rdm12(
            ci0_gpu, ci1_gpu, norb, nelec, link_index=gpu_link_index)
        dm12_default = gpu_direct_spin1.trans_rdm12(
            ci0_gpu, ci1_gpu, norb, nelec, link_index=None)
        np.testing.assert_allclose(dm12_cached[0].get(), dm12_default[0].get(), atol=1e-14)
        np.testing.assert_allclose(dm12_cached[1].get(), dm12_default[1].get(), atol=1e-14)

    def test_input_validation_and_error_handling(self):
        '''Verify that invalid CI vector shapes or sizes raise appropriate errors.'''
        norb = 4
        nelec = (2, 2)
        na = lib.comb(norb, 2)
        nb = lib.comb(norb, 2)
        correct_size = na * nb
        wrong_ci = cp.zeros(correct_size + 1)
        valid_ci = cp.zeros(correct_size)

        with self.assertRaises(ValueError):
            gpu_direct_spin1.trans_rdm1(wrong_ci, valid_ci, norb, nelec)
        with self.assertRaises(ValueError):
            gpu_direct_spin1.trans_rdm1(valid_ci, wrong_ci, norb, nelec)
        with self.assertRaises(ValueError):
            gpu_direct_spin1.trans_rdm12(wrong_ci, valid_ci, norb, nelec)
        with self.assertRaises(ValueError):
            gpu_direct_spin1.trans_rdm12(valid_ci, wrong_ci, norb, nelec)


if __name__ == '__main__':
    unittest.main()
