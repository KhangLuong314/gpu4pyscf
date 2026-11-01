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

import tempfile
import unittest
from unittest import mock
import numpy as np

import pyscf
from pyscf import lib, scf
from pyscf import mcscf as cpu_mcscf
from pyscf.mcscf import addons as cpu_addons
from pyscf.fci import direct_spin1 as cpu_direct_spin1

try:
    import cupy as cp
    cuda_available = cp.cuda.is_available() and cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp = None
    cuda_available = False

try:
    from gpu4pyscf import mcscf
    from gpu4pyscf.mcscf import addons as gpu_addons
    from gpu4pyscf.mcscf.addons import (
        StateAverageFCISolver,
        StateAverageMCSCF,
        StateAverageMCSCFSolver,
        state_average,
        state_average_,
    )
    from gpu4pyscf.mcscf import df as gpu_mcscf_df
except (ImportError, ModuleNotFoundError):
    import importlib.util, types, sys, os
    mcscf = None
    gpu_mcscf_df = None
    if 'gpu4pyscf' not in sys.modules:
        sys.modules['gpu4pyscf'] = types.ModuleType('gpu4pyscf')
    if 'gpu4pyscf.mcscf' not in sys.modules:
        sys.modules['gpu4pyscf.mcscf'] = types.ModuleType('gpu4pyscf.mcscf')
    if 'gpu4pyscf.fci' not in sys.modules:
        sys.modules['gpu4pyscf.fci'] = types.ModuleType('gpu4pyscf.fci')
    if 'gpu4pyscf.fci.direct_spin1' not in sys.modules:
        sys.modules['gpu4pyscf.fci.direct_spin1'] = types.ModuleType('gpu4pyscf.fci.direct_spin1')
    if 'gpu4pyscf.lib' not in sys.modules:
        sys.modules['gpu4pyscf.lib'] = types.ModuleType('gpu4pyscf.lib')
    if 'gpu4pyscf.lib.logger' not in sys.modules:
        from pyscf.lib import logger as cpu_logger
        sys.modules['gpu4pyscf.lib.logger'] = cpu_logger

    addons_path = os.path.join(os.path.dirname(__file__), '..', 'addons.py')
    spec = importlib.util.spec_from_file_location('gpu4pyscf.mcscf.addons', addons_path)
    gpu_addons = importlib.util.module_from_spec(spec)
    sys.modules['gpu4pyscf.mcscf.addons'] = gpu_addons
    spec.loader.exec_module(gpu_addons)

    StateAverageFCISolver = gpu_addons.StateAverageFCISolver
    StateAverageMCSCF = gpu_addons.StateAverageMCSCF
    StateAverageMCSCFSolver = gpu_addons.StateAverageMCSCFSolver
    state_average = gpu_addons.state_average
    state_average_ = gpu_addons.state_average_


class TestStateAverageMCSCFClassLogic(unittest.TestCase):
    '''Test suite for StateAverageMCSCF and StateAverageFCISolver class logic,
    properties, and arithmetic, runnable on any host environment.
    '''

    def setUp(self):
        self.mol = pyscf.M(
            atom='H 0 0 0; H 0 0 0.74', basis='sto-3g',
            verbose=0, output='/dev/null')
        self.mf = scf.RHF(self.mol).run()

    def tearDown(self):
        self.mol.stdout.close()

    def test_weights_normalization_and_validation(self):
        mc = cpu_mcscf.CASSCF(self.mf, 2, 2)
        # Unnormalized weights should normalize to sum 1.0
        gpu_addons.state_average_(mc, weights=[2.0, 1.0, 1.0])
        np.testing.assert_allclose(mc.weights, [0.5, 0.25, 0.25])
        self.assertEqual(mc.fcisolver.nroots, 3)

        # Unequal weights
        mc.weights = [0.7, 0.3]
        np.testing.assert_allclose(mc.weights, [0.7, 0.3])
        self.assertEqual(mc.fcisolver.nroots, 2)

        # Single root
        mc.weights = [1.0]
        np.testing.assert_allclose(mc.weights, [1.0])
        self.assertEqual(mc.fcisolver.nroots, 1)

        # Invalid weights (zero, negative, empty, non-finite)
        with self.assertRaises(ValueError):
            gpu_addons.state_average(mc, weights=[0.5, -0.5])
        with self.assertRaises(ValueError):
            mc.weights = [0.0, 1.0]
        with self.assertRaises(ValueError):
            gpu_addons.state_average(mc, weights=[])
        with self.assertRaises(ValueError):
            mc.weights = []
        with self.assertRaises(ValueError):
            gpu_addons.state_average(mc, weights=[0.5, np.nan])
        with self.assertRaises(ValueError):
            gpu_addons.state_average(mc, weights=[0.5, np.inf])

    def test_converged_property_scalar_boolean(self):
        fcibase = cpu_direct_spin1.FCISolver(self.mol)
        sa_fci = StateAverageFCISolver(fcibase, weights=[0.5, 0.5])

        # Test initial state
        self.assertFalse(sa_fci.converged)
        self.assertIsInstance(sa_fci.converged, bool)

        # When multi-root Davidson returns boolean array with all True
        sa_fci.converged = np.array([True, True])
        self.assertTrue(sa_fci.converged)
        self.assertIsInstance(sa_fci.converged, bool)
        # Evaluating in an if statement must NOT raise ValueError
        evaluated = bool(sa_fci.converged)
        self.assertTrue(evaluated)

        # When multi-root Davidson returns array with one False
        sa_fci.converged = np.array([True, False])
        self.assertFalse(sa_fci.converged)
        self.assertIsInstance(sa_fci.converged, bool)

        # When single root returns bool
        sa_fci.converged = True
        self.assertTrue(sa_fci.converged)
        self.assertIsInstance(sa_fci.converged, bool)

    def test_undo_state_average(self):
        mc = cpu_mcscf.CASSCF(self.mf, 2, 2)
        gpu_addons.state_average_(mc, weights=[0.6, 0.4])
        self.assertIsInstance(mc, StateAverageMCSCFSolver)
        self.assertIsInstance(mc.fcisolver, StateAverageFCISolver)

        # State-specific gradients must raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            mc.nuc_grad_method(state=0)

        mc_undone = mc.undo_state_average()
        self.assertNotIsInstance(mc_undone, StateAverageMCSCFSolver)
        self.assertNotIsInstance(mc_undone.fcisolver, StateAverageFCISolver)
        self.assertFalse(hasattr(mc_undone, 'weights'))
        self.assertFalse(hasattr(mc_undone.fcisolver, 'weights'))
        self.assertEqual(mc_undone.fcisolver.nroots, 1)
        self.assertTrue(hasattr(mc_undone.fcisolver, 'converged'))

    def test_rdm_averaging_arithmetic_vs_cpu(self):
        norb = 4
        nelec = (2, 2)
        na = lib.comb(norb, nelec[0])
        nb = lib.comb(norb, nelec[1])
        np.random.seed(42)
        c0 = np.random.randn(na, nb)
        c0 /= np.linalg.norm(c0)
        c1 = np.random.randn(na, nb)
        c1 /= np.linalg.norm(c1)

        weights = [0.65, 0.35]
        fcibase_cpu = cpu_direct_spin1.FCISolver(self.mol)
        sa_cpu = lib.set_class(
            cpu_addons.StateAverageFCISolver(fcibase_cpu, weights, wfnsym=None),
            (cpu_addons.StateAverageFCISolver, fcibase_cpu.__class__))
        sa_gpu = StateAverageFCISolver(fcibase_cpu, weights, wfnsym=None)

        # Check 1-RDM and 2-RDM averaging equivalence
        dm1_ref, dm2_ref = sa_cpu.make_rdm12([c0, c1], norb, nelec)
        dm1_test, dm2_test = sa_gpu.make_rdm12([c0, c1], norb, nelec)

        np.testing.assert_allclose(dm1_test, dm1_ref, atol=1e-12)
        np.testing.assert_allclose(dm2_test, dm2_ref, atol=1e-12)

        # Check spin-resolved 1-RDM averaging
        dm1s_ref = sa_cpu.make_rdm1s([c0, c1], norb, nelec)
        dm1s_test = sa_gpu.make_rdm1s([c0, c1], norb, nelec)
        np.testing.assert_allclose(dm1s_test[0], dm1s_ref[0], atol=1e-12)
        np.testing.assert_allclose(dm1s_test[1], dm1s_ref[1], atol=1e-12)

        # Check spin square averaging
        ss_ref = np.dot([fcibase_cpu.spin_square(c, norb, nelec)[0] for c in [c0, c1]], weights)
        mult_ref = np.dot([fcibase_cpu.spin_square(c, norb, nelec)[1] for c in [c0, c1]], weights)
        ss_test, mult_test = sa_gpu.spin_square([c0, c1], norb, nelec)
        self.assertAlmostEqual(ss_test, ss_ref, delta=1e-10)
        self.assertAlmostEqual(mult_test, mult_ref, delta=1e-10)

    def test_single_root_edge_case(self):
        norb = 2
        nelec = (1, 1)
        c0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        fcibase = cpu_direct_spin1.FCISolver(self.mol)
        sa_fci = StateAverageFCISolver(fcibase, weights=[1.0])

        dm1, dm2 = sa_fci.make_rdm12(c0, norb, nelec)
        self.assertEqual(dm1.shape, (norb, norb))
        self.assertEqual(dm2.shape, (norb, norb, norb, norb))

        # Check mock kernel handling scalar e
        with mock.patch.object(fcibase, 'kernel', return_value=( -1.137, c0)):
            e_avg, c_out = sa_fci.kernel(None, None, norb, nelec)
            self.assertAlmostEqual(e_avg, -1.137)
            self.assertIsInstance(c_out, list)
            self.assertEqual(len(c_out), 1)

    def test_dump_chk_ci_vector_sanitization(self):
        mc = cpu_mcscf.CASSCF(self.mf, 2, 2)
        gpu_addons.state_average_(mc, weights=[0.5, 0.5])
        mc.chk_ci = True
        c0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        c1 = np.array([[0.0, 1.0], [0.0, 0.0]])

        with tempfile.NamedTemporaryFile() as tmp:
            mc.chkfile = tmp.name
            envs = {
                'fcivec': [c0, c1],
                'e_tot': -1.137,
                'e_cas': -1.137,
                'casdm1': np.eye(2),
                'mo_coeff': np.eye(2),
            }
            # Must serialize without raising TypeError or ValueError
            mc.dump_chk(envs)
            data = lib.chkfile.load(mc.chkfile, 'mcscf')
            self.assertIn('ci', data)
            self.assertIn('e_tot', data)

        # Test object array as constructed by casscf.py:194
        with tempfile.NamedTemporaryFile() as tmp2:
            mc.chkfile = tmp2.name
            envs_obj = {
                'fcivec': np.array([c0, c1], dtype=object),
                'e_tot': -1.137,
                'e_cas': -1.137,
                'casdm1': np.eye(2),
                'mo_coeff': np.eye(2),
            }
            mc.dump_chk(envs_obj)
            data2 = lib.chkfile.load(mc.chkfile, 'mcscf')
            self.assertIn('ci', data2)
            self.assertIn('e_tot', data2)


@unittest.skipUnless(cuda_available, "CUDA device required for full GPU calculations")
class TestStateAverageMCSCFGPU(unittest.TestCase):
    '''Full integration tests on NVIDIA GPU device.'''

    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.M(
            atom='N 0 0 -0.7; N 0 0 0.7',
            basis='sto-3g',
            verbose=0,
            output='/dev/null')
        cls.mf_cpu = scf.RHF(cls.mol).density_fit(auxbasis='weigend')
        cls.mf_cpu.kernel()
        cls.mf_gpu = cls.mf_cpu.to_gpu()

    @classmethod
    def tearDownClass(cls):
        cls.mol.stdout.close()

    def test_gpu_state_average_rdm_accuracy(self):
        '''Test that GPU multi-root 1- and 2-RDMs match CPU PySCF within 1e-10.'''
        norb = 4
        nelec = (2, 2)
        na = lib.comb(norb, nelec[0])
        nb = lib.comb(norb, nelec[1])
        np.random.seed(123)
        c0_cpu = np.random.randn(na, nb)
        c0_cpu /= np.linalg.norm(c0_cpu)
        c1_cpu = np.random.randn(na, nb)
        c1_cpu /= np.linalg.norm(c1_cpu)
        c2_cpu = np.random.randn(na, nb)
        c2_cpu /= np.linalg.norm(c2_cpu)

        c0_gpu = cp.asarray(c0_cpu)
        c1_gpu = cp.asarray(c1_cpu)
        c2_gpu = cp.asarray(c2_cpu)

        weights = [0.5, 0.3, 0.2]
        # CPU reference
        sa_cpu = cpu_addons.StateAverageFCISolver(
            cpu_direct_spin1.FCISolver(self.mol), weights)
        dm1_ref, dm2_ref = sa_cpu.make_rdm12([c0_cpu, c1_cpu, c2_cpu], norb, nelec)

        # GPU evaluation
        from gpu4pyscf.fci import direct_spin1 as gpu_direct_spin1
        sa_gpu = StateAverageFCISolver(
            gpu_direct_spin1.FCISolver(self.mol), weights)
        dm1_gpu, dm2_gpu = sa_gpu.make_rdm12([c0_gpu, c1_gpu, c2_gpu], norb, nelec)

        # Verify device residency of resulting RDMs
        self.assertIsInstance(dm1_gpu, cp.ndarray)
        self.assertIsInstance(dm2_gpu, cp.ndarray)

        # Numerical comparison against CPU reference
        np.testing.assert_allclose(cp.asnumpy(dm1_gpu), dm1_ref, atol=1e-10)
        np.testing.assert_allclose(cp.asnumpy(dm2_gpu), dm2_ref, atol=1e-10)

    def test_df_sa_casscf_equal_weights_convergence(self):
        '''Ethylene / N2 CAS(4,4) 3-state equal weights SA-CASSCF convergence.'''
        mc_gpu = mcscf.DFCASSCF(self.mf_gpu, 4, 4)
        mc_gpu.state_average_([1/3, 1/3, 1/3])
        mc_gpu.max_cycle_macro = 20
        mc_gpu.conv_tol = 1e-8
        mc_gpu.conv_tol_grad = 1e-5
        e_tot = mc_gpu.kernel()[0]

        self.assertIsInstance(mc_gpu, StateAverageMCSCFSolver)
        self.assertIsInstance(mc_gpu.fcisolver, StateAverageFCISolver)
        self.assertTrue(mc_gpu.converged)
        self.assertTrue(mc_gpu.fcisolver.converged)
        self.assertIsInstance(mc_gpu.fcisolver.converged, bool)
        self.assertEqual(len(mc_gpu.e_states), 3)

        # Compare with CPU reference
        mc_cpu = cpu_mcscf.DFCASSCF(self.mf_cpu, 4, 4, auxbasis='weigend')
        mc_cpu.state_average_([1/3, 1/3, 1/3])
        mc_cpu.canonicalization = False
        mc_cpu.max_cycle_macro = 20
        e_cpu = mc_cpu.kernel()[0]

        self.assertAlmostEqual(e_tot, e_cpu, delta=1e-7)
        np.testing.assert_allclose(mc_gpu.e_states, mc_cpu.e_states, atol=1e-7)

    def test_df_sa_casscf_unequal_weights(self):
        '''SA-CASSCF with unequal weights [0.7, 0.3].'''
        mc_gpu = mcscf.DFCASSCF(self.mf_gpu, 4, 4)
        mc_gpu.state_average_([0.7, 0.3])
        mc_gpu.max_cycle_macro = 20
        mc_gpu.kernel()

        self.assertTrue(mc_gpu.converged)
        self.assertEqual(len(mc_gpu.e_states), 2)

        mc_cpu = cpu_mcscf.DFCASSCF(self.mf_cpu, 4, 4, auxbasis='weigend')
        mc_cpu.state_average_([0.7, 0.3])
        mc_cpu.canonicalization = False
        mc_cpu.max_cycle_macro = 20
        mc_cpu.kernel()

        self.assertAlmostEqual(mc_gpu.e_tot, mc_cpu.e_tot, delta=1e-7)
        np.testing.assert_allclose(mc_gpu.e_states, mc_cpu.e_states, atol=1e-7)

    def test_df_sa_casscf_single_state_edge_case(self):
        '''SA-CASSCF with single root [1.0].'''
        mc_gpu = mcscf.DFCASSCF(self.mf_gpu, 4, 4)
        mc_gpu.state_average_([1.0])
        mc_gpu.max_cycle_macro = 20
        mc_gpu.kernel()

        self.assertTrue(mc_gpu.converged)
        self.assertEqual(len(mc_gpu.e_states), 1)

    def test_to_cpu_from_cpu_roundtrip(self):
        mc_gpu = mcscf.DFCASSCF(self.mf_gpu, 4, 4)
        mc_gpu.state_average_([0.6, 0.4])

        # Populate CI vectors and convergence status to verify roundtrip
        c0 = cp.ones((6, 6))
        c1 = cp.zeros((6, 6))
        mc_gpu.ci = [c0, c1]
        mc_gpu.fcisolver.ci = [c0, c1]
        mc_gpu.fcisolver.converged = np.array([True, True])

        mc_cpu = mc_gpu.to_cpu()
        self.assertIsInstance(mc_cpu, cpu_addons.StateAverageMCSCFSolver)
        np.testing.assert_allclose(mc_cpu.weights, [0.6, 0.4])
        self.assertIsInstance(mc_cpu.ci, list)
        self.assertIsInstance(mc_cpu.ci[0], np.ndarray)
        self.assertNotIsInstance(mc_cpu.ci[0], cp.ndarray)
        self.assertTrue(mc_cpu.fcisolver.converged)

        mc_gpu_back = gpu_mcscf_df.from_cpu(mc_cpu)
        self.assertIsInstance(mc_gpu_back, StateAverageMCSCFSolver)
        np.testing.assert_allclose(mc_gpu_back.weights, [0.6, 0.4])
        self.assertIsInstance(mc_gpu_back.ci, list)
        self.assertIsInstance(mc_gpu_back.ci[0], cp.ndarray)
        self.assertTrue(mc_gpu_back.fcisolver.converged)


if __name__ == '__main__':
    unittest.main()
