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

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

from pyscf import lib
from pyscf.mcscf import addons as cpu_addons
from pyscf.mcscf.addons import StateAverageMCSCFSolver

from gpu4pyscf.fci import direct_spin1 as gpu_direct_spin1
from gpu4pyscf.lib import logger


class StateAverageFCISolver:
    __name_mixin__ = 'StateAverage'

    _keys = {'weights', 'e_states', '_converged'}

    def __init__(self, fcibase, weights, wfnsym=None):
        self._fcibase = fcibase
        self._nroots_orig = getattr(fcibase, 'nroots', 1)
        self.__dict__.update(fcibase.__dict__)
        if 'converged' in self.__dict__:
            self._converged = self.__dict__.pop('converged')
        else:
            self._converged = False

        weights = np.asarray(weights, dtype=float)
        if len(weights) == 0 or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError(f"State weights must be positive and finite, got {weights}")
        self.weights = weights / np.sum(weights)
        self.nroots = len(self.weights)
        if wfnsym is not None:
            self.wfnsym = wfnsym
        self.e_states = [None]
        self._base_class = fcibase.__class__

    def _get_fcibase(self):
        if hasattr(self, '_fcibase') and self._fcibase is not None:
            return self._fcibase
        return super()

    @property
    def converged(self):
        c = getattr(self, '_converged', None)
        if c is None:
            c = self.__dict__.get('converged', False)
        if isinstance(c, (bool, np.bool_)):
            return bool(c)
        return bool(np.all(c))

    @converged.setter
    def converged(self, val):
        self._converged = val

    @property
    def converged_roots(self):
        return getattr(self, '_converged', False)

    def undo_state_average(self):
        converged_val = getattr(self, '_converged', False)
        orig_nroots = getattr(self, '_nroots_orig', 1)
        obj = lib.view(self, lib.drop_class(self.__class__, StateAverageFCISolver))
        for attr in ('weights', 'e_states', '_converged', '_nroots_orig', '_fcibase'):
            if hasattr(obj, attr):
                try:
                    delattr(obj, attr)
                except AttributeError:
                    pass
        obj.nroots = orig_nroots
        obj.converged = converged_val
        return obj

    def dump_flags(self, verbose=None):
        fcibase = self._get_fcibase()
        if hasattr(fcibase, 'dump_flags'):
            fcibase.dump_flags(verbose)
        log = logger.new_logger(self, verbose)
        log.info('State-average over %d states with weights %s',
                 len(self.weights), self.weights)
        return self

    def kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
        if 'nroots' not in kwargs:
            kwargs['nroots'] = self.nroots

        fcibase = self._get_fcibase()
        wfnsym = getattr(self, 'wfnsym', None)
        if wfnsym is not None:
            kwargs['wfnsym'] = wfnsym
        e, c = fcibase.kernel(h1, h2, norb, nelec, ci0=ci0, **kwargs)

        e = np.atleast_1d(e)
        self.e_states = e
        self.converged = getattr(fcibase, 'converged', True)

        if not isinstance(c, (list, tuple)):
            c = [c]

        log = logger.new_logger(self, kwargs.get('verbose'))
        if log.verbose >= logger.DEBUG:
            if getattr(self, 'states_spin_square', None):
                try:
                    ss = self.states_spin_square(c, norb, nelec)[0]
                    for i, ei in enumerate(e):
                        log.debug('state %d  E = %.15g S^2 = %.7f', i, ei, ss[i])
                except Exception:
                    for i, ei in enumerate(e):
                        log.debug('state %d  E = %.15g', i, ei)
            else:
                for i, ei in enumerate(e):
                    log.debug('state %d  E = %.15g', i, ei)

        e_average = float(np.einsum('i,i->', e, self.weights))
        return e_average, c

    def approx_kernel(self, h1, h2, norb, nelec, ci0=None, **kwargs):
        if 'nroots' not in kwargs:
            kwargs['nroots'] = self.nroots

        fcibase = self._get_fcibase()
        wfnsym = getattr(self, 'wfnsym', None)
        if wfnsym is not None:
            kwargs['wfnsym'] = wfnsym
        if hasattr(fcibase, 'approx_kernel'):
            e, c = fcibase.approx_kernel(h1, h2, norb, nelec, ci0=ci0, **kwargs)
        else:
            e, c = fcibase.kernel(h1, h2, norb, nelec, ci0=ci0, **kwargs)

        e = np.atleast_1d(e)
        self.e_states = e
        self.converged = getattr(fcibase, 'converged', True)
        if not isinstance(c, (list, tuple)):
            c = [c]
        e_average = float(np.einsum('i,i->', e, self.weights))
        return e_average, c

    def states_make_rdm1(self, ci0, norb, nelec, *args, **kwargs):
        if not isinstance(ci0, (list, tuple)) and getattr(ci0, 'ndim', 0) == 3:
            ci0 = [ci0[i] for i in range(len(ci0))]
        elif not isinstance(ci0, (list, tuple)):
            ci0 = [ci0]

        ci0_roots = ci0[:len(self.weights)]
        rdm1 = []
        fcibase = self._get_fcibase()
        for c in ci0_roots:
            if cp is not None and isinstance(c, cp.ndarray):
                dm1, _ = gpu_direct_spin1.make_rdm12(
                    c, norb, nelec, *args, reorder=False, **kwargs)
            else:
                dm1 = fcibase.make_rdm1(c, norb, nelec, *args, **kwargs)
            rdm1.append(dm1)
        return rdm1

    def make_rdm1(self, ci0, norb, nelec, *args, **kwargs):
        rdm1 = self.states_make_rdm1(ci0, norb, nelec, *args, **kwargs)
        return sum(w * dm for w, dm in zip(self.weights, rdm1))

    def states_make_rdm1s(self, ci0, norb, nelec, *args, **kwargs):
        if not isinstance(ci0, (list, tuple)) and getattr(ci0, 'ndim', 0) == 3:
            ci0 = [ci0[i] for i in range(len(ci0))]
        elif not isinstance(ci0, (list, tuple)):
            ci0 = [ci0]

        ci0_roots = ci0[:len(self.weights)]
        dm1a = []
        dm1b = []
        fcibase = self._get_fcibase()
        for c in ci0_roots:
            c_cpu = cp.asnumpy(c) if (cp is not None and isinstance(c, cp.ndarray)) else c
            s1, s2 = fcibase.make_rdm1s(c_cpu, norb, nelec, *args, **kwargs)
            if cp is not None and isinstance(c, cp.ndarray):
                dm1a.append(cp.asarray(s1))
                dm1b.append(cp.asarray(s2))
            else:
                dm1a.append(s1)
                dm1b.append(s2)
        return dm1a, dm1b

    def make_rdm1s(self, ci0, norb, nelec, *args, **kwargs):
        dm1a, dm1b = self.states_make_rdm1s(ci0, norb, nelec, *args, **kwargs)
        return (sum(w * dm for w, dm in zip(self.weights, dm1a)),
                sum(w * dm for w, dm in zip(self.weights, dm1b)))

    def states_make_rdm12(self, ci0, norb, nelec, *args, **kwargs):
        if not isinstance(ci0, (list, tuple)) and getattr(ci0, 'ndim', 0) == 3:
            ci0 = [ci0[i] for i in range(len(ci0))]
        elif not isinstance(ci0, (list, tuple)):
            ci0 = [ci0]

        ci0_roots = ci0[:len(self.weights)]
        rdm1 = []
        rdm2 = []
        fcibase = self._get_fcibase()
        for c in ci0_roots:
            if cp is not None and isinstance(c, cp.ndarray):
                dm1, dm2 = gpu_direct_spin1.make_rdm12(
                    c, norb, nelec, *args, **kwargs)
            else:
                dm1, dm2 = fcibase.make_rdm12(
                    c, norb, nelec, *args, **kwargs)
            rdm1.append(dm1)
            rdm2.append(dm2)
        return rdm1, rdm2

    def make_rdm12(self, ci0, norb, nelec, *args, **kwargs):
        rdm1, rdm2 = self.states_make_rdm12(ci0, norb, nelec, *args, **kwargs)
        casdm1 = sum(w * dm for w, dm in zip(self.weights, rdm1))
        casdm2 = sum(w * dm for w, dm in zip(self.weights, rdm2))
        return casdm1, casdm2

    def states_spin_square(self, ci0, norb, nelec, *args, **kwargs):
        if not isinstance(ci0, (list, tuple)) and getattr(ci0, 'ndim', 0) == 3:
            ci0 = [ci0[i] for i in range(len(ci0))]
        elif not isinstance(ci0, (list, tuple)):
            ci0 = [ci0]

        ci0_roots = ci0[:len(self.weights)]
        fcibase = self._get_fcibase()
        ss = []
        mult = []
        for c in ci0_roots:
            c_cpu = cp.asnumpy(c) if (cp is not None and isinstance(c, cp.ndarray)) else c
            s, m = fcibase.spin_square(c_cpu, norb, nelec, *args, **kwargs)
            ss.append(s)
            mult.append(m)
        return ss, mult

    def spin_square(self, ci0, norb, nelec, *args, **kwargs):
        ss, mult = self.states_spin_square(ci0, norb, nelec, *args, **kwargs)
        return float(np.dot(ss, self.weights)), float(np.dot(mult, self.weights))

    def trans_rdm1(self, cibra, ciket, norb=None, nelec=None, link_index=None):
        fcibase = self._get_fcibase()
        if hasattr(fcibase, 'trans_rdm1'):
            return fcibase.trans_rdm1(cibra, ciket, norb, nelec, link_index=link_index)
        return gpu_direct_spin1.trans_rdm1(cibra, ciket, norb, nelec, link_index=link_index)

    def trans_rdm12(self, cibra, ciket, norb=None, nelec=None, link_index=None, reorder=True):
        fcibase = self._get_fcibase()
        if hasattr(fcibase, 'trans_rdm12'):
            return fcibase.trans_rdm12(cibra, ciket, norb, nelec, link_index=link_index, reorder=reorder)
        return gpu_direct_spin1.trans_rdm12(cibra, ciket, norb, nelec, link_index=link_index, reorder=reorder)


class StateAverageMCSCF(StateAverageMCSCFSolver):
    __name_mixin__ = 'StateAverage'

    def __init__(self, my_mc, fcisolver):
        self.__dict__.update(my_mc.__dict__)
        self.fcisolver = fcisolver

    @property
    def casdm1(self):
        if getattr(self, 'ci', None) is not None:
            return self.fcisolver.make_rdm1(self.ci, self.ncas, self.nelecas)
        return None

    def undo_state_average(self):
        obj = lib.view(self, lib.drop_class(self.__class__, StateAverageMCSCF))
        if isinstance(self.fcisolver, StateAverageFCISolver):
            obj.fcisolver = self.fcisolver.undo_state_average()
        return obj

    @property
    def _base_class(self):
        '''for convenience; this is equal to mcscf base class'''
        return self.__class__.__bases__[1]

    @property
    def weights(self):
        return self.fcisolver.weights

    @weights.setter
    def weights(self, x):
        weights = np.asarray(x, dtype=float)
        if len(weights) == 0 or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError(f"State weights must be positive and finite, got {weights}")
        self.fcisolver.weights = weights / np.sum(weights)
        self.fcisolver.nroots = len(self.fcisolver.weights)
        return self.fcisolver.weights

    @property
    def e_average(self):
        return float(np.dot(self.fcisolver.weights, self.fcisolver.e_states))

    @property
    def e_states(self):
        return self.fcisolver.e_states

    def _finalize(self):
        try:
            super()._finalize()
        except AttributeError:
            if hasattr(self, '_base_class') and hasattr(self._base_class, '_finalize'):
                self._base_class._finalize(self)

        logger.note(self, 'CASCI state-averaged energy = %.15g', self.e_average)
        logger.note(self, 'CASCI energy for each state')
        if getattr(self.fcisolver, 'states_spin_square', None):
            try:
                ss = self.fcisolver.states_spin_square(self.ci, self.ncas,
                                                       self.nelecas)[0]
                for i, ei in enumerate(self.e_states):
                    logger.note(self, '  State %d weight %g  E = %.15g S^2 = %.7f',
                                i, self.weights[i], ei, ss[i])
            except Exception:
                for i, ei in enumerate(self.e_states):
                    logger.note(self, '  State %d weight %g  E = %.15g',
                                i, self.weights[i], ei)
        else:
            for i, ei in enumerate(self.e_states):
                logger.note(self, '  State %d weight %g  E = %.15g',
                            i, self.weights[i], ei)
        return self

    def dump_chk(self, envs_or_file):
        if isinstance(envs_or_file, dict):
            envs = envs_or_file.copy()
            if 'fcivec' in envs:
                fcivec = envs['fcivec']
                if isinstance(fcivec, (list, tuple)) or (isinstance(fcivec, np.ndarray) and fcivec.dtype == object):
                    cleaned = []
                    for c in fcivec:
                        if cp is not None and isinstance(c, cp.ndarray):
                            c = cp.asnumpy(c)
                        c = np.asarray(c)
                        if c.dtype == object:
                            c = np.asarray(c, dtype=float)
                        cleaned.append(c)
                    envs['fcivec'] = cleaned
                elif cp is not None and isinstance(fcivec, cp.ndarray):
                    envs['fcivec'] = cp.asnumpy(fcivec)
            if 'casdm1' in envs and cp is not None and isinstance(envs['casdm1'], cp.ndarray):
                envs['casdm1'] = cp.asnumpy(envs['casdm1'])
            if 'mo' in envs and cp is not None and isinstance(envs['mo'], cp.ndarray):
                envs['mo'] = cp.asnumpy(envs['mo'])
            envs_or_file = envs
        try:
            return super().dump_chk(envs_or_file)
        except AttributeError:
            if hasattr(self, '_base_class') and hasattr(self._base_class, 'dump_chk'):
                return self._base_class.dump_chk(self, envs_or_file)
            return self

    def nuc_grad_method(self, state=None):
        from gpu4pyscf.grad import sacasscf as sacasscf_grad
        return sacasscf_grad.Gradients(self, state=state)

    Gradients = nuc_grad_method

    def nac_method(self, state=(0, 1), use_etfs=True, mult_ediff=False):
        from gpu4pyscf.nac import sacasscf as sacasscf_nac
        return sacasscf_nac.NonAdiabaticCoupling(
            self, state=state, use_etfs=use_etfs, mult_ediff=mult_ediff
        )

    NACs = nac_method
    NACGradients = nac_method



def _state_average_mcscf_solver(casscf, fcisolver):
    if isinstance(casscf, StateAverageMCSCFSolver):
        casscf = casscf.undo_state_average()

    return lib.set_class(
        StateAverageMCSCF(casscf, fcisolver),
        (StateAverageMCSCF, casscf.__class__)
    )


def state_average(casscf, weights=(0.5, 0.5), wfnsym=None):
    weights = np.asarray(weights, dtype=float)
    if len(weights) == 0 or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError(f"State weights must be positive and finite, got {weights}")
    weights = weights / np.sum(weights)

    fcisolver = casscf.fcisolver
    if isinstance(fcisolver, StateAverageFCISolver):
        fcisolver.nroots = len(weights)
        fcisolver.weights = weights
    else:
        fcisolver = lib.set_class(
            StateAverageFCISolver(fcisolver, weights, wfnsym),
            (StateAverageFCISolver, fcisolver.__class__)
        )
    return _state_average_mcscf_solver(casscf, fcisolver)


def state_average_(casscf, weights=(0.5, 0.5), wfnsym=None):
    '''Inplace version of state_average'''
    sacasscf = state_average(casscf, weights, wfnsym)
    casscf.__class__ = sacasscf.__class__
    casscf.__dict__ = sacasscf.__dict__
    return casscf
