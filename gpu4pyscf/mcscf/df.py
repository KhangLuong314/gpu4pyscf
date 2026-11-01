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

import cupy
import numpy
from pyscf import mcscf as cpu_mcscf
from pyscf.fci.addons import SpinPenaltyFCISolver
from pyscf.mcscf import mc1step as cpu_mc1step
from pyscf.mcscf.addons import StateAverageMCSCFSolver

from gpu4pyscf.df import df as gpu_df
from gpu4pyscf.lib import logger
from gpu4pyscf.lib import utils
from gpu4pyscf.mcscf.casci import _CASCI
from gpu4pyscf.mcscf.casscf import _CASSCF, _ERIS


def _get_with_df(mc, auxbasis=None, with_df=None):
    if with_df is None:
        scf_df = getattr(mc._scf, 'with_df', None)
        if (scf_df is not None and
                (auxbasis is None or auxbasis == scf_df.auxbasis)):
            return scf_df

        if auxbasis is None and isinstance(mc.mol.basis, str):
            from pyscf.df.addons import predefined_auxbasis
            auxbasis = predefined_auxbasis(mc.mol, mc.mol.basis, xc='HF')
        with_df = gpu_df.DF(mc.mol, auxbasis)
        with_df.max_memory = mc.max_memory
        with_df.stdout = mc.stdout
        with_df.verbose = mc.verbose
    return with_df


class _DFCAS:
    _keys = {'with_df'}

    def reset(self, mol=None):
        if self.with_df is not getattr(self._scf, 'with_df', None):
            self.with_df.reset(mol)
        return super().reset(mol)

    def get_jk(self, mol, dm, hermi=1, with_j=True, with_k=True, omega=None):
        return self.with_df.get_jk(
            dm, hermi, with_j=with_j, with_k=with_k, omega=omega)

    def get_h2eff(self, mo_coeff=None):
        log = logger.new_logger(self)
        t0 = log.init_timer()
        ncore = self.ncore
        nocc = ncore + self.ncas
        if mo_coeff is None:
            mo_coeff = self.mo_coeff[:, ncore:nocc]
        elif mo_coeff.shape[1] != self.ncas:
            mo_coeff = mo_coeff[:, ncore:nocc]
        eri = self.with_df.ao2mo(mo_coeff)
        out = eri.get() if isinstance(eri, cupy.ndarray) else eri
        log.timer('DF CASCI AO2MO', *t0)
        return out

    def to_cpu(self):
        if isinstance(self, _CASSCF):
            out = cpu_mcscf.DFCASSCF(
                self._scf.to_cpu(), self.ncas, self.nelecas,
                auxbasis=self.with_df.auxbasis, ncore=self.ncore)
        else:
            out = cpu_mcscf.DFCASCI(
                self._scf.to_cpu(), self.ncas, self.nelecas,
                auxbasis=self.with_df.auxbasis, ncore=self.ncore)
        if isinstance(self, StateAverageMCSCFSolver):
            out = cpu_mcscf.addons.state_average_(out, weights=self.weights)
        out = utils.to_cpu(self, out=out)
        if isinstance(getattr(out, 'ci', None), (list, tuple)):
            out.ci = [c.get() if hasattr(c, 'get') else numpy.asarray(c) for c in out.ci]
        elif hasattr(getattr(out, 'ci', None), 'get'):
            out.ci = out.ci.get()
        if hasattr(out, 'fcisolver'):
            if isinstance(getattr(out.fcisolver, 'ci', None), (list, tuple)):
                out.fcisolver.ci = [c.get() if hasattr(c, 'get') else numpy.asarray(c) for c in out.fcisolver.ci]
            elif hasattr(getattr(out.fcisolver, 'ci', None), 'get'):
                out.fcisolver.ci = out.fcisolver.ci.get()
        return out

    def state_average_(self, weights=(0.5, 0.5), wfnsym=None):
        from gpu4pyscf.mcscf import addons
        return addons.state_average_(self, weights, wfnsym)

    def state_average(self, weights=(0.5, 0.5), wfnsym=None):
        from gpu4pyscf.mcscf import addons
        return addons.state_average(self, weights, wfnsym)


class DFCASCI(_DFCAS, _CASCI):
    def __init__(self, mf_or_mol, ncas, nelecas, auxbasis=None, ncore=None,
                 with_df=None):
        _CASCI.__init__(self, mf_or_mol, ncas, nelecas, ncore)
        self.with_df = _get_with_df(self, auxbasis, with_df)


class DFCASSCF(_DFCAS, _CASSCF):
    def __init__(self, mf_or_mol, ncas, nelecas, auxbasis=None, ncore=None,
                 frozen=None, with_df=None):
        _CASSCF.__init__(self, mf_or_mol, ncas, nelecas, ncore, frozen)
        self.with_df = _get_with_df(self, auxbasis, with_df)

    def ao2mo(self, mo_coeff=None, hcore=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        ncore = self.ncore
        nocc = ncore + self.ncas
        mo_cas = mo_coeff[:, ncore:nocc]
        paaa = self.with_df.ao2mo(
            (mo_coeff, mo_cas, mo_cas, mo_cas), compact=False)
        paaa = paaa.reshape(mo_coeff.shape[1], self.ncas,
                            self.ncas, self.ncas)
        return _ERIS(self, mo_coeff, paaa, hcore)


def from_cpu(mc):
    if isinstance(mc.fcisolver, SpinPenaltyFCISolver):
        raise NotImplementedError('GPU FCI spin penalties are not implemented')
    cls = DFCASSCF if isinstance(mc, cpu_mc1step.CASSCF) else DFCASCI
    out = cls(mc._scf, mc.ncas, mc.nelecas,
              auxbasis=mc.with_df.auxbasis, ncore=mc.ncore)
    if isinstance(mc, (StateAverageMCSCFSolver, cpu_mcscf.addons.StateAverageMCSCFSolver)):
        from gpu4pyscf.mcscf import addons
        out = addons.state_average_(out, weights=mc.weights)
    for key, value in mc.__dict__.items():
        if key not in ('_scf', 'with_df', 'fcisolver'):
            if isinstance(value, numpy.ndarray):
                value = cupy.asarray(value)
            elif isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], numpy.ndarray):
                value = [cupy.asarray(v) for v in value]
            out.__dict__[key] = value
    out.fcisolver.__dict__.update(mc.fcisolver.__dict__)
    if hasattr(out.fcisolver, '_converged') and 'converged' in mc.fcisolver.__dict__:
        out.fcisolver._converged = mc.fcisolver.__dict__['converged']
    if isinstance(getattr(out.fcisolver, 'ci', None), (list, tuple)):
        out.fcisolver.ci = [cupy.asarray(c) for c in out.fcisolver.ci]
    elif isinstance(getattr(out.fcisolver, 'ci', None), numpy.ndarray):
        out.fcisolver.ci = cupy.asarray(out.fcisolver.ci)
    if isinstance(getattr(out, 'ci', None), (list, tuple)):
        out.ci = [cupy.asarray(c) for c in out.ci]
    elif isinstance(getattr(out, 'ci', None), numpy.ndarray):
        out.ci = cupy.asarray(out.ci)
    return out
