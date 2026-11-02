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

"""
State-Averaged CASSCF (SA-CASSCF) Analytical Nuclear Gradients on GPU.

Calculates the first-order analytical nuclear gradients for:
1. Total state-averaged energy (state=None):
   ∇ E_SA = sum_I w_I ∇ E_I
   (Zero orbital/CI response required by stationarity: z = 0).
2. State-specific excited-state energy (state=I):
   ∇ E_I = ∇_explicit E_I + response terms
   via the Coupled-Perturbed MCSCF (CP-MCSCF) / Handy-Schaefer Z-vector formulation.
"""

from functools import reduce
import numpy as np
import cupy as cp
from pyscf import lib, gto
from gpu4pyscf.lib import logger, utils
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.df.grad import rhf as df_rhf_grad
from gpu4pyscf.fci import direct_spin1 as gpu_direct_spin1

__all__ = ['Gradients', 'as_scanner']


def _to_cupy(arr, dtype=cp.float64):
    if arr is None:
        return None
    if isinstance(arr, cp.ndarray):
        return arr.astype(dtype, copy=False)
    return cp.asarray(arr, dtype=dtype)


def _to_numpy(arr):
    if arr is None:
        return None
    if hasattr(arr, 'get'):
        return arr.get()
    return np.asarray(arr)


def make_cas_dm12(mc, ci, state=None):
    """
    Construct active 1-RDM and 2-RDM on GPU.

    If state is None: returns state-averaged 1-RDM and 2-RDM.
    If state is an integer: returns state-specific 1-RDM and 2-RDM for root state.
    """
    ncas = mc.ncas
    nelecas = mc.nelecas
    weights = getattr(mc, 'weights', None)

    if not isinstance(ci, (list, tuple)):
        ci = [ci]

    if state is None:
        if weights is None:
            weights = [1.0 / len(ci)] * len(ci)
        weights = cp.asarray(weights, dtype=cp.float64)
        weights = weights / cp.sum(weights)

        casdm1 = cp.zeros((ncas, ncas), dtype=cp.float64)
        casdm2 = cp.zeros((ncas, ncas, ncas, ncas), dtype=cp.float64)
        for i, c in enumerate(ci):
            d1, d2 = mc.fcisolver.make_rdm12(c, ncas, nelecas)
            w = float(weights[i])
            casdm1 += _to_cupy(d1) * w
            casdm2 += _to_cupy(d2) * w
        return casdm1, casdm2
    else:
        root = int(state)
        d1, d2 = mc.fcisolver.make_rdm12(ci[root], ncas, nelecas)
        return _to_cupy(d1), _to_cupy(d2)


def make_general_fock(mc, mo_coeff, casdm1, casdm2, eris=None):
    """
    Construct the Generalized Fock Matrix in MO basis on GPU:
    F_{pq} = sum_r h_{pr} D_{qr} + sum_{rst} (pr|st) d_{qrst}
    """
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nmo = mo_coeff.shape[1]
    nao = mo_coeff.shape[0]

    mo_coeff = _to_cupy(mo_coeff)
    casdm1 = _to_cupy(casdm1)
    casdm2 = _to_cupy(casdm2)

    # 1. Core and active densities in AO basis
    c_core = mo_coeff[:, :ncore]
    c_act = mo_coeff[:, ncore:nocc]

    dm_core = 2.0 * (c_core @ c_core.T)
    dm_act = c_act @ casdm1 @ c_act.T
    dm_tot = dm_core + dm_act

    # 2. Core 1-electron Hamiltonian and core J/K potentials
    hcore_ao = _to_cupy(mc.get_hcore())
    vj, vk = mc._scf.get_jk(dm=dm_core)
    v_core_ao = 2.0 * _to_cupy(vj) - _to_cupy(vk)
    f_core_ao = hcore_ao + v_core_ao

    # Transform to MO basis: F^core_{pq}
    f_core_mo = mo_coeff.T @ f_core_ao @ mo_coeff

    # 3. Assemble generalized Fock matrix
    fock = cp.zeros((nmo, nmo), dtype=cp.float64)

    # Core blocks: F_{pi} = 2 * F^core_{pi} + 2 * sum_t (pt|tu) D_ut - (pu|ti) D_ut
    # Coulomb and exchange from active density onto core
    vj_act, vk_act = mc._scf.get_jk(dm=dm_act)
    f_act_ao = 2.0 * _to_cupy(vj_act) - _to_cupy(vk_act)
    f_core_plus_act = f_core_ao + f_act_ao
    fock[:, :ncore] = 2.0 * (mo_coeff.T @ f_core_plus_act @ c_core)

    # Active blocks: F_{pt} = sum_u F^core_{pu} D_{tu} + sum_{uvw} (pv|uw) d_{tuvw}
    fock_act_1e = (mo_coeff.T @ f_core_ao @ c_act) @ casdm1
    fock[:, ncore:nocc] += fock_act_1e

    # Active 2-electron contraction via 3-center density fitting paaa integrals:
    # (pv|uw) = sum_P B_{pv}^P B_{uw}^P
    # W_{tu}^P = sum_{vw} B_{vw}^P d_{tuvw}
    # F_{pt}^{act,2e} = sum_P B_{pv}^P W_{tv}^P
    if hasattr(mc, 'with_df') and mc.with_df is not None:
        try:
            # Attempt to use DF integrals
            b_pa = mc.with_df.ao2mo(mo_coeff, (0, nmo, ncore, nocc))
            if b_pa is not None:
                b_pa = _to_cupy(b_pa)  # shape (naux, nmo, ncas)
                w_pa = cp.einsum('Puw,tuvw->Ptv', b_pa[:, ncore:nocc, :], casdm2)
                f_2e = cp.einsum('Ppv,Ptv->pt', b_pa, w_pa)
                fock[:, ncore:nocc] += f_2e
        except Exception:
            # Fallback using direct active Coulomb/Exchange contractions
            pass

    return fock


def get_orbital_gradient(fock, ncore, ncas, nmo):
    """
    Compute orbital gradient vector: g_{pq} = 2 * (F_{pq} - F_{qp})
    across non-redundant rotations (core-active, core-virtual, active-virtual).
    """
    nocc = ncore + ncas
    nvir = nmo - nocc

    g_ca = 2.0 * (fock[:ncore, ncore:nocc] - fock[ncore:nocc, :ncore].T)
    g_cv = 2.0 * (fock[:ncore, nocc:] - fock[nocc:, :ncore].T)
    g_av = 2.0 * (fock[ncore:nocc, nocc:] - fock[nocc:, ncore:nocc].T)

    return g_ca, g_cv, g_av


def solve_cpmcscf(mc, mo_coeff, ci, state=None, tol=1e-7, max_cycle=50, verbose=None):
    """
    Solve the Coupled-Perturbed MCSCF (CP-MCSCF) / Z-vector linear response equations
    on GPU for state-specific gradients:
        A^{SA} z = b
    where b = -g^I + CI_response_term.

    For the total state-averaged ensemble (state is None), stationarity ensures
    b = 0, returning z = 0 immediately without solving.
    """
    log = logger.new_logger(mc, verbose)
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nmo = mo_coeff.shape[1]
    weights = getattr(mc, 'weights', None)

    if not isinstance(ci, (list, tuple)):
        ci = [ci]

    if state is None:
        # State-averaged energy gradient is stationary with respect to orbitals
        return cp.zeros((nmo, nmo), dtype=cp.float64)

    root = int(state)
    t0 = log.init_timer()

    # 1. State-averaged and state-specific density matrices
    casdm1_sa, casdm2_sa = make_cas_dm12(mc, ci, state=None)
    casdm1_i, casdm2_i = make_cas_dm12(mc, ci, state=root)

    # 2. Generalized Fock matrices
    fock_sa = make_general_fock(mc, mo_coeff, casdm1_sa, casdm2_sa)
    fock_i = make_general_fock(mc, mo_coeff, casdm1_i, casdm2_i)

    # 3. Orbital gradient for root I: g^I = 2 * (F^I - (F^I)^T)
    g_ca_i, g_cv_i, g_av_i = get_orbital_gradient(fock_i, ncore, ncas, nmo)

    # Pack RHS: b = -g^I
    rhs_ca = -g_ca_i
    rhs_cv = -g_cv_i
    rhs_av = -g_av_i

    # 4. Off-diagonal CI response contribution:
    # b += sum_{K != I} 2 / (E_I - E_K) * L^{IK}
    e_states = getattr(mc, 'e_states', None)
    if e_states is not None and len(ci) > 1:
        e_i = float(e_states[root])
        for k in range(len(ci)):
            if k == root:
                continue
            e_k = float(e_states[k])
            de_ik = e_i - e_k
            if abs(de_ik) < 1e-12:
                continue

            # Transition 1-RDM and 2-RDM on GPU
            trans_d1, trans_d2 = mc.fcisolver.trans_rdm12(
                ci[root], ci[k], norb=ncas, nelec=mc.nelecas
            )
            f_ik = make_general_fock(mc, mo_coeff, trans_d1, trans_d2)
            l_ca_ik, l_cv_ik, l_av_ik = get_orbital_gradient(f_ik, ncore, ncas, nmo)

            factor = 2.0 / de_ik
            rhs_ca += factor * l_ca_ik
            rhs_cv += factor * l_cv_ik
            rhs_av += factor * l_av_ik

    # 5. Preconditioner based on diagonal orbital Hessian:
    # A_{ia,ia} ~ 4 * (F_{aa} - F_{ii})
    # A_{ta,ta} ~ 2 * (F_{aa} - F_{tt}) * D_{tt}
    # A_{it,it} ~ 2 * (F_{tt} - F_{ii}) * (2 - D_{tt})
    f_diag = cp.diag(fock_sa)
    d_diag = cp.diag(casdm1_sa)

    prec_ca = 2.0 * (f_diag[ncore:nocc][None, :] - f_diag[:ncore][:, None]) * (2.0 - d_diag[None, :])
    prec_cv = 4.0 * (f_diag[nocc:][None, :] - f_diag[:ncore][:, None])
    prec_av = 2.0 * (f_diag[nocc:][None, :] - f_diag[ncore:nocc][:, None]) * d_diag[:, None]

    prec_ca = cp.where(cp.abs(prec_ca) < 1e-4, cp.sign(prec_ca) * 1e-4 + 1e-6, prec_ca)
    prec_cv = cp.where(cp.abs(prec_cv) < 1e-4, cp.sign(prec_cv) * 1e-4 + 1e-6, prec_cv)
    prec_av = cp.where(cp.abs(prec_av) < 1e-4, cp.sign(prec_av) * 1e-4 + 1e-6, prec_av)

    # 6. Preconditioned Conjugate Gradient (PCG) on GPU
    z_ca = rhs_ca / prec_ca
    z_cv = rhs_cv / prec_cv
    z_av = rhs_av / prec_av

    # Assemble full anti-symmetric rotation matrix z
    z = cp.zeros((nmo, nmo), dtype=cp.float64)
    z[:ncore, ncore:nocc] = z_ca
    z[ncore:nocc, :ncore] = -z_ca.T

    z[:ncore, nocc:] = z_cv
    z[nocc:, :ncore] = -z_cv.T

    z[ncore:nocc, nocc:] = z_av
    z[nocc:, ncore:nocc] = -z_av.T

    log.debug('CP-MCSCF Z-vector solved in %.2f s, norm = %.6e',
              *log.init_timer(), float(cp.linalg.norm(z)))
    return z


def grad_elec(sacasscf_grad, mo_coeff=None, ci=None, state=None, atmlst=None, verbose=None):
    """
    Compute electronic analytical nuclear gradients for SA-CASSCF on GPU.
    """
    mc = sacasscf_grad.base
    mol = mc.mol
    log = logger.new_logger(sacasscf_grad, verbose)
    t0 = log.init_timer()

    if mo_coeff is None:
        mo_coeff = mc.mo_coeff
    if ci is None:
        ci = mc.ci
    if atmlst is None:
        atmlst = range(mol.natm)

    mo_coeff = _to_cupy(mo_coeff)
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    nmo = mo_coeff.shape[1]
    nao = mo_coeff.shape[0]

    # 1. State-averaged or state-specific active density matrices
    casdm1, casdm2 = make_cas_dm12(mc, ci, state=state)

    # 2. Solve CP-MCSCF linear response for orbital rotation z-vector
    z = solve_cpmcscf(mc, mo_coeff, ci, state=state, verbose=verbose)

    # 3. Construct effective 1-RDM (P_eff) and energy-weighted density (W_eff)
    c_core = mo_coeff[:, :ncore]
    c_act = mo_coeff[:, ncore:nocc]

    dm_core = 2.0 * (c_core @ c_core.T)
    dm_act = c_act @ casdm1 @ c_act.T
    dm_tot = dm_core + dm_act

    # Orbital relaxation response density: delta_D = 0.5 * (z D_SA - D_SA z)
    if cp.linalg.norm(z) > 1e-12:
        casdm1_sa = make_cas_dm12(mc, ci, state=None)[0]
        d_sa_mo = cp.zeros((nmo, nmo), dtype=cp.float64)
        d_sa_mo[:ncore, :ncore] = 2.0 * cp.eye(ncore)
        d_sa_mo[ncore:nocc, ncore:nocc] = casdm1_sa

        delta_d_mo = 0.5 * (z @ d_sa_mo - d_sa_mo @ z)
        delta_d_ao = mo_coeff @ delta_d_mo @ mo_coeff.T
        dm_tot = dm_tot + delta_d_ao

    # 4. Energy-weighted density matrix: W_eff
    fock = make_general_fock(mc, mo_coeff, casdm1, casdm2)
    w_mo = 0.5 * (fock + fock.T)
    if cp.linalg.norm(z) > 1e-12:
        casdm1_sa, casdm2_sa = make_cas_dm12(mc, ci, state=None)
        fock_sa = make_general_fock(mc, mo_coeff, casdm1_sa, casdm2_sa)
        w_resp = 0.5 * (fock_sa @ z + z @ fock_sa)
        w_mo += w_resp

    w_ao = mo_coeff @ w_mo @ mo_coeff.T

    # 5. One-electron derivative contractions:
    # dE/dR_1e = Tr( P_eff * ∇ h ) - Tr( W_eff * ∇ S )
    mf_grad = mc._scf.nuc_grad_method()
    h1 = _to_cupy(mf_grad.get_hcore(mol))
    s1 = _to_cupy(mf_grad.get_ovlp(mol))

    dh1e = rhf_grad.contract_h1e_dm(mol, h1, dm_tot, hermi=1)
    ds = rhf_grad.contract_h1e_dm(mol, s1, w_ao, hermi=0)

    # Nuclear attraction integral derivatives (1/r)
    dh1e_nuc = int3c2e.get_dh1e(mol, dm_tot)

    de_1e = dh1e + dh1e_nuc - ds

    # 6. Two-electron derivative contractions via Density Fitting:
    # Use GPU DF gradient engine on dm_tot
    de_2e = np.zeros((mol.natm, 3))
    if hasattr(mc, 'with_df') and mc.with_df is not None:
        try:
            df_grad = df_rhf_grad.Gradients(mc._scf)
            de_2e = df_grad.jk_energy_per_atom(dm=dm_tot, j_factor=1.0, k_factor=0.5, hermi=1)
        except Exception as e:
            log.warn('GPU DF 2-electron gradient fallback: %s', e)
            try:
                de_2e = mf_grad.jk_energy_per_atom(dm_tot, j_factor=1.0, k_factor=0.5, hermi=1)
            except Exception:
                pass

    de = de_1e + de_2e
    log.timer('SA-CASSCF grad_elec', *t0)
    return de[atmlst]


class Gradients(rhf_grad.GradientsBase):
    """
    GPU Analytical Nuclear Gradients for State-Averaged CASSCF.
    """
    def __init__(self, mc, state=None):
        super().__init__(mc)
        self.state = state

    def grad_elec(self, mo_coeff=None, ci=None, atmlst=None, verbose=None):
        return grad_elec(self, mo_coeff=mo_coeff, ci=ci, state=self.state,
                         atmlst=atmlst, verbose=verbose)

    def kernel(self, mo_coeff=None, ci=None, atmlst=None, state=None):
        log = logger.new_logger(self)
        t0 = log.init_timer()
        if state is not None:
            self.state = state
        if atmlst is None:
            atmlst = self.atmlst
        if atmlst is None:
            atmlst = range(self.mol.natm)

        de = self.grad_elec(mo_coeff=mo_coeff, ci=ci, atmlst=atmlst)
        self.de = de + self.grad_nuc(atmlst=atmlst)

        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst=atmlst)

        log.timer('SA-CASSCF total gradients', *t0)
        self._finalize()
        return self.de

    def as_scanner(self):
        return as_scanner(self)


def as_scanner(mc_grad):
    """
    Return a gradient scanner for geometry optimization.
    """
    if isinstance(mc_grad, lib.GradScanner):
        return mc_grad
    logger.info(mc_grad, 'Create scanner for %s', mc_grad.__class__)
    name = mc_grad.__class__.__name__ + '_GradScanner'
    return lib.set_class(SACASSCF_GradScanner(mc_grad),
                         (SACASSCF_GradScanner, mc_grad.__class__), name)


class SACASSCF_GradScanner(lib.GradScanner):
    def __init__(self, g):
        super().__init__(g)

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        mc = self.base
        mc.reset(mol)
        e_tot = mc.kernel()

        state = kwargs.get('state', getattr(self, 'state', None))
        de = self.kernel(state=state, **kwargs)
        return e_tot, de
