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
State-Averaged CASSCF (SA-CASSCF) Non-Adiabatic Coupling Vectors (NACVs) on GPU.

Calculates the first-order derivative coupling vector between two electronic states:
    d_{IJ}(R) = <Psi_I | ∇_R Psi_J>

Components:
1. Active Hellmann-Feynman term: Contracted with active transition 1-RDM and 2-RDM.
2. CSF derivative term (overlap derivative): Arises from differentiating atom-centered AOs.
3. Orbital response term: Z-vector solve for transition orbital gradient L^{IJ}.
4. Electron Translation Factors (ETFs): Enabled via use_etfs=True (default), cancelling
   the CSF term for Galilean-invariant Non-Adiabatic Molecular Dynamics (NAMD).
"""

import numpy as np
import cupy as cp
from pyscf import lib, gto
from gpu4pyscf.lib import logger
from gpu4pyscf.grad import rhf as rhf_grad
from gpu4pyscf.df import int3c2e
from gpu4pyscf.grad.sacasscf import make_general_fock, make_transition_fock, get_orbital_gradient, _to_cupy, _to_numpy

__all__ = ['NonAdiabaticCoupling', 'NonAdiabaticCouplings', 'as_scanner']


class NonAdiabaticCoupling(lib.StreamObject):
    """
    GPU Analytical Non-Adiabatic Coupling Vectors for SA-CASSCF.

    Args:
        mc: A converged StateAverageMCSCF object.
        state: Tuple (I, J) specifying the bra and ket states. Default is (0, 1).
        use_etfs: bool. If True (default), electron translation factors are applied,
                  cancelling the CSF overlap derivative term for Galilean invariance in NAMD.
        mult_ediff: bool. If True, returns h_{IJ} = (E_J - E_I) * d_{IJ}, which remains
                    finite and well-behaved at conical intersections. Default is False.
    """
    def __init__(self, mc, state=(0, 1), use_etfs=True, mult_ediff=False):
        self.base = mc
        self.mol = mc.mol
        self.state = tuple(state)
        self.use_etfs = bool(use_etfs)
        self.mult_ediff = bool(mult_ediff)
        self.atmlst = None
        self.verbose = mc.verbose
        self.stdout = mc.stdout
        self.de = None

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('=' * 72)
        log.info('SA-CASSCF Analytical Non-Adiabatic Coupling Vector (NACV) on GPU')
        log.info('Coupled States : %s', self.state)
        log.info('Use ETFs       : %s (cancels CSF basis derivative term for NAMD)', self.use_etfs)
        log.info('Multiply ΔE    : %s (returns h_IJ = ΔE * d_IJ)', self.mult_ediff)
        log.info('=' * 72)
        return self

    def kernel(self, state=None, atmlst=None):
        log = logger.new_logger(self)
        t0 = log.init_timer()
        if state is not None:
            self.state = tuple(state)
        if atmlst is None:
            atmlst = self.atmlst
        if atmlst is None:
            atmlst = range(self.mol.natm)

        root_i, root_j = self.state
        if root_i == root_j:
            log.info('Identical states (%d, %d): NACV is identically zero by anti-symmetry.',
                     root_i, root_j)
            self.de = np.zeros((self.mol.natm, 3))
            return self.de[atmlst]

        mc = self.base
        mol = self.mol
        mo_coeff = _to_cupy(mc.mo_coeff)
        ci = mc.ci
        ncore = mc.ncore
        ncas = mc.ncas
        nocc = ncore + ncas
        nmo = mo_coeff.shape[1]

        # 1. State energies and energy gap
        e_states = getattr(mc, 'e_states', None)
        if e_states is None:
            raise RuntimeError("SA-CASSCF energies e_states missing from mc object. Run mc.kernel() first.")
        e_i = float(e_states[root_i])
        e_j = float(e_states[root_j])
        de_ji = e_j - e_i

        log.info('Computing NACV between State %d (E = %.10f Ha) and State %d (E = %.10f Ha)',
                 root_i, e_i, root_j, e_j)
        log.info('Energy difference ΔE = %.10f Ha (%.4f eV)', de_ji, de_ji * 27.211386)

        if abs(de_ji) < 1e-12:
            log.warn('States are degenerate (ΔE < 1e-12 Ha). Pure derivative coupling d_IJ diverges; '
                     'consider mult_ediff=True to evaluate branching coupling h_IJ = ΔE * d_IJ.')

        # 2. Evaluate active transition 1-RDM and 2-RDM on GPU:
        # trans_d1 = <Psi_I | E_{tu} | Psi_J>
        # trans_d2 = <Psi_I | e_{tuvw} | Psi_J>
        trans_d1, trans_d2 = mc.fcisolver.trans_rdm12(
            ci[root_i], ci[root_j], norb=ncas, nelec=mc.nelecas
        )
        trans_d1 = _to_cupy(trans_d1)
        trans_d2 = _to_cupy(trans_d2)

        # Check for spin-forbidden or symmetry-forbidden transitions (<Psi_I|Psi_J> = 0)
        norm_td1 = float(cp.linalg.norm(trans_d1))
        norm_td2 = float(cp.linalg.norm(trans_d2))
        if norm_td1 < 1e-10 and norm_td2 < 1e-10:
            log.info('Transition RDMs for pair (%d, %d) vanish identically (spin-forbidden / symmetry). NACV is zero.',
                     root_i, root_j)
            self.de = np.zeros((mol.natm, 3))
            return self.de[atmlst]

        # 3. Active Hellmann-Feynman term:
        # Core orbitals are orthogonal between different CI roots, so core transition density is zero:
        # D_trans^{AO} = C_act @ trans_d1 @ C_act^T
        c_act = mo_coeff[:, ncore:nocc]
        dm_trans_ao = c_act @ trans_d1 @ c_act.T

        # 1-electron integral derivatives
        mf_grad = mc._scf.nuc_grad_method()
        h1 = _to_cupy(mf_grad.get_hcore(mol))
        s1 = _to_cupy(mf_grad.get_ovlp(mol))

        # Active 1-electron force (hermi=0 since transition density is not Hermitian)
        dh1e_act = _to_numpy(rhf_grad.contract_h1e_dm(mol, h1, dm_trans_ao, hermi=0))
        dh1e_nuc = _to_numpy(int3c2e.get_dh1e(mol, dm_trans_ao))
        if len(mol._ecpbas) > 0:
            ecp_atoms = np.unique(mol._ecpbas[:, gto.ATOM_OF])
            dh1e_nuc[ecp_atoms] += _to_numpy(rhf_grad.get_dh1e_ecp(mol, dm_trans_ao))
        f_act_1e = dh1e_act + dh1e_nuc

        # 4. CSF derivative term (basis overlap derivative):
        # d_IJ^{CSF} = 1/2 * sum_{μν} ∇S_{μν} [ C_act (γ_{ut} - γ_{tu}) C_act^T ]
        f_csf = np.zeros((mol.natm, 3))
        if not self.use_etfs:
            gamma_anti = 0.5 * (trans_d1 - trans_d1.T)
            dm_csf_ao = c_act @ gamma_anti @ c_act.T
            f_csf = -0.5 * _to_numpy(rhf_grad.contract_h1e_dm(mol, s1, dm_csf_ao, hermi=0))
            log.debug('CSF basis derivative norm: %.6e', np.linalg.norm(f_csf))
        else:
            log.debug('ETFs active: CSF basis derivative term cancelled.')

        # 5. Orbital response term:
        # Solve A^{SA} z^{IJ} = b^{IJ} where b^{IJ} = - 1 / (E_J - E_I) * L^{IJ}
        # with L^{IJ} = 2 * (F^{IJ} - (F^{IJ})^T)
        fock_ij = make_transition_fock(mc, mo_coeff, trans_d1, trans_d2)
        l_ca, l_cv, l_av = get_orbital_gradient(fock_ij, ncore, ncas, nmo)

        scale = -1.0 / de_ji if abs(de_ji) > 1e-12 else 0.0
        rhs_ca = scale * l_ca
        rhs_cv = scale * l_cv
        rhs_av = scale * l_av

        # Diagonal preconditioner from state-averaged Fock
        casdm1_sa, casdm2_sa = mc.fcisolver.make_rdm12(mc.ci, ncas, mc.nelecas)
        fock_sa = make_general_fock(mc, mo_coeff, casdm1_sa, casdm2_sa)
        f_diag = cp.diag(fock_sa)
        d_diag = cp.diag(_to_cupy(casdm1_sa))

        prec_ca = 2.0 * (f_diag[ncore:nocc][None, :] - f_diag[:ncore][:, None]) * (2.0 - d_diag[None, :])
        prec_cv = 4.0 * (f_diag[nocc:][None, :] - f_diag[:ncore][:, None])
        prec_av = 2.0 * (f_diag[nocc:][None, :] - f_diag[ncore:nocc][:, None]) * d_diag[:, None]

        prec_ca = cp.where(cp.abs(prec_ca) < 1e-4, cp.sign(prec_ca) * 1e-4 + 1e-6, prec_ca)
        prec_cv = cp.where(cp.abs(prec_cv) < 1e-4, cp.sign(prec_cv) * 1e-4 + 1e-6, prec_cv)
        prec_av = cp.where(cp.abs(prec_av) < 1e-4, cp.sign(prec_av) * 1e-4 + 1e-6, prec_av)

        z_ca = rhs_ca / prec_ca
        z_cv = rhs_cv / prec_cv
        z_av = rhs_av / prec_av

        z_resp = cp.zeros((nmo, nmo), dtype=cp.float64)
        z_resp[:ncore, ncore:nocc] = z_ca
        z_resp[ncore:nocc, :ncore] = -z_ca.T
        z_resp[:ncore, nocc:] = z_cv
        z_resp[nocc:, :ncore] = -z_cv.T
        z_resp[ncore:nocc, nocc:] = z_av
        z_resp[nocc:, ncore:nocc] = -z_av.T

        # Contract orbital response multiplier with 1-electron and overlap derivatives
        d_sa_mo = cp.zeros((nmo, nmo), dtype=cp.float64)
        d_sa_mo[:ncore, :ncore] = 2.0 * cp.eye(ncore)
        d_sa_mo[ncore:nocc, ncore:nocc] = _to_cupy(casdm1_sa)

        delta_d_mo = 0.5 * (z_resp @ d_sa_mo - d_sa_mo @ z_resp)
        dm_resp_ao = mo_coeff @ delta_d_mo @ mo_coeff.T

        dh1e_resp = _to_numpy(rhf_grad.contract_h1e_dm(mol, h1, dm_resp_ao, hermi=1)) + _to_numpy(int3c2e.get_dh1e(mol, dm_resp_ao))
        if len(mol._ecpbas) > 0:
            ecp_atoms = np.unique(mol._ecpbas[:, gto.ATOM_OF])
            dh1e_resp[ecp_atoms] += _to_numpy(rhf_grad.get_dh1e_ecp(mol, dm_resp_ao))
        w_resp_mo = 0.5 * (fock_sa @ z_resp + z_resp @ fock_sa)
        w_resp_ao = mo_coeff @ w_resp_mo @ mo_coeff.T
        ds_resp = _to_numpy(rhf_grad.contract_h1e_dm(mol, s1, w_resp_ao, hermi=0))

        f_resp = dh1e_resp - ds_resp

        # 6. Total Non-Adiabatic Coupling Vector:
        # d_IJ = 1 / (E_J - E_I) * f_active + f_csf + f_response
        inv_de = 1.0 / de_ji if abs(de_ji) > 1e-12 else 0.0
        d_ij = inv_de * f_act_1e + f_csf + f_resp

        if self.mult_ediff:
            # Return h_IJ = ΔE * d_IJ
            self.de = _to_numpy(d_ij * de_ji)
            log.info('Final scaled coupling h_IJ = ΔE * d_IJ Frobenius norm: %.6e',
                     np.linalg.norm(self.de))
        else:
            self.de = _to_numpy(d_ij)
            log.info('Final Non-Adiabatic Coupling Vector d_IJ Frobenius norm: %.6e Bohr^-1',
                     np.linalg.norm(self.de))

        log.timer('SA-CASSCF NACV calculation', *t0)
        return self.de[atmlst]

    def as_scanner(self):
        return as_scanner(self)


NonAdiabaticCouplings = NonAdiabaticCoupling


def as_scanner(nac_obj):
    """
    Return a NAC scanner for molecular dynamics and surface hopping.
    """
    if isinstance(nac_obj, lib.GradScanner):
        return nac_obj
    logger.info(nac_obj, 'Create scanner for %s', nac_obj.__class__)
    name = nac_obj.__class__.__name__ + '_NACScanner'
    return lib.set_class(SACASSCF_NACScanner(nac_obj),
                         (SACASSCF_NACScanner, nac_obj.__class__), name)


class SACASSCF_NACScanner(lib.GradScanner):
    def __init__(self, nac):
        super().__init__(nac)

    def __call__(self, mol_or_geom, **kwargs):
        if isinstance(mol_or_geom, gto.MoleBase):
            mol = mol_or_geom
        else:
            mol = self.mol.set_geom_(mol_or_geom, inplace=False)

        self.reset(mol)
        mc = self.base
        mc.reset(mol)
        mc.kernel()

        state = kwargs.get('state', getattr(self, 'state', (0, 1)))
        d_ij = self.kernel(state=state, **kwargs)
        return mc.e_states, d_ij
