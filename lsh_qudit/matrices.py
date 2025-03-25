"""Hamiltonian terms and corresponding unitaries represented in numpy matrices."""
from numbers import Number
import numpy as np
from .utils import op_matrix
from .constants import (BOSON_TRUNC, BOSONIC_QUBITS, BDIM, cincrp, ocincrp, pauliz, sigmaplus,
                        sigmaminus)
from .hamiltonian import boundary_conditions


def mass_site_hamiltonian(
    site: int,
    time_step: Number,
    mass_mu: Number
) -> np.ndarray:
    op = np.array([1., -1.]) * (-1 + 2 * (site % 2)) * mass_mu / 2. * time_step
    diags = op[:, None] + op[None, :]
    return np.diagflat(diags).astype(np.complex128)


def mass_hamiltonian(
    num_sites: int,
    time_step: Number,
    mass_mu: Number,
    npmod=np
) -> np.ndarray:
    shape = (2,) * (2 * num_sites)
    dim = np.prod(shape)
    mat = npmod.zeros((dim, dim), dtype=np.complex128)
    for site in range(num_sites):
        mat += op_matrix(mass_site_hamiltonian(site, time_step, mass_mu), shape,
                         (2 * site + 1, 2 * site), npmod=npmod)
    return mat


def electric_12_site_hamiltonian(
    time_step: Number,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1
) -> np.ndarray:
    nmax, dim = nl_bounds(max_left_flux, max_right_flux)

    if nmax == 0:
        return np.zeros((1, 1), dtype=np.complex128)

    nl = np.arange(dim)
    nl[nmax + 1:] = 0
    diags = (nl / 2. + nl * nl / 4.) * time_step
    return np.diagflat(diags).astype(np.complex128)


def electric_12_hamiltonian(
    num_sites: int,
    time_step: Number,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    npmod=np
) -> np.ndarray:
    conditions = boundary_conditions(num_sites, max_left_flux, max_right_flux)
    site_matrices = [
        electric_12_site_hamiltonian(time_step, **bc) for bc in conditions[:-1]
    ]
    shape = tuple(m.shape[0] for m in site_matrices[::-1])
    dim = np.prod(shape)
    mat = npmod.zeros((dim, dim), dtype=np.complex128)
    for site, op in enumerate(site_matrices):
        mat += op_matrix(op, shape, site, npmod=npmod)
    return mat


def electric_3f_site_hamiltonian(
    time_step: Number
) -> np.ndarray:
    # 3/4 * (1 - n_i) * n_o
    nop = np.arange(2)
    diags = 0.75 * (1 - nop[None, :]) * nop[:, None] * time_step
    return np.diagflat(diags)


def electric_3f_hamiltonian(
    num_sites: int,
    time_step: Number,
    npmod=np
) -> np.ndarray:
    shape = (2,) * (2 * (num_sites - 1))
    dim = np.prod(shape)
    mat = npmod.zeros((dim, dim), dtype=np.complex128)
    for site in range(num_sites - 1):
        mat += op_matrix(electric_3f_site_hamiltonian(time_step), shape,
                         (2 * site + 1, 2 * site), npmod=npmod)
    return mat


def electric_3b_site_hamiltonian(
    time_step: Number,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1
) -> np.ndarray:
    nl_max, nl_dim = nl_bounds(max_left_flux, max_right_flux)
    if nl_max == 0:
        return np.zeros((1, 1), dtype=np.complex128)
    diags = np.zeros((nl_dim, 2, 2), dtype=np.complex128)
    diags[:nl_max + 1, 1, 0] = np.arange(nl_max + 1) * 0.5 * time_step
    return np.diagflat(diags)


def electric_3b_hamiltonian(
    num_sites: int,
    time_step: Number,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    npmod=np
) -> np.ndarray:
    conditions = boundary_conditions(num_sites, max_left_flux, max_right_flux)
    site_matrices = [
        electric_3b_site_hamiltonian(time_step, **bc) for bc in conditions[:-1]
    ]
    shape = tuple(m.shape[0] for m in site_matrices[::-1])
    dim = np.prod(shape)
    mat = npmod.zeros((dim, dim), dtype=np.complex128)
    for site, op in enumerate(site_matrices):
        mat += op_matrix(op, shape, site, npmod=npmod)
    return mat


def hopping_site_hamiltonian(
    term_type: int,
    time_step: Number,
    interaction_x: Number,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
) -> np.ndarray:
    nl_max_l, nl_dim_l = nl_bounds(max_left_flux, max_right_flux + 1)
    nl_max_r, nl_dim_r = nl_bounds(max_left_flux + 1, max_right_flux)
    cincrp_tl = cincrp.reshape((2, BDIM, 2, BDIM))[:, :nl_dim_l, :, :nl_dim_l]
    cincrp_tl = cincrp_tl.reshape((2 * nl_dim_l, 2 * nl_dim_l))
    cincrp_tr = cincrp.reshape((2, BDIM, 2, BDIM))[:, :nl_dim_r, :, :nl_dim_r]
    cincrp_tr = cincrp_tr.reshape((2 * nl_dim_r, 2 * nl_dim_r))
    ocincrp_tl = ocincrp.reshape((2, BDIM, 2, BDIM))[:, :nl_dim_l, :, :nl_dim_l]
    ocincrp_tl = ocincrp_tl.reshape((2 * nl_dim_l, 2 * nl_dim_l))
    ocincrp_tr = ocincrp.reshape((2, BDIM, 2, BDIM))[:, :nl_dim_r, :, :nl_dim_r]
    ocincrp_tr = ocincrp_tr.reshape((2 * nl_dim_r, 2 * nl_dim_r))
    nl_l = np.zeros(nl_dim_l)
    nl_l[:nl_max_l + 1] = np.arange(nl_max_l + 1)
    diag_fn_l = np.sqrt((nl_l[:, None, None] + np.arange(1, 3)[None, :, None])
                        / (nl_l[:, None, None] + np.arange(1, 3)[None, None, :]))
    nl_r = np.zeros(nl_dim_r)
    nl_r[:nl_max_r + 1] = np.arange(nl_max_r + 1)
    diag_fn_r = np.sqrt((nl_r[:, None, None] + np.arange(1, 3)[None, :, None])
                        / (nl_r[:, None, None] + np.arange(1, 3)[None, None, :]))
    shape = (nl_dim_r, 2, 2, nl_dim_l, 2, 2)

    if term_type == 1:
        mat = op_matrix(np.diagflat(diag_fn_l), shape, (2, 1, 4))
        mat = op_matrix(cincrp_tr, shape, (4, 5)) @ mat
        mat = op_matrix(ocincrp_tl, shape, (1, 2)) @ mat
        mat = op_matrix(sigmaminus, shape, 3) @ mat
        mat = op_matrix(pauliz, shape, 1) @ mat
        mat = op_matrix(sigmaplus, shape, 0) @ mat
    else:
        mat = op_matrix(np.diagflat(diag_fn_r), shape, (5, 3, 0))
        mat = op_matrix(cincrp_tl, shape, (0, 2)) @ mat
        mat = op_matrix(ocincrp_tr, shape, (3, 5)) @ mat
        mat = op_matrix(sigmaminus, shape, 1) @ mat
        mat = op_matrix(pauliz, shape, 3) @ mat
        mat = op_matrix(sigmaplus, shape, 4) @ mat
    mat += mat.conjugate().T
    mat *= interaction_x * time_step
    return mat


def hopping_hamiltonian(
    num_sites: int,
    site_parity: int,
    term_type: int,
    time_step: Number,
    interaction_x: Number,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    npmod=np
) -> np.ndarray:
    conditions = boundary_conditions(num_sites, max_left_flux, max_right_flux, num_local=2)
    site_matrices = [
        hopping_site_hamiltonian(term_type, time_step, interaction_x, **conditions[site])
        for site in range(site_parity, num_sites - 1, 2)
    ]
    shape = tuple(m.shape[0] for m in site_matrices[::-1])
    dim = np.prod(shape)
    mat = npmod.zeros((dim, dim), dtype=np.complex128)
    for site, op in enumerate(site_matrices):
        mat += op_matrix(op, shape, site, npmod=npmod)
    return mat


def nl_bounds(
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1
) -> tuple[int, int]:
    bounds = [BOSON_TRUNC - 1]
    if max_left_flux < BOSON_TRUNC - 1:
        bounds.append(max_left_flux)
    if max_right_flux < BOSON_TRUNC - 1:
        bounds.append(max_right_flux)
    nmax = min(bounds)

    if BOSONIC_QUBITS == 'qutrit' and nmax == 2:
        dim = 3
    else:
        dim = 2 ** np.ceil(np.log2(nmax + 1)).astype(int)

    return nmax, dim
