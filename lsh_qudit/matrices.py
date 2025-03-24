"""Hamiltonian terms and corresponding unitaries represented in numpy matrices."""

import numpy as np
from .utils import op_matrix
from .hamiltonian import BOSON_TRUNC, BOSONIC_QUBITS, boundary_conditions


def mass_term_site_matrix(
    site: int,
    time_step: float,
    mass_mu: float
) -> np.ndarray:
    op = np.array([1., -1.]) * (-1 + 2 * (site % 2)) * mass_mu / 2. * time_step
    diags = op[:, None] + op[None, :]
    return np.diagflat(diags).astype(np.complex128)


def mass_term_matrix(
    num_sites: int,
    time_step: float,
    mass_mu: float,
    npmod=np
) -> np.ndarray:
    shape = (2,) * (2 * num_sites)
    dim = np.prod(shape)
    mat = npmod.zeros((dim, dim), dtype=np.complex128)
    for site in range(num_sites):
        mat += op_matrix(mass_term_site_matrix(site, time_step, mass_mu), shape,
                         (2 * site + 1, 2 * site), npmod=npmod)
    return mat


def electric_12_term_site_matrix(
    time_step: float,
    max_left_flux: int = -1,
    max_right_flux: int = -1
) -> np.ndarray:
    nmax, dim = _nl_bounds(max_left_flux, max_right_flux)

    if nmax == 0:
        return np.zeros((1, 1), dtype=np.complex128)

    nl = np.arange(dim)
    nl[nmax + 1:] = 0
    diags = (nl / 2. + nl * nl / 4.) * time_step
    return np.diagflat(diags).astype(np.complex128)


def electric_12_term_matrix(
    num_sites: int,
    time_step: float,
    max_left_flux: int = -1,
    max_right_flux: int = -1,
    npmod=np
) -> np.ndarray:
    conditions = boundary_conditions(num_sites, max_left_flux, max_right_flux)
    site_matrices = [
        electric_12_term_site_matrix(time_step, **bc) for bc in conditions[:-1]
    ]
    shape = tuple(m.shape[0] for m in site_matrices[::-1])
    dim = np.prod(shape)
    mat = npmod.zeros((dim, dim), dtype=np.complex128)
    for site, op in enumerate(site_matrices):
        mat += op_matrix(op, shape, site, npmod=npmod)
    return mat


def electric_3f_term_site_matrix(
    time_step: float
) -> np.ndarray:
    # 3/4 * (1 - n_i) * n_o
    nop = np.arange(2)
    diags = 0.75 * (1 - nop[None, :]) * nop[:, None] * time_step
    return np.diagflat(diags)


def electric_3f_term_matrix(
    num_sites: int,
    time_step: float,
    npmod=np
) -> np.ndarray:
    shape = (2,) * (2 * (num_sites - 1))
    dim = np.prod(shape)
    mat = npmod.zeros((dim, dim), dtype=np.complex128)
    for site in range(num_sites - 1):
        mat += op_matrix(electric_3f_term_site_matrix(time_step), shape,
                         (2 * site + 1, 2 * site), npmod=npmod)
    return mat


def electric_3b_term_site_matrix(
    time_step: float,
    max_left_flux: int = -1,
    max_right_flux: int = -1
) -> np.ndarray:
    nl_max, nl_dim = _nl_bounds(max_left_flux, max_right_flux)
    if nl_max == 0:
        return np.zeros((1, 1), dtype=np.complex128)
    diags = np.zeros((nl_dim, 2, 2), dtype=np.complex128)
    diags[:nl_max + 1, 1, 0] = np.arange(nl_max + 1) * 0.5 * time_step
    return np.diagflat(diags)


def electric_3b_term_matrix(
    num_sites: int,
    time_step: float,
    max_left_flux: int = -1,
    max_right_flux: int = -1,
    npmod=np
) -> np.ndarray:
    conditions = boundary_conditions(num_sites, max_left_flux, max_right_flux)
    site_matrices = [
        electric_3b_term_site_matrix(time_step, **bc) for bc in conditions[:-1]
    ]
    shape = tuple(m.shape[0] for m in site_matrices[::-1])
    dim = np.prod(shape)
    mat = npmod.zeros((dim, dim), dtype=np.complex128)
    for site, op in enumerate(site_matrices):
        mat += op_matrix(op, shape, site, npmod=npmod)
    return mat


def _nl_bounds(max_left_flux, max_right_flux):
    bounds = [BOSON_TRUNC - 1]
    if max_left_flux >= 0:
        bounds.append(max_left_flux)
    if max_right_flux >= 0:
        bounds.append(max_right_flux)
    nmax = min(bounds)

    if BOSONIC_QUBITS == 'qutrit' and nmax == 2:
        dim = 3
    else:
        dim = 2 ** np.ceil(np.log2(nmax + 1)).astype(int)

    return nmax, dim
