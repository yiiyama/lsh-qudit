"""Functions to manage AGL and boundary conditions."""
import numpy as np
from .constants import BOSON_TRUNC


def physical_states(
    num_sites: int = 1,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    as_multi: bool = False
) -> np.ndarray:
    """Returns an array of AGL-satisfying states with optional boundary conditions.

    When as_multi=True, a 2-dimensional array is returned with the inner dimension corresponding
    to occupation numbers in the (i, o, l) order in the increasing site number.
    """
    shape = (2, 2, BOSON_TRUNC) * num_sites
    states = np.array(np.unravel_index(np.arange(np.prod(shape)), shape)).T
    agl_mask = np.ones(states.shape[:1], dtype=bool)
    for iconn in range(num_sites - 1):
        il = iconn * 3
        agl_mask &= np.equal((1 - states[:, il + 0]) * states[:, il + 1] + states[:, il + 2],
                             states[:, il + 3] * (1 - states[:, il + 4]) + states[:, il + 5])
    states = states[agl_mask]
    if max_left_flux < BOSON_TRUNC - 1:
        mask = np.zeros(states.shape[0], dtype=bool)
        for val in range(max_left_flux + 1):
            mask |= np.equal(states[:, 0] * (1 - states[:, 1]) + states[:, 2], val)
        states = states[mask]
    if max_right_flux < BOSON_TRUNC - 1:
        mask = np.zeros(states.shape[0], dtype=bool)
        for val in range(max_right_flux + 1):
            mask |= np.equal((1 - states[:, -3]) * states[:, -2] + states[:, -1], val)
        states = states[mask]

    if as_multi:
        return states
    return np.sum(states * np.cumprod((1,) + shape[-1:0:-1])[None, ::-1], axis=1)


def boundary_conditions(
    num_sites: int,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    num_local: int = 1
) -> list[dict[str, int]]:
    """Translate the lattice-wide BC to site-level BCs."""
    conditions = [{} for _ in range(num_sites - num_local + 1)]
    if max_left_flux < BOSON_TRUNC - 1:
        for lsite in range(min(num_sites, BOSON_TRUNC - 1 - max_left_flux)):
            conditions[lsite]['max_left_flux'] = max_left_flux + lsite
    if max_right_flux < BOSON_TRUNC - 1:
        for rsite in range(max(0, num_sites - BOSON_TRUNC + 1 + max_right_flux), num_sites):
            conditions[rsite - num_local + 1]['max_right_flux'] = (max_right_flux
                                                                   + (num_sites - rsite - 1))
    return conditions
