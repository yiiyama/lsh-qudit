"""Trotter step functions."""
from numbers import Number
import numpy as np
import scipy
import jax
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from .constants import BOSON_TRUNC
from .utils import op_matrix
from .hamiltonian_alltoall import (make_circuit, mass_term, electric_12_term, electric_3f_term,
                                   electric_3b_term, hopping_term, boundary_conditions)
from .matrices import (mass_hamiltonian, electric_12_hamiltonian, electric_3f_hamiltonian,
                       electric_3b_hamiltonian, hopping_hamiltonian, nl_bounds)


def trotter_step_circuit(
    num_sites: int,
    mass_mu: Number | ParameterExpression,
    interaction_x: Number | ParameterExpression,
    time_step: Number | ParameterExpression,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    with_barrier: bool = False,
    second_order: bool = False
) -> QuantumCircuit:
    full_circuit, _ = make_circuit(num_sites)

    if second_order:
        dt = time_step * 0.5
    else:
        dt = time_step

    # H_M
    full_circuit.compose(
        mass_term(num_sites, mass_mu, dt),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_E[1] + H_E[2]
    full_circuit.compose(
        electric_12_term(num_sites, dt, max_left_flux, max_right_flux),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_E[3] bosonic
    full_circuit.compose(
        electric_3b_term(num_sites, dt, max_left_flux, max_right_flux),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_I[1](r even)
    full_circuit.compose(
        hopping_term(num_sites, 0, 1, interaction_x, dt, max_left_flux, max_right_flux,
                     with_barrier=with_barrier),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_I[2](r odd)
    full_circuit.compose(
        hopping_term(num_sites, 1, 2, interaction_x, dt, max_left_flux, max_right_flux,
                     with_barrier=with_barrier),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_E[3] fermionc
    full_circuit.compose(
        electric_3f_term(num_sites, dt),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_I[1](r odd)
    full_circuit.compose(
        hopping_term(num_sites, 1, 1, interaction_x, dt, max_left_flux, max_right_flux,
                     with_barrier=with_barrier),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_I[2](r even)
    full_circuit.compose(
        hopping_term(num_sites, 0, 2, interaction_x, time_step, max_left_flux, max_right_flux,
                     with_barrier=with_barrier),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    if not second_order:
        return full_circuit

    # H_I[1](r odd)
    full_circuit.compose(
        hopping_term(num_sites, 1, 1, interaction_x, dt, max_left_flux, max_right_flux,
                     with_barrier=with_barrier),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_E[3] fermionc
    full_circuit.compose(
        electric_3f_term(num_sites, dt),
        inplace=True
    )

    # H_E[3] bosonic
    full_circuit.compose(
        electric_3b_term(num_sites, dt, max_left_flux, max_right_flux),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_I[2](r odd)
    full_circuit.compose(
        hopping_term(num_sites, 1, 2, interaction_x, dt, max_left_flux, max_right_flux,
                     with_barrier=with_barrier),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_I[1](r even)
    full_circuit.compose(
        hopping_term(num_sites, 0, 1, interaction_x, dt, max_left_flux, max_right_flux,
                     with_barrier=with_barrier),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_E[1] + H_E[2]
    full_circuit.compose(
        electric_12_term(num_sites, dt, max_left_flux, max_right_flux),
        inplace=True
    )
    if with_barrier:
        full_circuit.barrier()

    # H_M
    full_circuit.compose(
        mass_term(num_sites, mass_mu, dt),
        inplace=True
    )

    return full_circuit


def trotter_step_unitary(
    num_sites: int,
    time_step: Number,
    mass_mu: Number,
    interaction_x: Number,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    second_order: bool = False,
    npmod=np
) -> np.ndarray:
    if npmod is np:
        expm = scipy.linalg.expm
    else:
        expm = jax.scipy.linalg.expm

    conditions = boundary_conditions(num_sites, max_left_flux, max_right_flux)
    shape = ()
    for bc in conditions:
        shape = (nl_bounds(**bc)[1], 2, 2) + shape

    umat = npmod.eye(np.prod(shape), dtype=np.complex128)

    if second_order:
        dt = time_step * 0.5
    else:
        dt = time_step

    # HM
    hmat = mass_hamiltonian(num_sites, mass_mu, npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems = sum(((3 * site + 1, 3 * site) for site in list(range(num_sites))[::-1]), ())
    umat = op_matrix(umat_local, shape, subsystems, npmod=npmod) @ umat

    # HE[12]
    hmat = electric_12_hamiltonian(num_sites,
                                   max_left_flux=max_left_flux, max_right_flux=max_right_flux,
                                   npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems_r = ()
    for site in range(num_sites - 1):
        if shape[::-1][3 * site + 2] != 1:
            subsystems_r += (3 * site + 2,)
    umat = op_matrix(umat_local, shape, subsystems_r[::-1], npmod=npmod) @ umat

    # HI[1](r even)
    hmat = hopping_hamiltonian(num_sites, 0, 1, interaction_x,
                               max_left_flux=max_left_flux, max_right_flux=max_right_flux,
                               npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    umat = umat_local @ umat

    # HI[2](r odd)
    hmat = hopping_hamiltonian(num_sites, 1, 2, interaction_x,
                               max_left_flux=max_left_flux, max_right_flux=max_right_flux,
                               npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems = tuple(range(3, 3 * (num_sites - 1)))[::-1]
    umat = op_matrix(umat_local, shape, subsystems, npmod=npmod) @ umat

    # HE[3]
    hmat = electric_3b_hamiltonian(num_sites, max_left_flux=max_left_flux,
                                   max_right_flux=max_right_flux, npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems_r = ()
    for site in range(num_sites - 1):
        if shape[::-1][3 * site + 2] != 1:
            subsystems_r += tuple(range(3 * site, 3 * (site + 1)))
    umat = op_matrix(umat_local, shape, subsystems_r[::-1], npmod=npmod) @ umat

    hmat = electric_3f_hamiltonian(num_sites, npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems = sum(((3 * site + 1, 3 * site) for site in list(range(num_sites - 1))[::-1]), ())
    umat = op_matrix(umat_local, shape, subsystems, npmod=npmod) @ umat

    # HI[1](r odd)
    hmat = hopping_hamiltonian(num_sites, 1, 1, interaction_x, max_left_flux=max_left_flux,
                               max_right_flux=max_right_flux, npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems = tuple(range(3, 3 * (num_sites - 1)))[::-1]
    umat = op_matrix(umat_local, shape, subsystems, npmod=npmod) @ umat

    # HI[2](r even)
    hmat = hopping_hamiltonian(num_sites, 0, 2, interaction_x,
                               max_left_flux=max_left_flux, max_right_flux=max_right_flux,
                               npmod=npmod)
    umat_local = expm(-1.j * hmat * time_step)
    umat = umat_local @ umat

    if not second_order:
        return umat

    # HI[1](r odd)
    hmat = hopping_hamiltonian(num_sites, 1, 1, interaction_x, max_left_flux=max_left_flux,
                               max_right_flux=max_right_flux, npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems = tuple(range(3, 3 * (num_sites - 1)))[::-1]
    umat = op_matrix(umat_local, shape, subsystems, npmod=npmod) @ umat

    # HE[3]
    hmat = electric_3b_hamiltonian(num_sites, max_left_flux=max_left_flux,
                                   max_right_flux=max_right_flux, npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems_r = ()
    for site in range(num_sites - 1):
        if shape[::-1][3 * site + 2] != 1:
            subsystems_r += tuple(range(3 * site, 3 * (site + 1)))
    umat = op_matrix(umat_local, shape, subsystems_r[::-1], npmod=npmod) @ umat

    hmat = electric_3f_hamiltonian(num_sites, npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems = sum(((3 * site + 1, 3 * site) for site in list(range(num_sites - 1))[::-1]), ())
    umat = op_matrix(umat_local, shape, subsystems, npmod=npmod) @ umat

    # HI[2](r odd)
    hmat = hopping_hamiltonian(num_sites, 1, 2, interaction_x,
                               max_left_flux=max_left_flux, max_right_flux=max_right_flux,
                               npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems = tuple(range(3, 3 * (num_sites - 1)))[::-1]
    umat = op_matrix(umat_local, shape, subsystems, npmod=npmod) @ umat

    # HI[1](r even)
    hmat = hopping_hamiltonian(num_sites, 0, 1, interaction_x,
                               max_left_flux=max_left_flux, max_right_flux=max_right_flux,
                               npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    umat = umat_local @ umat

    # HE[12]
    hmat = electric_12_hamiltonian(num_sites,
                                   max_left_flux=max_left_flux, max_right_flux=max_right_flux,
                                   npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems_r = ()
    for site in range(num_sites - 1):
        if shape[::-1][3 * site + 2] != 1:
            subsystems_r += (3 * site + 2,)
    umat = op_matrix(umat_local, shape, subsystems_r[::-1], npmod=npmod) @ umat

    # HM
    hmat = mass_hamiltonian(num_sites, mass_mu, npmod=npmod)
    umat_local = expm(-1.j * hmat * dt)
    subsystems = sum(((3 * site + 1, 3 * site) for site in list(range(num_sites))[::-1]), ())
    umat = op_matrix(umat_local, shape, subsystems, npmod=npmod) @ umat

    return umat
