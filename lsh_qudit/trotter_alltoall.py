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
    time_step: Number | ParameterExpression,
    mass_mu: Number | ParameterExpression,
    interaction_x: Number | ParameterExpression,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    with_barrier: bool = False,
    second_order: bool = False
) -> QuantumCircuit:
    circuit, _ = make_circuit(num_sites)

    if second_order:
        dt = time_step * 0.5
    else:
        dt = time_step

    def compose(term_fn, *args, **kwargs):
        _dt = kwargs.pop('_dt', dt)
        circuit.compose(
            term_fn(num_sites, _dt, *args, **kwargs),
            inplace=True
        )
        if with_barrier:
            circuit.barrier()

    # H_M
    compose(mass_term, mass_mu)
    # H_E[1] + H_E[2]
    compose(electric_12_term, max_left_flux, max_right_flux)
    # H_E[3] fermionc
    compose(electric_3f_term)
    # H_E[3] bosonic
    compose(electric_3b_term, max_left_flux, max_right_flux)
    # H_I[1](r even)
    compose(hopping_term, 0, 1, interaction_x, max_left_flux, max_right_flux,
            with_barrier=with_barrier)
    # H_I[2](r odd)
    compose(hopping_term, 1, 2, interaction_x, max_left_flux, max_right_flux,
            with_barrier=with_barrier)
    # H_I[1](r odd)
    compose(hopping_term, 1, 1, interaction_x, max_left_flux, max_right_flux,
            with_barrier=with_barrier)
    # H_I[2](r even)
    compose(hopping_term, 0, 2, interaction_x, max_left_flux, max_right_flux,
            with_barrier=with_barrier, _dt=time_step)

    if not second_order:
        return circuit

    # H_I[1](r odd)
    compose(hopping_term, 1, 1, interaction_x, max_left_flux, max_right_flux,
            with_barrier=with_barrier)
    # H_I[2](r odd)
    compose(hopping_term, 1, 2, interaction_x, max_left_flux, max_right_flux,
            with_barrier=with_barrier)
    # H_I[1](r even)
    compose(hopping_term, 0, 1, interaction_x, max_left_flux, max_right_flux,
            with_barrier=with_barrier)
    # H_E[3] bosonic
    compose(electric_3b_term, max_left_flux, max_right_flux)
    # H_E[3] fermionc
    compose(electric_3f_term)
    # H_E[1] + H_E[2]
    compose(electric_12_term, max_left_flux, max_right_flux)
    # H_M
    compose(mass_term, mass_mu)

    return circuit


def trotter_step_unitary(
    num_sites: int,
    mass_mu: float,
    interaction_x: float,
    time_step: float,
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

    def compose(hmat_fn, qubits, *args, **kwargs):
        nonlocal umat
        _dt = kwargs.pop('_dt', dt)
        hmat = hmat_fn(num_sites, *args, npmod=npmod, **kwargs)
        umat_local = expm(-1.j * hmat * _dt)
        if qubits:
            qubits = tuple(sorted(qubits, reverse=True))
            umat_local = op_matrix(umat_local, shape, qubits, npmod=npmod)
        umat = umat_local @ umat

    # H_M
    qubits = list(range(0, 3 * num_sites, 3)) + list(range(1, 3 * num_sites, 3))
    compose(mass_hamiltonian, qubits, mass_mu)
    # H_E[1] + H_E[2]
    qubits = list(range(2, 3 * num_sites - 3, 3))
    compose(electric_12_hamiltonian, qubits,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux)
    # H_E[3] fermionic
    qubits = list(range(0, 3 * num_sites - 3, 3)) + list(range(1, 3 * num_sites - 3, 3))
    compose(electric_3f_hamiltonian, qubits)
    # H_E[3] bosonic
    qubits = []
    for site in range(num_sites - 1):
        if shape[::-1][3 * site + 2] != 1:
            qubits += list(range(3 * site, 3 * (site + 1)))
    compose(electric_3b_hamiltonian, qubits,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux)
    # H_I[1](r even)
    compose(hopping_hamiltonian, None, 0, 1, interaction_x,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux)
    # H_I[2](r odd)
    qubits = list(range(3, 3 * (num_sites - 1)))
    compose(hopping_hamiltonian, qubits, 1, 2, interaction_x,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux)
    # H_I[1](r odd)
    qubits = list(range(3, 3 * (num_sites - 1)))
    compose(hopping_hamiltonian, qubits, 1, 1, interaction_x,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux)
    # H_I[2](r even)
    compose(hopping_hamiltonian, None, 0, 2, interaction_x,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux,
            _dt=time_step)

    if not second_order:
        return umat

    # H_I[1](r odd)
    qubits = list(range(3, 3 * (num_sites - 1)))
    compose(hopping_hamiltonian, qubits, 1, 1, interaction_x,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux)
    # H_I[2](r odd)
    qubits = list(range(3, 3 * (num_sites - 1)))
    compose(hopping_hamiltonian, qubits, 1, 2, interaction_x,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux)
    # H_I[1](r even)
    compose(hopping_hamiltonian, None, 0, 1, interaction_x,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux)
    # H_E[3] bosonic
    qubits = []
    for site in range(num_sites - 1):
        if shape[::-1][3 * site + 2] != 1:
            qubits += list(range(3 * site, 3 * (site + 1)))
    compose(electric_3b_hamiltonian, qubits,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux)
    # H_E[3] fermionic
    qubits = list(range(0, 3 * num_sites - 3, 3)) + list(range(1, 3 * num_sites - 3, 3))
    compose(electric_3f_hamiltonian, qubits)
    # H_E[1] + H_E[2]
    qubits = list(range(2, 3 * num_sites - 3, 3))
    compose(electric_12_hamiltonian, qubits,
            max_left_flux=max_left_flux, max_right_flux=max_right_flux)
    # H_M
    qubits = list(range(0, 3 * num_sites, 3)) + list(range(1, 3 * num_sites, 3))
    compose(mass_hamiltonian, qubits, mass_mu)

    return umat
