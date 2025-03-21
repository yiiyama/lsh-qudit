# pylint: disable=unused-argument, missing-class-docstring
"""Circuit validation against hermitian / unitary matrices."""
from collections.abc import Sequence
from typing import Optional
import numpy as np
import jax
import jax.numpy as jnp
from qiskit import QuantumCircuit
from .utils import clean_array


def circuit_unitary(
    circuit: QuantumCircuit,
    qutrits=(),
    boson_regs=None,
    ancillae=(),
    diagonal=False,
    clean=True
):
    boson_regs = boson_regs or []
    num_qubits = circuit.num_qubits
    # Qubit to axis map
    qubit_map = {qubit: i for i, qubit in enumerate(circuit.qubits[::-1])}
    full_shape = tuple(3 if iq in qutrits else 2 for iq in range(num_qubits - 1, -1, -1))

    unitary = jnp.eye(np.prod(full_shape), dtype=np.complex128).reshape(full_shape + full_shape)
    for datum in circuit.data:
        gate = datum.operation
        if gate.name == 'barrier':
            continue
        axes = tuple(qubit_map[q] for q in datum.qubits[::-1])
        nq = len(axes)
        if nq == 1 and num_qubits - axes[0] - 1 in qutrits:
            op_unitary = np.array(gate)
            if op_unitary.shape[0] == 2:
                tmp = np.eye(3, dtype=np.complex128)
                tmp[:2, :2] = op_unitary
                op_unitary = tmp
            op_unitary = jnp.array(op_unitary)
        else:
            op_unitary = jnp.array(gate)

        shape = tuple(full_shape[iq] for iq in axes)
        op_unitary = op_unitary.reshape(shape + shape)
        op_axes = list(range(nq, 2 * nq))
        unitary = jnp.tensordot(op_unitary, unitary, (op_axes, axes))
        unitary = jnp.moveaxis(unitary, list(range(nq)), axes)

    if ancillae:
        source = tuple(num_qubits - i - 1 for i in ancillae)
        source += tuple(2 * num_qubits - i - 1 for i in ancillae)
        num_anc_axes = len(ancillae) * 2
        dest = tuple(range(num_anc_axes))
        unitary = jnp.moveaxis(unitary, source, dest)
        unitary = unitary[(0,) * num_anc_axes]
        full_shape = tuple(full_shape[iq] for iq in range(num_qubits - 1, -1, -1)
                           if iq not in ancillae)
        num_qubits -= len(ancillae)
        # update the qubit map
        qubit_map = {}
        for iq in list(range(circuit.num_qubits))[::-1]:
            if iq not in ancillae:
                qubit_map[circuit.qubits[iq]] = len(qubit_map)

    for reg in boson_regs:
        axes = np.array([qubit_map[circuit.qubits[iq]] for iq in reg[::-1]])
        nq = len(axes)
        new_ax = min(axes)
        unitary = np.moveaxis(unitary, axes, np.arange(new_ax, new_ax + nq))
        new_ax += num_qubits
        unitary = np.moveaxis(unitary, axes + num_qubits, np.arange(new_ax, new_ax + nq))

    full_dim = np.prod(full_shape)
    unitary = unitary.reshape(full_dim, full_dim)
    if diagonal:
        unitary = jnp.diagonal(unitary)

    if clean:
        return clean_array(unitary)
    return unitary


def validate_circuit(
    circuit: QuantumCircuit,
    target: np.ndarray,
    subspace: Optional[tuple[Sequence[int]]] = None,
    shape: Optional[tuple[int, ...]] = None,
    qutrits=(),
    boson_regs=None,
    ancillae=(),
    exponentiate=True,
    diagonal=False,
    result_only=True
):
    unitary = circuit_unitary(circuit, qutrits=qutrits, boson_regs=boson_regs,
                              ancillae=ancillae, clean=False)
    return validate_unitary(unitary, target,
                            subspace=subspace, shape=shape, exponentiate=exponentiate,
                            diagonal=diagonal, result_only=result_only)


def validate_unitary(
    unitary: np.ndarray,
    target: np.ndarray,
    subspace: Sequence[int] | tuple[Sequence[int], ...] = None,
    shape: Optional[tuple[int, ...]] = None,
    exponentiate=True,
    diagonal=False,
    result_only=True
):
    target = jnp.asarray(target)
    if exponentiate:
        target = jax.scipy.linalg.expm(-1.j * target)
    else:
        target = target.copy()

    if isinstance(subspace, tuple):
        if shape is None:
            shape = (2,) * len(subspace)
        full_dim = np.prod(shape)
        unitary = unitary.reshape((full_dim,) + shape)[(slice(None),) + subspace]
        unitary = unitary.reshape(full_dim, len(subspace[0]))
        target = target.reshape((full_dim,) + shape)[(slice(None),) + subspace]
        target = target.reshape(full_dim, len(subspace[0]))
    elif subspace is not None:
        unitary = unitary[:, subspace]
        target = target[:, subspace]

    test = jnp.einsum('ij,ik->jk', unitary.conjugate(), target)
    is_identity = bool(jnp.allclose(test * test[0, 0].conjugate(), jnp.eye(test.shape[0])))
    if result_only:
        return is_identity

    if diagonal and subspace is None:
        unitary = np.diagonal(unitary)
        target = np.diagonal(target)

    unitary = clean_array(unitary)
    target = clean_array(target)

    return is_identity, unitary, target
