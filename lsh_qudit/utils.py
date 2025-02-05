"""Tools."""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


def count_gates(qc: QuantumCircuit):
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def remove_idle_wires(qc: QuantumCircuit, inplace=False, flatten=False):
    if not inplace:
        qc = qc.copy()

    gate_count = count_gates(qc)
    for qubit, count in gate_count.items():
        if count == 0:
            qc.qubits.remove(qubit)

    if flatten:
        qubit_map = {qubit: i for i, qubit in enumerate(qc.qubits)}
        new_circ = QuantumCircuit(len(qubit_map))
        for datum in qc.data:
            qubits = [qubit_map[q] for q in datum.qubits]
            new_circ.append(datum.operation, qubits)
        qc = new_circ

    return qc


def op_matrix(op, shape, qubits):
    shape = tuple(shape)
    if isinstance(qubits, int):
        qubits = (qubits,)
    full_nq = len(shape)
    full_dim = np.prod(shape)
    op_nq = len(qubits)
    op_shape = tuple(shape[::-1][i] for i in qubits)
    op_dim = np.prod(op_shape)
    idle_shape = tuple(dim for i, dim in enumerate(shape) if full_nq - i - 1 not in qubits)
    idle_dim = np.prod(idle_shape)
    mat = np.zeros((idle_dim, idle_dim, op_dim, op_dim), dtype=np.complex128)
    didx = np.arange(idle_dim)
    if isinstance(op, SparsePauliOp):
        op = op.to_matrix()
    mat[didx, didx] = op
    mat = mat.reshape(idle_shape * 2 + op_shape * 2)
    source = tuple(range(-2 * op_nq, 0))
    dest = tuple(full_nq - q - 1 for q in qubits) + tuple(2 * full_nq - q - 1 for q in qubits)
    mat = np.moveaxis(mat, source, dest)
    mat = mat.reshape(full_dim, full_dim)
    return mat


def physical_states(left_flux=None, right_flux=None):
    states = np.array(np.unravel_index(np.arange(12 * 12), (2, 2, 3, 2, 2, 3))).T
    agl_mask = np.equal((1 - states[:, 0]) * states[:, 1] + states[:, 2],
                        states[:, 3] * (1 - states[:, 4]) + states[:, 5])
    states = states[agl_mask]
    if isinstance(left_flux, int):
        left_flux = (left_flux,)
    if left_flux:
        mask = np.zeros(states.shape[0], dtype=bool)
        for val in left_flux:
            mask |= np.equal(states[:, 0] * (1 - states[:, 1]) + states[:, 2], val)
        states = states[mask]
    if isinstance(right_flux, int):
        right_flux = (right_flux,)
    if right_flux:
        mask = np.zeros(states.shape[0], dtype=bool)
        for val in right_flux:
            mask |= np.equal((1 - states[:, 3]) * states[:, 4] + states[:, 5], val)
        states = states[mask]

    return states
