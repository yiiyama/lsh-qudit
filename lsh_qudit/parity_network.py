import numpy as np
from qiskit import QuantumCircuit


def _full_parity_walk_sub(num_qubits, angles):
    circ = QuantumCircuit(num_qubits)
    if num_qubits == 1:
        return circ

    circ.compose(_full_parity_walk_sub(num_qubits - 1, angles[::2]),
                 qubits=list(range(1, num_qubits)), inplace=True)

    if num_qubits == 2:
        circ.cx(1, 0)
        circ.rz(2. * angles[3], 0)
    elif num_qubits == 3:
        for i1 in range(2):
            circ.cx(1, 0)
            circ.cx(2, 1)
            circ.rz(2. * angles[5 + 2 * i1], 0)
    elif num_qubits == 4:
        for i1 in range(2):
            circ.cx(1, 0)
            circ.cx(2, 1)
            circ.cx(3, 2)
            circ.rz(2. * angles[9 + 4 * i1], 0)
            circ.cx(1, 0)
            circ.cx(2, 1)
            circ.rz(2. * angles[11 + 4 * i1], 0)

    return circ


def full_parity_walk(num_qubits, angles):
    circ = QuantumCircuit(num_qubits)

    for iq in range(num_qubits):
        circ.rz(2. * angles[2 ** iq], iq)

    for nq in range(2, num_qubits + 1):
        circ.compose(_full_parity_walk_sub(nq, angles), qubits=list(range(nq)), inplace=True)

    for iq in range(num_qubits - 1):
        circ.cx(iq + 1, iq)
    return circ


def trace_parity(circuit):
    nq = circuit.num_qubits
    parity = np.eye(nq, dtype=bool)
    visited = set(2 ** np.arange(nq))
    for datum in circuit.data:
        if datum.operation.name == 'cx':
            ctrl = circuit.find_bit(datum.qubits[0]).index
            targ = circuit.find_bit(datum.qubits[1]).index
            parity[targ] ^= parity[ctrl]
            visited.add(np.sum(parity[targ] * (2 ** np.arange(nq))))
        elif datum.operation.name == 'swap':
            q1 = circuit.find_bit(datum.qubits[0]).index
            q2 = circuit.find_bit(datum.qubits[1]).index
            tmp = np.array(parity[q2])
            parity[q2] = parity[q1]
            parity[q1] = tmp
    return parity, visited
