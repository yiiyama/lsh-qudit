"""Generators of parity network circuits for linearly connected qubits."""
import numpy as np
from qiskit import QuantumCircuit
from .utils import to_bin


class RZSetter:
    """Helper class to apply Rz gates only for nonzero angles."""
    def __init__(
        self,
        circuit: QuantumCircuit,
        angles: np.ndarray,
        init: bool = True
    ):
        self.circuit = circuit
        self.angles = angles
        if init:
            for idim in range(self.angles.ndim):
                idx = [0] * self.angles.ndim
                idx[self.angles.ndim - idim - 1] = 1
                self.apply(idim, *idx)

    def apply(self, qubit: int, *idx):
        if not is_zero(self.angles[idx]):
            self.circuit.rz(2. * self.angles[idx], qubit)


def is_zero(angle):
    try:
        return np.isclose(angle, 0.)
    except TypeError:
        return angle.sympify() == 0


def parity_network(angles: np.ndarray):
    """Generates a parity network circuit for linearly connected qubits with the given angles."""
    num_qubits = angles.ndim
    match num_qubits:
        case 2:
            return _parity_walk_2q_up(angles)
        case 3:
            return _parity_walk_3q_up(angles)
        case 4:
            if all(is_zero(angle) for angle in angles[..., 0, :].flat):
                return _parity_walk_4q_up_z1(angles)
            if all(is_zero(angle) for angle in angles[:, 0].flat):
                return _parity_walk_4q_down_z2(angles)
            if (all(is_zero(angles[to_bin(idx, 4)]) for idx in range(5, 8))
                    or all(is_zero(angles[to_bin(idx, 4)]) for idx in range(9, 15))):
                return _parity_walk_downr(angles)
            return _parity_walk_upr(angles)
        case 5:
            if all(is_zero(angle) for angle in angles[:, :, 0].flat):
                return _parity_walk_5q_up_z2(angles)
            if (all(is_zero(angles[to_bin(idx, 5)]) for idx in range(5, 8))
                    or all(is_zero(angles[to_bin(idx, 5)]) for idx in range(9, 15))
                    or all(is_zero(angles[to_bin(idx, 5)]) for idx in range(17, 31))):
                return _parity_walk_downr(angles)
            return _parity_walk_upr(angles)
        case _:
            raise NotImplementedError(f'Parity network for {num_qubits} is not implemented.')


def _parity_walk_2q_up(angles: np.ndarray, init: bool = True, close: bool = True):
    """2Q parity network with CX acting from qubit 0 to 1.

    CX count: 2
    """
    circuit = QuantumCircuit(2)
    rz = RZSetter(circuit, angles, init=init)
    circuit.cx(0, 1)
    rz.apply(1, 1, 1)
    if close:
        _close_network(circuit, 'up')
    return circuit


def _parity_walk_2q_down(angles: np.ndarray, init: bool = True, close: bool = True):
    """2Q parity network with CX acting from qubit 0 to 1.

    CX count: 2
    """
    circuit = QuantumCircuit(2)
    rz = RZSetter(circuit, angles, init=init)
    circuit.cx(1, 0)
    rz.apply(0, 1, 1)
    if close:
        _close_network(circuit, 'down')
    return circuit


def _parity_walk_3q_up(angles: np.ndarray, init: bool = True, close: bool = True):
    """3Q parity network with the first CX acting from qubit 0 to 1.

    CX count: 8
    """
    circuit = QuantumCircuit(3)
    rz = RZSetter(circuit, angles, init=init)
    circuit.cx(0, 1)
    rz.apply(1, 0, 1, 1)
    circuit.cx(1, 2)
    rz.apply(2, 1, 1, 1)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    rz.apply(2, 1, 0, 1)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    rz.apply(2, 1, 1, 0)
    if close:
        _close_network(circuit, 'up')
    return circuit


def _parity_walk_3q_down(angles: np.ndarray, init: bool = True, close: bool = True):
    """3Q parity network with the first CX acting from qubit 2 to 1.

    CX count: 8
    """
    circuit = QuantumCircuit(3)
    rz = RZSetter(circuit, angles, init=init)
    circuit.cx(2, 1)
    rz.apply(1, 1, 1, 0)
    circuit.cx(1, 0)
    rz.apply(2, 1, 1, 1)
    circuit.cx(2, 1)
    rz.apply(1, 0, 1, 0)
    circuit.cx(1, 0)
    rz.apply(2, 1, 0, 1)
    circuit.cx(2, 1)
    circuit.cx(1, 0)
    rz.apply(2, 0, 1, 1)
    if close:
        _close_network(circuit, 'down')
    return circuit


def _parity_walk_4q_up_z1(angles: np.ndarray):
    """4Q parity network optimized for angles with **Z* form.

    Note that the parity of qubit 0 is tied to that of qubit 1 in this construction, so the short
    version of this circuit cannot express parities where q0 is 1 and q1 is 0.
    CX count: 18 (fixed Z1), 30 (else)
    """
    circuit = QuantumCircuit(4)
    rz = RZSetter(circuit, angles)
    circuit.compose(_parity_walk_3q_up(angles[..., 0], init=False), qubits=[1, 2, 3], inplace=True)
    circuit.cx(0, 1)
    rz.apply(1, 0, 0, 1, 1)
    circuit.compose(_parity_walk_3q_up(angles[..., 1], init=False), qubits=[1, 2, 3], inplace=True)
    circuit.cx(0, 1)
    if any(not is_zero(angle) for angle in angles[..., 0, 1].flat):
        circuit.swap(0, 1)
        subangles = np.zeros((2,) * 3)
        subangles[..., 1] = angles[..., 0, 1]
        circuit.compose(_parity_walk_3q_up(subangles, init=False), qubits=[1, 2, 3],
                        inplace=True)
        circuit.swap(0, 1)
    return circuit


def _parity_walk_4q_down_z2(angles: np.ndarray):
    """4Q parity network optimized for angles with *Z** form.

    CX count: 18 (fixed Z2), 30 (else)
    """
    circuit = QuantumCircuit(4)
    rz = RZSetter(circuit, angles)
    circuit.compose(_parity_walk_3q_down(angles[0], init=False), qubits=[0, 1, 2], inplace=True)
    circuit.cx(3, 2)
    rz.apply(2, 1, 1, 0, 0)
    circuit.compose(_parity_walk_3q_down(angles[1], init=False), qubits=[0, 1, 2], inplace=True)
    circuit.cx(3, 2)
    if any(not is_zero(angle) for angle in angles[1, 0].flat):
        circuit.swap(3, 2)
        subangles = np.zeros((2,) * 3)
        subangles[1] = angles[1, 0]
        circuit.compose(_parity_walk_3q_down(subangles, init=False), qubits=[0, 1, 2],
                        inplace=True)
        circuit.swap(3, 2)
    return circuit


def _parity_walk_5q_up_z2(angles: np.ndarray):
    """5Q parity network optimized for angles with **Z** form.

    Note that the parities of qubit 0 and 1 are tied to that of qubit 2 in this construction, so the
    circuit cannot express parities where q0 and/or q1 is 1 and q2 is 0.
    CX count: 40
    """
    circuit = QuantumCircuit(5)
    rz = RZSetter(circuit, angles)
    circuit.compose(_parity_walk_3q_up(angles[..., 0, 0], init=False), qubits=[2, 3, 4],
                    inplace=True)
    circuit.cx(1, 2)
    rz.apply(2, 0, 0, 1, 1, 0)
    circuit.compose(_parity_walk_3q_up(angles[..., 1, 0], init=False), qubits=[2, 3, 4],
                    inplace=True)
    circuit.cx(0, 1)
    rz.apply(1, 0, 0, 0, 1, 1)
    circuit.cx(1, 2)
    rz.apply(2, 0, 0, 1, 0, 1)
    circuit.compose(_parity_walk_3q_up(angles[..., 0, 1], init=False), qubits=[2, 3, 4],
                    inplace=True)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    rz.apply(2, 0, 0, 1, 1, 1)
    circuit.compose(_parity_walk_3q_up(angles[..., 1, 1], init=False), qubits=[2, 3, 4],
                    inplace=True)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    return circuit


def _parity_walk_upr(angles: np.ndarray):
    """Recursively constructed parity network. Empirically known to use fewest CXs.

    The parity network is built sequentially in the increasing number of involved qubits. This
    allows ommission of circuit blocks if angles for certain contiguous parities are all zero. Each
    block of a given qubit number is constructed recursively from smaller qubit-number blocks.

    CX count:
        nq=2: 2
        nq=3: 8
        nq=4: 24 (full)
              20 (no I3)
        nq=5: ?
    """
    num_qubits = angles.ndim
    circuit = QuantumCircuit(num_qubits)
    RZSetter(circuit, angles)
    for nq in range(2, num_qubits + 1):
        # Each block covers reverse-order binaries of [2^{nq-1}+1, 2^nq]
        bin_indices = [to_bin(idx, num_qubits)[::-1] for idx in range(2 ** (nq - 1) + 1, 2 ** nq)]
        if all(is_zero(angles[idx]) for idx in bin_indices):
            # Load up all qubits with full parity
            for iq in range(num_qubits - 1, num_qubits - nq + 1, -1):
                circuit.cx(iq - 1, iq)
            for iq in range(num_qubits - nq, num_qubits - 1):
                circuit.cx(iq, iq + 1)
        else:
            subangles = angles[(slice(None),) * nq + (0,) * (num_qubits - nq)]
            circuit.compose(_parity_walk_upr_sub(subangles),
                            qubits=list(range(num_qubits - nq, num_qubits)), inplace=True)

    _close_network(circuit, 'upr')
    return circuit


def _parity_walk_upr_sub(angles: np.ndarray):
    num_qubits = angles.ndim
    circuit = QuantumCircuit(num_qubits)
    if num_qubits == 1:
        return circuit

    subangles = angles[0, ...]
    circuit.compose(_parity_walk_upr_sub(subangles), qubits=list(range(num_qubits - 1)),
                    inplace=True)
    rz = RZSetter(circuit, angles, init=False)
    match num_qubits:
        case 2:
            circuit.cx(0, 1)
            rz.apply(1, 1, 1)
        case 3:
            for irep in range(2):
                circuit.cx(1, 2)
                circuit.cx(0, 1)
                rz.apply(2, *to_bin(5 + 2 * irep, 3))
        case 4:
            for irep in range(2):
                circuit.cx(2, 3)
                circuit.cx(1, 2)
                circuit.cx(0, 1)
                rz.apply(3, *to_bin(9 + 2 * irep, 4))
                circuit.cx(2, 3)
                circuit.cx(1, 2)
                rz.apply(3, *to_bin(13 + 2 * irep, 4))

    return circuit


def _parity_walk_downr(angles: np.ndarray):
    """Recursively constructed parity network. Empirically known to use fewest CXs.

    The parity network is built sequentially in the increasing number of involved qubits. This
    allows ommission of circuit blocks if angles for certain contiguous parities are all zero. Each
    block of a given qubit number is constructed recursively from smaller qubit-number blocks.
    """
    num_qubits = angles.ndim
    circuit = QuantumCircuit(num_qubits)
    RZSetter(circuit, angles)
    for nq in range(2, num_qubits + 1):
        indices = range(2 ** (nq - 1) + 1, 2 ** nq)
        if all(is_zero(angles[to_bin(idx, num_qubits)]) for idx in indices):
            # Load up all qubits with full parity
            for iq in range(1, nq - 1):
                circuit.cx(iq, iq - 1)
            for iq in range(nq - 1, 0, -1):
                circuit.cx(iq, iq - 1)
        else:
            subangles = angles[(0,) * (num_qubits - nq)]
            circuit.compose(_parity_walk_downr_sub(subangles), qubits=list(range(nq)), inplace=True)

    _close_network(circuit, 'downr')
    return circuit


def _parity_walk_downr_sub(angles: np.ndarray):
    num_qubits = angles.ndim
    circuit = QuantumCircuit(num_qubits)
    if num_qubits == 1:
        return circuit

    subangles = angles[..., 0]
    circuit.compose(_parity_walk_downr_sub(subangles), qubits=list(range(1, num_qubits)),
                    inplace=True)
    rz = RZSetter(circuit, angles, init=False)
    match num_qubits:
        case 2:
            circuit.cx(1, 0)
            rz.apply(0, 1, 1)
        case 3:
            for irep in range(2):
                circuit.cx(1, 0)
                circuit.cx(2, 1)
                rz.apply(0, *to_bin(5 + 2 * irep, 3))
        case 4:
            for irep in range(2):
                circuit.cx(1, 0)
                circuit.cx(2, 1)
                circuit.cx(3, 2)
                rz.apply(0, *to_bin(9 + 4 * irep, 4))
                circuit.cx(1, 0)
                circuit.cx(2, 1)
                rz.apply(0, *to_bin(11 + 4 * irep, 4))

    return circuit


def _close_network(circuit, pattern):
    match pattern:
        case 'up':
            for iq in range(circuit.num_qubits - 1):
                circuit.cx(iq, iq + 1)
        case 'upr':
            for iq in range(circuit.num_qubits - 1, 0, -1):
                circuit.cx(iq - 1, iq)
        case 'down':
            for iq in range(circuit.num_qubits - 1, 0, -1):
                circuit.cx(iq, iq - 1)
        case 'downr':
            for iq in range(circuit.num_qubits - 1):
                circuit.cx(iq + 1, iq)


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
