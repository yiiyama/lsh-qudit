"""Generators of parity network circuits for linearly connected qubits."""
from typing import Optional
import numpy as np
from qiskit import QuantumCircuit
from .utils import to_bin, is_zero


class ParityTracer:
    """Tracker of the parity expressed by each qubit of the circuit."""
    def __init__(
        self,
        angles: np.ndarray,
        init: Optional[np.ndarray] = None,
        visited: Optional[set[tuple[int, ...]]] = None,
        name: Optional[str] = None
    ):
        if init is None:
            self.circuit = QuantumCircuit(angles.ndim, name=name)
            self.parities = np.flip(np.eye(angles.ndim, dtype=bool), axis=1)
            self.visited = set()
        else:
            self.circuit = QuantumCircuit(init.shape[0], name=name)
            self.parities = np.array(init)
            self.visited = set(tuple(np.asarray(row, dtype=int)) for row in init)
        self.angles = angles
        if visited:
            self.visited = set(visited)
        self._apply_rz()

    @property
    def num_qubits(self):
        return self.circuit.num_qubits

    def _apply_rz(self, qubit: Optional[int] = None):
        if qubit is None:
            rows = enumerate(self.parities)
        else:
            rows = [(qubit, self.parities[qubit])]

        for iq, row in rows:
            if (trow := tuple(np.asarray(row, dtype=int))) not in self.visited:
                angle = self.angles[trow]
                if not is_zero(angle):
                    self.circuit.rz(2. * angle, iq)
                self.visited.add(trow)

    def cx(self, ctrl: int, targ: int):
        self.circuit.cx(ctrl, targ)
        self.parities[targ] ^= self.parities[ctrl]
        self._apply_rz(qubit=targ)

    def swap(self, q1: int, q2: int):
        self.circuit.swap(q1, q2)
        self.parities[[q1, q2]] = self.parities[[q2, q1]]


def parity_network(
    angles: np.ndarray,
    singles_front: bool = True,
    name: Optional[str] = None
):
    """Generates a parity network circuit for linearly connected qubits with the given angles."""
    num_qubits = angles.ndim
    match num_qubits:
        case 2 | 3:
            return parity_walk_up(angles, singles_front=singles_front, name=name)
        case 4:
            if all(is_zero(angle) for angle in angles[..., 0, :].flat):
                return _parity_walk_4q_up_z1(angles, singles_front=singles_front, name=name)
            if all(is_zero(angle) for angle in angles[:, 0].flat):
                return _parity_walk_4q_down_z2(angles, singles_front=singles_front, name=name)
            if (all(is_zero(angles[to_bin(idx, 4)]) for idx in range(5, 8))
                    or all(is_zero(angles[to_bin(idx, 4)]) for idx in range(9, 15))):
                return parity_walk_downr(angles, singles_front=singles_front, name=name)
            return parity_walk_upr(angles, singles_front=singles_front, name=name)
        case 5:
            if all(is_zero(angle) for angle in angles[..., 0, :].flat):
                return _parity_walk_5q_up_z1(angles, singles_front=singles_front, name=name)
            if all(is_zero(angle) for angle in angles[:, :, 0].flat):
                return _parity_walk_5q_up_z2(angles, singles_front=singles_front, name=name)
            if all(is_zero(angle) for angle in angles[:, 0].flat):
                return _parity_walk_5q_down_z3(angles, singles_front=singles_front, name=name)
            if (all(is_zero(angles[to_bin(idx, 5)]) for idx in range(5, 8))
                    or all(is_zero(angles[to_bin(idx, 5)]) for idx in range(9, 15))
                    or all(is_zero(angles[to_bin(idx, 5)]) for idx in range(17, 31))):
                return parity_walk_downr(angles, singles_front=singles_front, name=name)
            return parity_walk_upr(angles, singles_front=singles_front, name=name)
        case _:
            raise NotImplementedError(f'Parity network for {num_qubits} is not implemented.')


def parity_walk_up(
    angles: np.ndarray,
    singles_front: bool = True,
    name: Optional[str] = None
):
    """Generic non-recursive parity walk.

    CX count:
        nq=2: 2
        nq=3: 8
    """
    if singles_front:
        singles = None
    else:
        singles = set(to_bin(2 ** i, angles.ndim) for i in range(angles.ndim))
    tracer = ParityTracer(angles, visited=singles, name=name)
    _parity_walk_up_sub(tracer)
    if not singles_front:
        tracer.visited -= singles
        tracer._apply_rz()

    return tracer.circuit


def parity_walk_down(
    angles: np.ndarray,
    singles_front: bool = True,
    name: Optional[str] = None
):
    if singles_front:
        singles = None
    else:
        singles = set(to_bin(2 ** i, angles.ndim) for i in range(angles.ndim))
    tracer = ParityTracer(angles, visited=singles, name=name)
    _parity_walk_down_sub(tracer)
    if not singles_front:
        tracer.visited -= singles
        tracer._apply_rz()

    return tracer.circuit


def _parity_walk_up_sub(
    tracer: ParityTracer,
    num_qubits: Optional[int] = None,
    offset: int = 0
):
    if num_qubits is None:
        num_qubits = tracer.num_qubits
    match num_qubits:
        case 2:
            tracer.cx(offset + 0, offset + 1)
            tracer.cx(offset + 0, offset + 1)
        case 3:
            for _ in range(4):
                tracer.cx(offset + 0, offset + 1)
                tracer.cx(offset + 1, offset + 2)
        case _:
            raise NotImplementedError(f'_parity_walk_sub for {num_qubits} qubits not implemented')


def _parity_walk_down_sub(
    tracer: ParityTracer,
    num_qubits: Optional[int] = None,
    offset: int = 0
):
    if num_qubits is None:
        num_qubits = tracer.num_qubits
    match num_qubits:
        case 2:
            tracer.cx(offset + 1, offset + 0)
            tracer.cx(offset + 1, offset + 0)
        case 3:
            for _ in range(4):
                tracer.cx(offset + 2, offset + 1)
                tracer.cx(offset + 1, offset + 0)
        case _:
            raise NotImplementedError(f'_parity_walk_sub for {num_qubits} qubits not implemented')


def _parity_walk_4q_up_z1(
    angles: np.ndarray,
    singles_front: bool = True,
    name: Optional[str] = None
):
    """4Q parity network optimized for angles with **Z* form.

    Note that the parity of qubit 0 is tied to that of qubit 1 in this construction, so the short
    version of this circuit cannot express parities where q0 is 1 and q1 is 0.
    CX count: 18 (fixed Z1), 30 (else)
    """
    block1 = [6, 10, 12, 14]
    block2 = [7, 11, 15]
    block3 = [5, 9, 13]
    if singles_front:
        singles = None
    else:
        singles = set(to_bin(2 ** i, angles.ndim) for i in range(angles.ndim))
    tracer = ParityTracer(angles, visited=singles, name=name)
    if any(not is_zero(angles[to_bin(idx, 4)]) for idx in block1):
        _parity_walk_up_sub(tracer, num_qubits=3, offset=1)
    if any(not is_zero(angles[to_bin(idx, 4)]) for idx in [3] + block2):
        tracer.cx(0, 1)
        if any(not is_zero(angles[to_bin(idx, 4)]) for idx in block2):
            _parity_walk_up_sub(tracer, num_qubits=3, offset=1)
        tracer.cx(0, 1)
    if any(not is_zero(angles[to_bin(idx, 4)]) for idx in block3):
        tracer.swap(0, 1)
        _parity_walk_up_sub(tracer, num_qubits=3, offset=1)
        tracer.swap(0, 1)
    if not singles_front:
        tracer.visited -= singles
        tracer._apply_rz()
    return tracer.circuit


def _parity_walk_4q_down_z2(
    angles: np.ndarray,
    singles_front: bool = True,
    name: Optional[str] = None
):
    """4Q parity network optimized for angles with *Z** form.

    CX count: 18 (fixed Z2), 30 (else)
    """
    block1 = [3, 5, 6, 7]
    block2 = [13, 14, 15]
    block3 = [9, 10, 11]
    if singles_front:
        singles = None
    else:
        singles = set(to_bin(2 ** i, angles.ndim) for i in range(angles.ndim))
    tracer = ParityTracer(angles, visited=singles, name=name)
    if any(not is_zero(angles[to_bin(idx, 4)]) for idx in block1):
        _parity_walk_down_sub(tracer, num_qubits=3)
    if any(not is_zero(angles[to_bin(idx, 4)]) for idx in [12] + block2):
        tracer.cx(3, 2)
        if any(not is_zero(angles[to_bin(idx, 4)]) for idx in block2):
            _parity_walk_down_sub(tracer, num_qubits=3)
        tracer.cx(3, 2)
    if any(not is_zero(angles[to_bin(idx, 4)]) for idx in block3):
        tracer.swap(3, 2)
        _parity_walk_down_sub(tracer, num_qubits=3)
        tracer.swap(3, 2)
    if not singles_front:
        tracer.visited -= singles
        tracer._apply_rz()
    return tracer.circuit


def _parity_walk_5q_up_z1(
    angles: np.ndarray,
    singles_front: bool = True,
    name: Optional[str] = None
):
    """5Q parity network specialized for angles with ***Z* form.

    CX count: 42
    """
    block1 = list(range(6, 32, 4))
    block2 = list(range(7, 32, 4))
    if singles_front:
        singles = None
    else:
        singles = set(to_bin(2 ** i, angles.ndim) for i in range(angles.ndim))
    tracer = ParityTracer(angles, visited=singles, name=name)

    if any(not is_zero(angles[to_bin(idx, 5)]) for idx in block1):
        tracer.cx(2, 3)
        tracer.cx(3, 4)
        _parity_walk_upr_sub(tracer, num_qubits=4, offset=1)
        tracer.cx(3, 4)
        tracer.cx(2, 3)
        tracer.cx(1, 2)
    if any(not is_zero(angles[to_bin(idx, 5)]) for idx in [3] + block2):
        tracer.cx(0, 1)
        if any(not is_zero(angles[to_bin(idx, 5)]) for idx in block2):
            tracer.cx(2, 3)
            tracer.cx(3, 4)
            _parity_walk_upr_sub(tracer, num_qubits=4, offset=1)
            tracer.cx(3, 4)
            tracer.cx(2, 3)
            tracer.cx(1, 2)
        tracer.cx(0, 1)
    if not singles_front:
        tracer.visited -= singles
        tracer._apply_rz()
    return tracer.circuit


def _parity_walk_5q_down_z3(
    angles: np.ndarray,
    singles_front: bool = True,
    name: Optional[str] = None
):
    """5Q parity network specialized for angles with *Z*** form.

    CX count: 42
    """
    block1 = list(range(9, 16))
    block2 = list(range(25, 32))
    if singles_front:
        singles = None
    else:
        singles = set(to_bin(2 ** i, angles.ndim) for i in range(angles.ndim))
    tracer = ParityTracer(angles, visited=singles, name=name)

    if any(not is_zero(angles[to_bin(idx, 5)]) for idx in block1):
        tracer.cx(2, 1)
        tracer.cx(1, 0)
        _parity_walk_downr_sub(tracer, num_qubits=4)
        tracer.cx(1, 0)
        tracer.cx(2, 1)
        tracer.cx(3, 2)
    if any(not is_zero(angles[to_bin(idx, 5)]) for idx in [24] + block2):
        tracer.cx(4, 3)
        if any(not is_zero(angles[to_bin(idx, 5)]) for idx in block2):
            tracer.cx(2, 1)
            tracer.cx(1, 0)
            _parity_walk_downr_sub(tracer, num_qubits=4)
            tracer.cx(1, 0)
            tracer.cx(2, 1)
            tracer.cx(3, 2)
        tracer.cx(4, 3)
    if not singles_front:
        tracer.visited -= singles
        tracer._apply_rz()
    return tracer.circuit


def _parity_walk_5q_up_z2(
    angles: np.ndarray,
    singles_front: bool = True,
    name: Optional[str] = None
):
    """5Q parity network optimized for angles with **Z** form.

    Note that the parities of qubit 0 and 1 are tied to that of qubit 2 in this construction, so the
    circuit cannot express parities where q0 and/or q1 is 1 and q2 is 0.
    CX count: 40
    """
    block1 = [12, 20, 28, 24]
    block2 = [14, 22, 30]
    block3 = [13, 21, 29]
    block4 = [15, 23, 31]
    if singles_front:
        singles = None
    else:
        singles = set(to_bin(2 ** i, angles.ndim) for i in range(angles.ndim))
    tracer = ParityTracer(angles, visited=singles, name=name)
    if any(not is_zero(angles[to_bin(idx, 5)]) for idx in block1):
        _parity_walk_up_sub(tracer, num_qubits=3, offset=2)
    tracer.cx(1, 2)
    if any(not is_zero(angles[to_bin(idx, 5)]) for idx in block2):
        _parity_walk_up_sub(tracer, num_qubits=3, offset=2)
    tracer.cx(0, 1)
    tracer.cx(1, 2)
    if any(not is_zero(angles[to_bin(idx, 5)]) for idx in block3):
        _parity_walk_up_sub(tracer, num_qubits=3, offset=2)
    tracer.cx(0, 1)
    tracer.cx(1, 2)
    if any(not is_zero(angles[to_bin(idx, 5)]) for idx in block4):
        _parity_walk_up_sub(tracer, num_qubits=3, offset=2)
    tracer.cx(0, 1)
    tracer.cx(1, 2)
    tracer.cx(0, 1)
    if not singles_front:
        tracer.visited -= singles
        tracer._apply_rz()
    return tracer.circuit


def parity_walk_upr(
    angles: np.ndarray,
    singles_front: bool = True,
    name: Optional[str] = None
):
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
    if singles_front:
        singles = None
    else:
        singles = set(to_bin(2 ** i, angles.ndim) for i in range(angles.ndim))
    tracer = ParityTracer(angles, visited=singles, name=name)
    for nq in range(2, tracer.num_qubits + 1):
        # Each block covers reverse-order binaries of [2^{nq-1}+1, 2^nq]
        bin_indices = [to_bin(idx, tracer.num_qubits)[::-1]
                       for idx in range(2 ** (nq - 1) + 1, 2 ** nq)]
        if all(is_zero(angles[idx]) for idx in bin_indices):
            # Load up all qubits with full parity
            for iq in range(tracer.num_qubits - 1, tracer.num_qubits - nq + 1, -1):
                tracer.cx(iq - 1, iq)
            for iq in range(tracer.num_qubits - nq, tracer.num_qubits - 1):
                tracer.cx(iq, iq + 1)
        else:
            _parity_walk_upr_sub(tracer, num_qubits=nq, offset=tracer.num_qubits - nq)

    if not singles_front:
        tracer.visited -= singles
        tracer._apply_rz()

    for iq in range(tracer.num_qubits - 1, 0, -1):
        tracer.cx(iq - 1, iq)

    return tracer.circuit


def _parity_walk_upr_sub(tracer: ParityTracer, num_qubits: Optional[int] = None, offset: int = 0):
    if num_qubits is None:
        num_qubits = tracer.num_qubits
    if num_qubits == 1:
        return

    _parity_walk_upr_sub(tracer, num_qubits=num_qubits - 1, offset=offset)

    match num_qubits:
        case 2:
            tracer.cx(offset + 0, offset + 1)
        case 3:
            for _ in range(2):
                tracer.cx(offset + 1, offset + 2)
                tracer.cx(offset + 0, offset + 1)
        case 4:
            for _ in range(2):
                tracer.cx(offset + 2, offset + 3)
                tracer.cx(offset + 1, offset + 2)
                tracer.cx(offset + 0, offset + 1)
                tracer.cx(offset + 2, offset + 3)
                tracer.cx(offset + 1, offset + 2)
        case _:
            raise NotImplementedError(f'Recursive parity walk for {num_qubits} qubits not'
                                      ' implemented')


def parity_walk_downr(
    angles: np.ndarray,
    singles_front: bool = True,
    name: Optional[str] = None
):
    """Recursively constructed parity network. Empirically known to use fewest CXs.

    The parity network is built sequentially in the increasing number of involved qubits. This
    allows ommission of circuit blocks if angles for certain contiguous parities are all zero. Each
    block of a given qubit number is constructed recursively from smaller qubit-number blocks.
    """
    if singles_front:
        singles = None
    else:
        singles = set(to_bin(2 ** i, angles.ndim) for i in range(angles.ndim))
    tracer = ParityTracer(angles, visited=singles, name=name)
    for nq in range(2, tracer.num_qubits + 1):
        # Each block covers reverse-order binaries of [2^{nq-1}+1, 2^nq]
        bin_indices = [to_bin(idx, tracer.num_qubits)
                       for idx in range(2 ** (nq - 1) + 1, 2 ** nq)]
        if all(is_zero(angles[idx]) for idx in bin_indices):
            # Load up all qubits with full parity
            for iq in range(1, nq - 1):
                tracer.cx(iq, iq - 1)
            for iq in range(nq - 1, 0, -1):
                tracer.cx(iq, iq - 1)
        else:
            _parity_walk_downr_sub(tracer, num_qubits=nq)

    if not singles_front:
        tracer.visited -= singles
        tracer._apply_rz()

    for iq in range(tracer.num_qubits - 1):
        tracer.cx(iq + 1, iq)

    return tracer.circuit


def _parity_walk_downr_sub(tracer: ParityTracer, num_qubits: Optional[int] = None, offset: int = 0):
    if num_qubits is None:
        num_qubits = tracer.num_qubits
    if num_qubits == 1:
        return

    _parity_walk_downr_sub(tracer, num_qubits=num_qubits - 1, offset=offset + 1)

    match num_qubits:
        case 2:
            tracer.cx(offset + 1, offset + 0)
        case 3:
            for _ in range(2):
                tracer.cx(offset + 1, offset + 0)
                tracer.cx(offset + 2, offset + 1)
        case 4:
            for _ in range(2):
                tracer.cx(offset + 1, offset + 0)
                tracer.cx(offset + 2, offset + 1)
                tracer.cx(offset + 3, offset + 2)
                tracer.cx(offset + 1, offset + 0)
                tracer.cx(offset + 2, offset + 1)
        case _:
            raise NotImplementedError(f'Recursive parity walk for {num_qubits} qubits not'
                                      ' implemented')

    return tracer.circuit


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
