import numpy as np
import scipy
from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, PhaseGate, XGate, CPhaseGate, CXGate, CCXGate, HGate
from qiskit.quantum_info import SparsePauliOp
import cirq
from qutrit_experiments.gates import (X12Gate, P1Gate, P2Gate, RZ12Gate, QGate, XplusGate,
                                      XminusGate, QubitQutritCRxPlusPiGate,
                                      QubitQutritCRxMinusPiGate)


def circuit_unitary(
    circuit: QuantumCircuit,
    qutrits=(),
    diagonal=False
):
    # Translate
    qubit_map = {}
    for qreg in circuit.qregs:
        for qubit in qreg:
            qubit_map[qubit] = len(qubit_map)

    out_qubits = cirq.LineQubit.range(circuit.num_qubits)
    for iq in qutrits:
        out_qubits[iq] = cirq.LineQid(iq, dimension=3)

    out_circuit = cirq.Circuit()
    for datum in circuit.data:
        gate = datum.operation
        qubits = tuple(out_qubits[qubit_map[q]] for q in datum.qubits)
        cirq_gate = GATE_TRANSLATION[gate.name]
        if gate.params:
            out_circuit.append(cirq_gate(*gate.params).on(*qubits))
        else:
            try:
                out_circuit.append(cirq_gate.on(*qubits))
            except ValueError:
                print(datum)
                print(cirq_gate)
                print(qubits)
                raise
    unitary = out_circuit.unitary(qubit_order=out_qubits[::-1])
    if diagonal:
        unitary = np.diagonal(unitary)
    return unitary


def validate_circuit(
    circuit: QuantumCircuit,
    hamiltonian_t: np.ndarray,
    qutrits=(),
    diagonal=False,
    result_only=True
):
    unitary = circuit_unitary(circuit, qutrits=qutrits)
    target = scipy.linalg.expm(-1.j * hamiltonian_t)
    test = np.einsum('ij,ik->jk', unitary.conjugate(), target)
    is_identity = np.allclose(test * test[0, 0].conjugate(), np.eye(test.shape[0]))
    if result_only:
        return is_identity

    if diagonal:
        unitary = np.diagonal(unitary)
        target = np.diagonal(target)

    return is_identity, unitary, target


class PhaseGateC(cirq.Gate):
    def __init__(self, phi):
        super(PhaseGateC, self)
        self.phi = phi

    def _num_qubits_(self):
        return 1

    def _unitary_(self):
        return np.array([
            [1., 0.],
            [0., np.exp(1.j * self.phi)]
        ])

    def _circuit_diagram_info_(self, args):
        return f'P({self.phi})'


class CPhaseGateC(cirq.Gate):
    def __init__(self, phi):
        super(CPhaseGateC, self)
        self.phi = phi

    def _num_qubits_(self):
        return 2

    def _unitary_(self):
        return np.array([
            [1., 0., 0., 0.],
            [0., 1., 0., 0.],
            [0., 0., 1., 0.],
            [0., 0., 0., np.exp(1.j * self.phi)]
        ])

    def _circuit_diagram_info_(self, args):
        return '@', f'P({self.phi})'


class RZ01GateC(cirq.Gate):
    def __init__(self, phi):
        super(RZ01GateC, self)
        self.phi = phi

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [np.exp(-0.5j * self.phi), 0., 0.],
            [0., np.exp(0.5j * self.phi), 0.],
            [0., 0., 1.]
        ])

    def _circuit_diagram_info_(self, args):
        return f'Rz01({self.phi})'


class RZ12GateC(cirq.Gate):
    def __init__(self, phi):
        super(RZ12GateC, self)
        self.phi = phi

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [1., 0., 0.],
            [0., np.exp(-0.5j * self.phi), 0.],
            [0., 0., np.exp(0.5j * self.phi)]
        ])

    def _circuit_diagram_info_(self, args):
        return f'Rz12({self.phi})'


class P1GateC(cirq.Gate):
    def __init__(self, phi):
        super(P1GateC, self)
        self.phi = phi

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [1., 0., 0.],
            [0., np.exp(1.j * self.phi), 0.],
            [0., 0., 1.]
        ])

    def _circuit_diagram_info_(self, args):
        return f'P1({self.phi})'


class P2GateC(cirq.Gate):
    def __init__(self, phi):
        super(P2GateC, self)
        self.phi = phi

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [0., 0., np.exp(1.j * self.phi)]
        ])

    def _circuit_diagram_info_(self, args):
        return f'P2({self.phi})'


class QGateC(cirq.Gate):
    def __init__(self, phi):
        super(QGateC, self)
        self.phi = phi

    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [1., 0., 0.],
            [0., np.exp(1.j * self.phi), 0.],
            [0., 0., np.exp(2.j * self.phi)]
        ])

    def _circuit_diagram_info_(self, args):
        return f'Q({self.phi})'


class X12GateC(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.]
        ])

    def _circuit_diagram_info_(self, args):
        return 'X12'


class XplusGateC(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.]
        ])

    def _circuit_diagram_info_(self, args):
        return 'X+'


class XminusGateC(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.]
        ])

    def _circuit_diagram_info_(self, args):
        return 'X-'


class CXiGateC(cirq.Gate):
    def _qid_shape_(self):
        return (2, 3)

    def _unitary_(self):
        return np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., -1.j, 0.],
            [0., 0., 0., -1.j, 0., 0.],
            [0., 0., 0., 0., 0., 1.]
        ])

    def _circuit_diagram_info_(self, args):
        return '@', 'Ξ'


class CXiDagGateC(cirq.Gate):
    def _qid_shape_(self):
        return (2, 3)

    def _unitary_(self):
        return np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 1.j, 0.],
            [0., 0., 0., 1.j, 0., 0.],
            [0., 0., 0., 0., 0., 1.]
        ])

    def _circuit_diagram_info_(self, args):
        return '@', 'Ξ†'


GATE_TRANSLATION = {
    'rz': cirq.rz,
    'p': PhaseGateC,
    'x': cirq.X,
    'cp': CPhaseGateC,
    'cx': cirq.CNOT,
    'ccx': cirq.CCNOT,
    'h': cirq.H,
    'x12': X12GateC(),
    'q': QGateC,
    'p1': P1GateC,
    'p2': P2GateC,
    'qubit_qutrit_crx_pluspi': CXiGateC(),
    'qubit_qutrit_crx_minuspi': CXiDagGateC()
}
