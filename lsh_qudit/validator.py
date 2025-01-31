# pylint: disable=unused-argument, missing-class-docstring
"""Circuit validation against hermitian / unitary matrices."""
import numpy as np
import scipy
from qiskit import QuantumCircuit
import cirq  # pylint: disable=import-error


def circuit_unitary(
    circuit: QuantumCircuit,
    qutrits=(),
    ancillae=(),
    diagonal=False
):
    # Translate
    qubit_map = {qubit: i for i, qubit in enumerate(circuit.qubits)}
    num_qubits = len(qubit_map)  # circuit.num_qubits is not reduced after remove_idle_wires

    out_qubits = cirq.LineQubit.range(num_qubits)
    for iq in qutrits:
        out_qubits[iq] = cirq.LineQid(iq, dimension=3)

    out_circuit = cirq.Circuit()
    for datum in circuit.data:
        gate = datum.operation
        qubits = tuple(out_qubits[qubit_map[q]] for q in datum.qubits)
        cirq_gate = GATE_TRANSLATION[gate.name]
        if isinstance(cirq_gate, tuple):
            if qubits[0].x in qutrits:
                cirq_gate = cirq_gate[1]
            else:
                cirq_gate = cirq_gate[0]
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
    if ancillae:
        shape = tuple(qubit.dimension for qubit in out_qubits[::-1])
        unitary = unitary.reshape(shape + shape)
        source = tuple(num_qubits - i - 1 for i in ancillae)
        source += tuple(2 * num_qubits - i - 1 for i in ancillae)
        num_anc_axes = len(ancillae) * 2
        dest = tuple(range(num_anc_axes))
        unitary = np.moveaxis(unitary, source, dest)
        unitary = unitary[(0,) * num_anc_axes]
        new_dim = np.prod(shape) // np.prod([out_qubits[i].dimension for i in ancillae])
        unitary = unitary.reshape(new_dim, new_dim)
    if diagonal:
        unitary = np.diagonal(unitary)
    unitary.real = np.where(np.isclose(unitary.real, 0.), 0., unitary.real)
    unitary.imag = np.where(np.isclose(unitary.imag, 0.), 0., unitary.imag)
    return unitary


def validate_circuit(
    circuit: QuantumCircuit,
    target: np.ndarray,
    qutrits=(),
    ancillae=(),
    exponentiate=True,
    diagonal=False,
    result_only=True
):
    unitary = circuit_unitary(circuit, qutrits=qutrits, ancillae=ancillae)
    if exponentiate:
        target = scipy.linalg.expm(-1.j * target)
    else:
        target = target.copy()
    target.real = np.where(np.isclose(target.real, 0.), 0., target.real)
    target.imag = np.where(np.isclose(target.imag, 0.), 0., target.imag)

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


class X01GateC(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        return np.array([
            [0., 1., 0.],
            [1., 0., 0.],
            [0., 0., 1.]
        ])

    def _circuit_diagram_info_(self, args):
        return 'X01'


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


class H01GateC(cirq.Gate):
    def _qid_shape_(self):
        return (3,)

    def _unitary_(self):
        v = 1. / np.sqrt(2.)
        return np.array([
            [v, v, 0.],
            [v, -v, 0.],
            [0., 0., 1.]
        ])

    def _circuit_diagram_info_(self, args):
        return 'X01'


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
    'rz': (cirq.rz, RZ01GateC),
    'p': (PhaseGateC, P1GateC),
    'x': (cirq.X, X01GateC),
    'cp': CPhaseGateC,
    'cx': cirq.CNOT,
    'ccx': cirq.CCNOT,
    'h': (cirq.H, H01GateC),
    'x12': X12GateC(),
    'xminus': XminusGateC(),
    'xplus': XplusGateC(),
    'q': QGateC,
    'p1': P1GateC,
    'p2': P2GateC,
    'qubit_qutrit_crx_pluspi': CXiGateC(),
    'qubit_qutrit_crx_minuspi': CXiDagGateC()
}
