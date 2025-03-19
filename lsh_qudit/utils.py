"""Tools."""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import SparsePauliOp


class QubitPlacement:
    """Qubit-occupation number correspondence tracker."""
    def __init__(self, qubit_labels, swap_history=()):
        self.qubit_labels = tuple(qubit_labels)
        self.swap_history = swap_history

    @property
    def num_qubits(self):
        return len(self.qubit_labels)

    def swap(self, label1, label2):
        i1 = self[label1]
        i2 = self[label2]
        imin = min(i1, i2)
        imax = max(i1, i2)
        qubit_labels = (self.qubit_labels[:imin]
                        + (self.qubit_labels[imax],)
                        + self.qubit_labels[imin + 1:imax]
                        + (self.qubit_labels[imin],)
                        + self.qubit_labels[imax + 1:])
        return QubitPlacement(qubit_labels, self.swap_history + ((imin, imax),))

    def __getitem__(self, index):
        if isinstance(index, tuple):
            try:
                return next(i for i, l in enumerate(self.qubit_labels) if l == index)
            except StopIteration as ex:
                raise KeyError(index) from ex
        if isinstance(index, str):
            return [i for i, l in enumerate(self.qubit_labels) if l[0] == index]
        raise IndexError('Invalid index')


def sort_qubits(circuit, initial_placement):
    """Reorder the qubits into i, o, l order in increasing site number."""
    sites = set(label[1] for label in initial_placement.qubit_labels)
    qregs = []
    mapping = [None] * initial_placement.num_qubits
    for isite in sorted(sites):
        for name in ['i', 'o', 'l', 'd']:
            try:
                i_in = next(i for i, l in enumerate(initial_placement.qubit_labels)
                            if l == (name, isite))
            except StopIteration:
                continue

            mapping[i_in] = len(qregs)
            qregs.append(QuantumRegister(1, name=f'{name}({isite})'))

    circ = QuantumCircuit(*qregs)
    circ.compose(circuit, qubits=mapping, inplace=True)
    return circ


def draw_circuit(circuit, initial_placement, *args, reorder=True, **kwargs):
    if reorder:
        circuit = sort_qubits(circuit, initial_placement)
    else:
        qregs = []
        for name, isite in initial_placement.qubit_labels:
            qregs.append(QuantumRegister(1, name=f'{name}({isite})'))
        circ = QuantumCircuit(*qregs)
        circ.compose(circuit, inplace=True)
        circuit = circ
    return circuit.draw('mpl', *args, **kwargs)


def count_gates(qc: QuantumCircuit):
    gate_count = {qubit: 0 for qubit in qc.qubits}
    for gate in qc.data:
        for qubit in gate.qubits:
            gate_count[qubit] += 1
    return gate_count


def remove_idle_wires(qc: QuantumCircuit, inplace=False, flatten=False, keep_qubits=None):
    if not inplace:
        qc = qc.copy()

    gate_count = count_gates(qc)
    for qubit, count in gate_count.items():
        if keep_qubits and qubit in keep_qubits:
            continue
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
    """Embed a square matrix op into a larger square matrix given by shape. Essentially a series of
    np.krons with identity matrices.
    Args:
        op: Square matrix.
        shape: Qudit dimensions from major to minor.
        qubits: Qubit ids (minor=0) to embed the op into.
    """
    shape = tuple(shape)
    if isinstance(qubits, int):
        qubits = (qubits,)
    full_nq = len(shape)
    full_dim = np.prod(shape)
    op_nq = len(qubits)
    op_shape = tuple(shape[::-1][i] for i in qubits)
    op_dim = np.prod(op_shape)
    idle_shape = tuple(dim for i, dim in enumerate(shape) if full_nq - i - 1 not in qubits)
    idle_dim = int(np.prod(idle_shape))
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


def clean_array(arr):
    return (np.where(np.isclose(arr.real, 0.), 0., arr.real)
            + 1.j * np.where(np.isclose(arr.imag, 0.), 0., arr.imag))


def diag_to_iz(arr):
    conv = {
        2: np.array([[1., 1.], [1., -1.]]) / 2.,
        3: np.array([[1., 1., 1.], [2., -1., -1.], [-1., -1., 2.]]) / 3.
    }
    for idim, size in enumerate(arr.shape):
        arr = np.moveaxis(np.tensordot(conv[size], arr, (1, idim)), 0, idim)
    return arr


def to_bin(idx, nbits):
    return tuple((idx >> np.arange(nbits)[::-1]) % 2)
