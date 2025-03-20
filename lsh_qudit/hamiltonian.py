from collections import namedtuple
from numbers import Number
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, ParameterExpression
from qiskit.circuit.library import RCCXGate
from .gates import (X12Gate, P1Gate, P2Gate, QGate, XplusGate, XminusGate,
                    ParameterValueType, QubitQutritCompositeGate,
                    QubitQutritCRxMinusPiGate, QubitQutritCRxPlusPiGate)
from .utils import QubitPlacement, diag_to_iz
from .parity_network import parity_network


BOSON_TRUNC = 3
BOSONIC_QUBITS = 2


if BOSONIC_QUBITS == 'qutrit':
    class CQGate(QubitQutritCompositeGate):
        """Controled Q gate."""
        gate_name = 'cq'

        def __init__(
            self,
            phi: ParameterValueType,
            label: Optional[str] = None
        ):
            super().__init__([phi], label=label)

        def _define(self):
            phi = self.params[0]
            qr = QuantumRegister(2, 'q')
            qc = QuantumCircuit(qr, name=self.name)
            qc.append(X12Gate(), [1])
            qc.append(QubitQutritCRxPlusPiGate(), [0, 1])
            qc.append(X12Gate(), [1])
            qc.append(QGate(-0.5 * phi), [1])
            qc.append(X12Gate(), [1])
            qc.append(QubitQutritCRxMinusPiGate(), [0, 1])
            qc.append(X12Gate(), [1])
            qc.append(QGate(0.5 * phi), [1])
            qc.rz(phi, 0)
            self.definition = qc

        def inverse(self, annotated: bool = False):
            return CQGate(-self.params[0])

        def __eq__(self, other):
            return isinstance(other, CQGate) and self._compare_parameters(other)

        def __array__(self, dtype=None, copy=None):
            if copy is False:
                raise ValueError("unable to avoid copy while creating an array as requested")
            phi = self.params[0]
            return np.diagflat(
                    [1., 1., 1., np.exp(1.j * phi), 1., np.exp(2.j * phi)]
                ).astype(dtype or complex)

    class CXminusGate(QubitQutritCompositeGate):
        """Controlled X- gate."""
        gate_name = 'cxminus'

        def __init__(
            self,
            label: Optional[str] = None
        ):
            super().__init__([], label=label)

        def _define(self):
            qr = QuantumRegister(2, 'q')
            qc = QuantumCircuit(qr, name=self.name)
            qc.append(QubitQutritCRxMinusPiGate(), [0, 1])
            qc.append(XminusGate(), [1])
            qc.append(QubitQutritCRxMinusPiGate(), [0, 1])
            qc.append(XplusGate(), [1])
            qc.rz(-0.5 * np.pi, 0)
            self.definition = qc

        def inverse(self, annotated: bool = False):
            return CXplusGate()

        def __eq__(self, other):
            return isinstance(other, CXminusGate)

        def __array__(self, dtype=None, copy=None):
            if copy is False:
                raise ValueError("unable to avoid copy while creating an array as requested")
            arr = np.diagflat([1., 1., 1., 0., 0., 0.]).astype(dtype or complex)
            arr[[1, 3], [3, 5]] = 1.
            arr[5, 1] = 1.j
            return arr

    class CXplusGate(QubitQutritCompositeGate):
        """Controlled X+ gate."""
        gate_name = 'cxplus'

        def __init__(
            self,
            label: Optional[str] = None
        ):
            super().__init__([], label=label)

        def _define(self):
            qr = QuantumRegister(2, 'q')
            qc = QuantumCircuit(qr, name=self.name)
            qc.rz(0.5 * np.pi, 0)
            qc.append(XminusGate(), [1])
            qc.append(QubitQutritCRxPlusPiGate(), [0, 1])
            qc.append(XplusGate(), [1])
            qc.append(QubitQutritCRxPlusPiGate(), [0, 1])
            self.definition = qc

        def inverse(self, annotated: bool = False):
            return CXminusGate()

        def __eq__(self, other):
            return isinstance(other, CXplusGate)

        def __array__(self, dtype=None, copy=None):
            if copy is False:
                raise ValueError("unable to avoid copy while creating an array as requested")
            arr = np.diagflat([1., 1., 1., 0., 0., 0.]).astype(dtype or complex)
            arr[[3, 5], [1, 3]] = 1.
            arr[1, 5] = -1.j
            return arr

    class CCXminusGate(Gate):
        """Doubly controlled Xminus gate using an ancilla qubit."""
        def __init__(self):
            super().__init__('ccxminus', 4, [])

        def _define(self):
            qr = QuantumRegister(4)
            qc = QuantumCircuit(qr, name=self.name)
            qc.append(RCCXGate(), [0, 1, 2])
            qc.append(CXminusGate(), [2, 3])
            qc.append(RCCXGate(), [0, 1, 2])
            self.definition = qc

        def inverse(self, annotated: bool = False):
            return CCXplusGate()

        def __eq__(self, other):
            return isinstance(other, CCXminusGate)

    class CCXplusGate(Gate):
        """Doubly controlled Xminus gate using an ancilla qubit."""
        def __init__(self):
            super().__init__('ccxplus', 4, [])

        def _define(self):
            qr = QuantumRegister(4)
            qc = QuantumCircuit(qr, name=self.name)
            qc.append(RCCXGate(), [0, 1, 2])
            qc.append(CXplusGate(), [2, 3])
            qc.append(RCCXGate(), [0, 1, 2])
            self.definition = qc

        def inverse(self, annotated: bool = False):
            return CCXminusGate()

        def __eq__(self, other):
            return isinstance(other, CCXplusGate)

HoppingTermConfig = namedtuple('HoppingTermConfig',
                               ['qubit_labels', 'boson_ops', 'default_qp', 'gl_states', 'indices'])


def physical_states(left_flux=None, right_flux=None, num_sites=1, as_multi=False):
    """Returns an array of AGL-satisfying states with optional boundary conditions.

    When as_multi=True, a 2-dimensional array is returned with the inner dimension corresponding
    to occupation numbers in the (i, o, l) order in the increasing site number.
    """
    shape = (2, 2, BOSON_TRUNC) * num_sites
    states = np.array(np.unravel_index(np.arange(np.prod(shape)), shape)).T
    agl_mask = np.ones(states.shape[:1], dtype=bool)
    for iconn in range(num_sites - 1):
        il = iconn * 3
        agl_mask &= np.equal((1 - states[:, il + 0]) * states[:, il + 1] + states[:, il + 2],
                             states[:, il + 3] * (1 - states[:, il + 4]) + states[:, il + 5])
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
            mask |= np.equal((1 - states[:, -3]) * states[:, -2] + states[:, -1], val)
        states = states[mask]

    if as_multi:
        return states
    return np.sum(states * np.cumprod((1,) + shape[-1:0:-1])[None, ::-1], axis=1)


def mass_term(
    site: int,
    time_step: Number | ParameterExpression,
    mass_mu: Union[Number, ParameterExpression],
    qp: Optional[QubitPlacement] = None
):
    if qp is None:
        qp = QubitPlacement([('i', site), ('o', site)])

    sign = -1 + 2 * (site % 2)

    circuit = QuantumCircuit(2)
    circuit.rz(sign * mass_mu * time_step, qp['i', site])
    circuit.rz(sign * mass_mu * time_step, qp['o', site])

    return circuit, qp, qp


def electric_12_term(
    site: int,
    time_step: Number | ParameterExpression,
    left_flux: Optional[int | tuple[int, ...]] = None,
    right_flux: Optional[int | tuple[int, ...]] = None,
    qp: Optional[QubitPlacement] = None
):
    gl_states = physical_states(left_flux=left_flux, right_flux=right_flux, as_multi=True)
    lvals = np.unique(gl_states[:, 2])
    if lvals.shape[0] == 1:
        if qp is None:
            qp = QubitPlacement([])
        return QuantumCircuit(0), qp, qp
    if np.amax(lvals) == 1:
        if qp is None:
            qp = QubitPlacement([('l', site)])
        circuit = QuantumCircuit(1)
        # H_E^(1) + H_E^(2)
        circuit.p(-0.75 * time_step, qp['l'])
        return circuit, qp, qp

    if BOSONIC_QUBITS == 'qutrit':
        if qp is None:
            qp = QubitPlacement([('d', site)])
        circuit = QuantumCircuit(1)
        # H_E^(1) + H_E^(2)
        circuit.append(P1Gate(-0.75 * time_step), [qp['d']])
        circuit.append(P2Gate(-2. * time_step), [qp['d']])
    else:
        if qp is None:
            qp = QubitPlacement([(f'd{i}', site) for i in range(BOSONIC_QUBITS)])
        circuit = QuantumCircuit(BOSONIC_QUBITS)
        # H_E^(1) + H_E^(2) = 1/2 n_l + 1/4 n_l^2
        if BOSON_TRUNC == 3 and BOSONIC_QUBITS == 2:
            # Optimization specific for BOSON_TRUNC=3: there should never be a state (1, 1), so we
            # can just make this a sum of local Zs
            circuit.p(-0.75 * time_step, qp['d0'])
            circuit.p(-2. * time_step, qp['d1'])
        else:
            coeffs = np.zeros(2 ** BOSONIC_QUBITS)
            nl = np.arange(BOSON_TRUNC)
            coeffs[:BOSON_TRUNC] = 0.5 * nl + 0.25 * np.square(nl)
            coeffs = coeffs.reshape((2,) * BOSONIC_QUBITS)
            circuit.compose(parity_network(diag_to_iz(coeffs) * time_step),
                            qubits=[qp[f'd{i}', site] for i in range(BOSONIC_QUBITS)], inplace=True)

    return circuit, qp, qp


def electric_3f_term(
    site: int,
    time_step: Number | ParameterExpression,
    qp: Optional[QubitPlacement] = None
):
    # 3/4 * (1 - n_i) * n_o
    if qp is None:
        qp = QubitPlacement([('i', site), ('o', site)])
    circuit = QuantumCircuit(2)

    circuit.x(qp['i'])
    circuit.cp(-0.75 * time_step, qp['i'], qp['o'])
    circuit.x(qp['i'])

    return circuit, qp, qp


def electric_3b_term(
    site: int,
    time_step: Number | ParameterExpression,
    left_flux: Optional[int | tuple[int, ...]] = None,
    right_flux: Optional[int | tuple[int, ...]] = None,
    qp: Optional[QubitPlacement] = None
):
    gl_states = physical_states(left_flux=left_flux, right_flux=right_flux, as_multi=True)
    lvals = np.unique(gl_states[:, 2])
    if lvals.shape[0] == 1:
        if lvals[0] == 0:
            qp = QubitPlacement([])
            return QuantumCircuit(0), qp, qp

        if qp is None:
            qp = QubitPlacement([('i', site), ('o', site)])
        circuit = QuantumCircuit(2)
        circuit.x(qp['i'])
        circuit.cp(-0.5 * lvals[0] * time_step, qp['i'], qp['o'])
        circuit.x(qp['i'])

    elif np.amax(lvals) == 1:
        if qp is None:
            if site % 2 == 0:
                labels = ['l', 'o', 'i']
            else:
                labels = ['i', 'l', 'o']
            qp = QubitPlacement([(lab, site) for lab in labels])

        circuit = QuantumCircuit(qp.num_qubits)
        # 1/2 n_l n_o (1 - n_i)
        coeffs = np.zeros((2, 2, 2))
        idx = [1, 1, 1]
        idx[2 - qp['i', site]] = 0
        coeffs[tuple(idx)] = 0.5
        coeffs = diag_to_iz(coeffs)
        circuit.compose(parity_network(coeffs * time_step), inplace=True)

    else:
        if qp is None:
            if site % 2 == 0:
                labels = ['l', 'o', 'i']
            else:
                labels = ['i', 'l', 'o']
            if BOSONIC_QUBITS == 'qutrit':
                labels.append('d')
            else:
                labels.extend(['d1', 'd0'])
            qp = QubitPlacement([(lab, site) for lab in labels])

        circuit = QuantumCircuit(qp.num_qubits)

        # 1/2 n_l n_o (1 - n_i)
        circuit.x(qp['i'])
        circuit.append(RCCXGate(), [qp['i', site], qp['o', site], qp['l', site]])
        if BOSONIC_QUBITS == 'qutrit':
            circuit.append(CQGate(-0.5 * time_step), [qp['l', site], qp['d', site]])
        else:
            coeffs = np.zeros((2, 2, 2))
            coeffs[0, 1, 1] = 0.5
            coeffs[1, 0, 1] = 1.
            circuit.compose(parity_network(diag_to_iz(coeffs) * time_step),
                            qubits=[qp['l', site], qp['d0', site], qp['d1', site]],
                            inplace=True)

        circuit.append(RCCXGate(), [qp['i', site], qp['o', site], qp['l', site]])
        circuit.x(qp['i'])

    return circuit, qp, qp


def hopping_term_config(term_type, site, left_flux=None, right_flux=None):
    """Return the x, y, p, q indices and simplifications.

    Lattice boundary conditions can impose certain constraints on the available LSH flux from the
    left or the right. Certain simplifications to U_SVD are possible depending on the resulting
    reductions of the matrix elements.

    To check for simplifications, we first identify the physical states (satisfying the Gauss's law
    and the boundary conditions) with non-coinciding n_i(r) and n_i(r+1). Then, within the list of
    such states,

    - If n_l(r) is 0 or 1 with n_o(r)=0, n_l(r)=1 occurring only when n_i(r+1)=1, the decrementer is
      replaced with X, and the o(r)-l(r) projector is replaced with [|0)(0|_l(r)]^{1-n_o(r)}.
    - Similarly, if n_l(r+1) is 0 or 1 with n_o(r+1)=1, n_l(r+1)=1 occurring only when n_i(r+1)=1,
      the decrementer is replaced with X, and the o(r+1)-l(r+1) projector is replaced with
      [|0)(0|_l(r+1)]^{n_o(r+1)}.
    - If n_o(r)=0, n_l(r)=Lambda never happens in the diagonal term, the projector is replaced with
      identity.
    - If n_o(r+1)=1, n_l(r+1)=Lambda never happens in the diagonal term, the projector is replaced
      with identity.
    - Excluding states with (n_i, n_o, n_l)_{r+1}=(1, 1, 0) or (0, 0, Lambda), if n_o(r) is
      identically 1, the n_i(r+1)-n_o(r) controlled decrementer and the corresponding projector are
      replaced with identity.
    - Similarly, excluding states with (n_i(r+1), n_o(r), n_l(r))=(1, 0, 0) or (0, 1, Lambda), if
      n_o(r+1) is identically 0, the n_i(r+1)-n_o(r+1) controlled decrementer and the corresponding
      projector are replaced with identity.
    """
    if term_type == 1:
        qubit_labels = {
            "x'": ('i', site),
            "x": ('o', site),
            "y'": ('i', site + 1),
            "y": ('o', site + 1),
            "p": ('l', site),
            "q": ('l', site + 1)
        }
        if BOSONIC_QUBITS == 'qutrit':
            qubit_labels |= {
                "pd": ('d', site),
                "qd": ('d', site + 1)
            }
        else:
            qubit_labels |= {f"pd{i}": (f'd{i}', site) for i in range(BOSONIC_QUBITS)}
            qubit_labels |= {f"qd{i}": (f'd{i}', site + 1) for i in range(BOSONIC_QUBITS)}

        indices = {"x'": 0, "x": 1, "y'": 3, "y": 4, "p": 2, "q": 5}
    else:
        qubit_labels = {
            "x'": ('o', site + 1),
            "x": ('i', site + 1),
            "y'": ('o', site),
            "y": ('i', site),
            "p": ('l', site + 1),
            "q": ('l', site)
        }
        if BOSONIC_QUBITS == 'qutrit':
            qubit_labels |= {
                "pd": ('d', site + 1),
                "qd": ('d', site)
            }
        else:
            qubit_labels |= {f"pd{i}": (f'd{i}', site + 1) for i in range(BOSONIC_QUBITS)}
            qubit_labels |= {f"qd{i}": (f'd{i}', site) for i in range(BOSONIC_QUBITS)}

        indices = {"x'": 4, "x": 3, "y'": 1, "y": 0, "p": 5, "q": 2}

    gl_states = physical_states(left_flux=left_flux, right_flux=right_flux, num_sites=2,
                                as_multi=True)
    states = gl_states[gl_states[:, indices["x'"]] != gl_states[:, indices["y'"]]]
    states_preproj = states

    def occnum(label, filt=None):
        if filt is None:
            return states[:, indices[label]]
        return states[filt, indices[label]]

    # Default: decrement p and q by lambda and project |Lambda) out
    controls = {'p': ('x', 0), 'q': ('y', 1)}
    # Check for simplifications following the projection on the other side
    ops_candidates = []
    for pboson in ['p', 'q']:
        states = states_preproj.copy()
        tboson = 'q' if pboson == 'p' else 'p'
        pfermion, pval = controls[pboson]
        tfermion, tval = controls[tboson]

        cyp = occnum("y'") == 1
        cf = occnum(pfermion) == pval
        if np.all(occnum(pboson, cf) < 2) and np.all(occnum(pboson, ~cyp & cf) == 0):
            # Special case of l=0,1
            ops = {pboson: ('X', 'zero')}
        else:
            ops = {pboson: ('lambda', 'Lambda')}

        mask = ~((cyp & cf & (occnum(pboson) == 0))
                 | (~cyp & cf & (occnum(pboson) == BOSON_TRUNC - 1)))
        if np.all(mask):
            ops[pboson] = (ops[pboson][0], 'id')
        else:
            states = states[mask]

        cyp = occnum("y'") == 1
        cf = occnum(tfermion) == tval
        if not np.any(cf):
            ops[tboson] = ('id', 'id')
            ops_candidates.append(ops)
            continue

        if np.all(occnum(tboson, cf) < 2) and np.all(occnum(tboson, ~cyp & cf) == 0):
            # Special case of l=0,1
            ops[tboson] = ('X', 'zero')
        else:
            ops[tboson] = ('lambda', 'Lambda')

        mask = ~((cyp & cf & (occnum(tboson) == 0))
                 | (~cyp & cf & (occnum(tboson) == BOSON_TRUNC - 1)))
        if np.all(mask):
            ops[tboson] = (ops[tboson][0], 'id')

        ops_candidates.append(ops)

    # Evaluate and select the better site to project on according to the following score matrix:
    #                test
    #  |    |id-id X-id λ-id X-0 λ-Λ
    # p|X-id|    0    2    3   6   9
    # r|λ-id|    1    3    4   7  10
    # o|X-0 |    5    6    7  11  12
    # j|λ-Λ |    8    9   10  12  13
    op_combs = [('id', 'id'), ('X', 'id'), ('lambda', 'id'), ('X', 'zero'), ('lambda', 'Lambda')]
    proj_keys = {c: i for i, c in enumerate(op_combs[1:])}
    test_keys = {c: i for i, c in enumerate(op_combs)}
    scores = [
        [0, 2, 3, 6, 9],
        [1, 3, 4, 7, 10],
        [5, 6, 7, 11, 12],
        [8, 9, 10, 12, 13]
    ]
    s0 = scores[proj_keys[ops_candidates[0]['p']]][test_keys[ops_candidates[0]['q']]]
    s1 = scores[proj_keys[ops_candidates[1]['q']]][test_keys[ops_candidates[1]['p']]]
    if s0 <= s1:
        boson_ops = ops_candidates[0]
    else:
        boson_ops = ops_candidates[1]

    match (term_type, site % 2):
        case (1, 0):
            labels = ["p", "x", "x'", "y'", "q", "y"]
        case (1, 1):
            labels = ["x", "p", "x'", "q", "y'", "y"]
        case (2, 0):
            labels = ["q", "y", "y'", "x'", "p", "x"]
        case (2, 1):
            labels = ["y", "q", "y'", "p", "x'", "x"]
    if boson_ops['p'][0] == 'id':
        labels.remove("p")
    elif boson_ops['p'][0] == 'lambda':
        if BOSONIC_QUBITS == 'qutrit':
            labels.append("pd")
        else:
            labels.extend(f"pd{i}" for i in range(BOSONIC_QUBITS))
    if boson_ops['q'][0] == 'id':
        labels.remove("q")
    elif boson_ops['q'][0] == 'lambda':
        if BOSONIC_QUBITS == 'qutrit':
            labels.append("qd")
        else:
            labels.extend(f"qd{i}" for i in range(BOSONIC_QUBITS))

    default_qp = QubitPlacement([qubit_labels[label] for label in labels])

    return HoppingTermConfig(qubit_labels, boson_ops, default_qp, gl_states, indices)


def hopping_term(
    term_type,
    site,
    time_step,
    interaction_x,
    left_flux=None,
    right_flux=None,
    qp=None,
    with_barrier=False
):
    config = hopping_term_config(term_type, site, left_flux=left_flux, right_flux=right_flux)

    usvd_circuit, init_p, final_p = hopping_usvd(term_type, site, config=config, qp=qp)
    diag_circuit = hopping_diagonal_term(term_type, site, time_step, interaction_x, config=config,
                                         qp=final_p)[0]

    circuit = QuantumCircuit(init_p.num_qubits)
    circuit.compose(usvd_circuit, inplace=True)
    if with_barrier:
        circuit.barrier()
    circuit.compose(diag_circuit, inplace=True)
    if with_barrier:
        circuit.barrier()
    circuit.compose(usvd_circuit.inverse(), inplace=True)

    return circuit, init_p, init_p


def hopping_usvd(
    term_type,
    site,
    left_flux=None,
    right_flux=None,
    config=None,
    qp=None
):
    if config is None:
        config = hopping_term_config(term_type, site, left_flux=left_flux, right_flux=right_flux)
    if qp is None:
        qp = config.default_qp
    init_p = qp

    circuit = QuantumCircuit(init_p.num_qubits)

    def qpl(label):
        return qp[config.qubit_labels[label]]

    def swap(label1, label2):
        circuit.swap(qpl(label1), qpl(label2))
        return qp.swap(config.qubit_labels[label1], config.qubit_labels[label2])

    if (term_type, site % 2) in [(1, 0), (2, 1)]:
        # x' as the SVD key qubit
        # Default QP
        # (1, 0): ["p", "x", "x'", "y'", "q", "y"]
        # (2, 1): ["y", "q", "y'", "p", "x'", "x"]
        circuit.x(qpl("x'"))

        circuit.x(qpl("x"))
        if config.boson_ops['p'][0] == 'X':
            circuit.ccx(qpl("x'"), qpl("x"), qpl("p"))
        elif config.boson_ops['p'][0] == 'lambda':
            _hopping_usvd_decrementer(circuit, qpl, ["x'", "x"], "p")
        circuit.x(qpl("x"))

        # Bring y' next to x' and do CX
        if term_type == 2:
            qp = swap("x'", "p")
        circuit.cx(qpl("x'"), qpl("y'"))

        # Current QP
        # (1, 0): ["p", "x", "x'", "y'", "q", "y"]
        # (2, 1): ["y", "q", "y'", "x'", "p", "x"]
        # Need x' y q contiguous
        qp = swap("y'", "x'")
        if config.boson_ops['q'][0] == 'X':
            circuit.ccx(qpl("x'"), qpl("y"), qpl("q"))
        elif config.boson_ops['q'][0] == 'lambda':
            _hopping_usvd_decrementer(circuit, qpl, ["x'", "y"], "q")

        circuit.x(qpl("x'"))
        circuit.h(qpl("x'"))
    else:
        # y' as the SVD key qubit
        # Default QP
        # (1, 1): ["x", "p", "x'", "q", "y'", "y"]
        # (2, 0): ["q", "y", "y'", "x'", "p", "x"]
        if config.boson_ops['q'][0] == 'X':
            circuit.ccx(qpl("y'"), qpl("y"), qpl("q"))
        elif config.boson_ops['q'][0] == 'lambda':
            _hopping_usvd_decrementer(circuit, qpl, ["y'", "y"], "q")

        if term_type == 1:
            qp = swap("y'", "q")
        circuit.cx(qpl("y'"), qpl("x'"))

        # Current QP
        # (1, 1): ["x", "p", "x'", "y'", "q", "y"]
        # (2, 0): ["q", "y", "y'", "x'", "p", "x"]
        # Need y' x p contiguous
        qp = swap("y'", "x'")
        circuit.x(qpl("x"))
        if config.boson_ops['p'][0] == 'X':
            circuit.ccx(qpl("y'"), qpl("x"), qpl("p"))
        elif config.boson_ops['q'][0] == 'lambda':
            _hopping_usvd_decrementer(circuit, qpl, ["y'", "x"], "p")
        circuit.x(qpl("x"))

        circuit.h(qpl("y'"))

    return circuit, init_p, qp


def _hopping_usvd_decrementer(circuit, qpl, fermions, boson):
    # While the RCCX gate is conceptually symmetric with respect to the two controls,
    # transpiler InverseCancellation pass cannot handle CCXplus-CCXminus with
    # different ordering. We therefore need to manually specify the control qubit order
    # here as (far control)-(near control)-(target).
    qargs = sorted((qpl(f) for f in fermions), key=lambda q: -abs(qpl(boson) - q))
    qargs.append(qpl(boson))

    if BOSONIC_QUBITS == 'qutrit':
        circuit.append(CCXminusGate(), qargs + [qpl(boson + "d")])
    else:
        circuit.append(RCCXGate(), qargs)
        if BOSON_TRUNC == 2 ** BOSONIC_QUBITS:
            raise NotImplementedError('QFT-based decrementer')
        elif BOSON_TRUNC == 3 and BOSONIC_QUBITS == 2:
            circuit.cx(*[qpl(boson + suffix) for suffix in ['', 'd0']])
            circuit.ccx(*[qpl(boson + suffix) for suffix in ['', 'd0', 'd1']])
            circuit.ccx(*[qpl(boson + suffix) for suffix in ['', 'd1', 'd0']])
        else:
            raise NotImplementedError('No decrementer circuit for'
                                      f' BOSON_TRUNC={BOSON_TRUNC} and'
                                      f' BOSONIC_QUBITS={BOSONIC_QUBITS}')
        circuit.append(RCCXGate(), qargs)


def hopping_diagonal_op(
    term_type,
    site,
    left_flux=None,
    right_flux=None,
    config=None,
):
    """Compute the diagonal term for the given term type and site number as a sum of Pauli products.
    """
    if config is None:
        config = hopping_term_config(term_type, site, left_flux=left_flux, right_flux=right_flux)

    # Pass the Gauss's law-satisfying states through the decrementers in Usvd
    states = config.gl_states.copy()
    if (term_type, site % 2) in [(1, 0), (2, 1)]:
        idx = config.indices["x'"]
        mask_fp = states[:, idx] == 0
        idx = config.indices["y'"]
        states[mask_fp, idx] *= -1
        states[mask_fp, idx] += 1
    else:
        idx = config.indices["y'"]
        mask_fp = states[:, idx] == 1
        idx = config.indices["x'"]
        states[mask_fp, idx] *= -1
        states[mask_fp, idx] += 1

    mask = mask_fp & (states[:, config.indices["y"]] == 1)
    idx = config.indices["q"]
    if config.boson_ops['q'][0] == 'lambda':
        states[mask, idx] -= 1
        states[mask, idx] %= BOSON_TRUNC
    elif config.boson_ops['q'][0] == 'X':
        states[mask, idx] *= -1
        states[mask, idx] += 1
    mask = mask_fp & (states[:, config.indices["x"]] == 0)
    idx = config.indices["p"]
    if config.boson_ops['p'][0] == 'lambda':
        states[mask, idx] -= 1
        states[mask, idx] %= BOSON_TRUNC
    elif config.boson_ops['p'][0] == 'X':
        states[mask, idx] *= -1
        states[mask, idx] += 1

    # Apply the projectors and construct the diagonal operator
    # Start with the diagonal function
    diag_fn = np.sqrt((np.arange(BOSON_TRUNC)[:, None, None] + np.arange(1, 3)[None, :, None])
                      / (np.arange(BOSON_TRUNC)[:, None, None] + np.arange(1, 3)[None, None, :]))
    diag_op = np.tile(diag_fn[..., None, None], (1, 1, 1, 2, 2))
    if config.boson_ops['p'][1] == 'id':
        # Diag op is expressed in terms of n_q
        op_dims = ['q']
    elif config.boson_ops['q'][1] == 'id':
        op_dims = ['p']
    elif (term_type, site % 2) == (1, 0):
        op_dims = ['q']
    elif (term_type, site % 2) == (2, 0):
        op_dims = ['p']
    else:
        op_dims = ['p']
    op_dims += ["x", "y", "x'", "y'"]

    if (term_type, site % 2) in [(1, 0), (2, 1)]:
        # Project onto y' == 0 and multiply Zx'
        states = states[states[:, config.indices["y'"]] == 0]
        diag_op *= np.expand_dims(np.array([1., 0.]), [0, 1, 2, 3])
        diag_op *= np.expand_dims(np.array([1., -1.]), [0, 1, 2, 4])
    else:
        # Project onto x' == 1 and multiply Zy'
        states = states[states[:, config.indices["x'"]] == 1]
        diag_op *= np.expand_dims(np.array([0., 1.]), [0, 1, 2, 4])
        diag_op *= np.expand_dims(np.array([1., -1.]), [0, 1, 2, 3])

    # Zx
    diag_op *= np.expand_dims(np.array([1., -1.]), [0, 2, 3, 4])

    for boson, fermion, cval in [('p', 'x', 0), ('q', 'y', 1)]:
        projector = config.boson_ops[boson][1]
        if projector == 'id':
            continue
        mask = states[:, config.indices[fermion]] == cval
        if projector == 'Lambda':
            mask &= states[:, config.indices[boson]] == BOSON_TRUNC - 1
        else:
            mask &= states[:, config.indices[boson]] != 0
        states = states[~mask]

        fdim = op_dims.index(fermion)
        diag_op = np.moveaxis(diag_op, fdim, 1)
        if projector == 'Lambda':
            diag_op[-1, cval] = 0.
        else:
            diag_op[1:, cval] = 0.
        diag_op = np.moveaxis(diag_op, 1, fdim)

    for fermion in ["x", "y"]:
        unique = np.unique(states[:, config.indices[fermion]])
        if unique.shape[0] == 1:
            fdim = op_dims.index(fermion)
            diag_op = np.moveaxis(diag_op, fdim, 0)[unique[0]]
            op_dims.pop(fdim)

    if config.boson_ops[op_dims[0]][0] != 'lambda':
        diag_op = diag_op[:2]

    return diag_op, op_dims


def hopping_diagonal_term(
    term_type,
    site,
    time_step,
    interaction_x,
    left_flux=None,
    right_flux=None,
    config=None,
    qp=None
):
    """Construct the circuit for the diagonal term.

    Using the hopping term config for the given boundary condition, identify the qubits that appear
    in the circuit, compute the Z rotation angles, and compose the parity network circuit.
    """
    diag_op, op_dims = hopping_diagonal_op(term_type, site,
                                           left_flux=left_flux, right_flux=right_flux,
                                           config=config)

    # Multiply the diag_op with the physical parameters (can be Parameters)
    shape = diag_op.shape
    diag_op = np.array([interaction_x * time_step * c for c in diag_op.reshape(-1)])
    diag_op = diag_op.reshape(shape)

    if config is None:
        config = hopping_term_config(term_type, site, left_flux=left_flux, right_flux=right_flux)
    if qp is None:
        qp = config.default_qp

    circuit = QuantumCircuit(qp.num_qubits)

    def qpl(label):
        return qp[config.qubit_labels[label]]

    def swap(label1, label2):
        circuit.swap(qpl(label1), qpl(label2))
        return qp.swap(config.qubit_labels[label1], config.qubit_labels[label2])

    # op_dims as qubit labels
    reverse_map = {value: key for key, value in config.qubit_labels.items()}

    # Final QP of Usvd
    # (1, 0): ["p", "x", "y'", "x'", "q", "y"]
    # (1, 1): ["x", "p", "y'", "x'", "q", "y"]
    # (2, 0): ["q", "y", "x'", "y'", "p", "x"]
    # (2, 1): ["y", "q", "x'", "y'", "p", "x"]
    match (term_type, site % 2, config.boson_ops["p"][1], config.boson_ops["q"][1]):
        case (1, 0, 'id', 'zero'):
            swaps = []
        case (1, 1, 'zero', 'id'):
            swaps = [("q", "y")]
        case (2, 0, 'zero', 'id'):
            swaps = []
        case (2, 1, 'id', 'zero'):
            swaps = [("p", "x")]
        case _:
            raise NotImplementedError('General diagonal term not implemented')

    for q1, q2 in swaps:
        qp = swap(q1, q2)
    # Permute the axis of diag_ops to match the QP order
    ord_op_dims = [reverse_map[lab] for lab in qp.qubit_labels[::-1] if reverse_map[lab] in op_dims]
    perm = tuple(op_dims.index(lab) for lab in ord_op_dims)
    circ = parity_network(diag_to_iz(diag_op.transpose(perm)))
    circuit.compose(circ, qubits=[qpl(lab) for lab in ord_op_dims[::-1]], inplace=True)
    for q1, q2 in swaps[::-1]:
        qp = swap(q1, q2)

    return circuit, qp, qp
