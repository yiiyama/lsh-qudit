from collections import namedtuple
from numbers import Number
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.quantum_info import SparsePauliOp
from qutrit_experiments.gates import (X12Gate, P1Gate, P2Gate, QGate, XplusGate,
                                      XminusGate, QubitQutritCRxMinusPiGate,
                                      QubitQutritCRxPlusPiGate)
from .utils import QubitPlacement, op_matrix, physical_states, diag_to_iz
from .parity_network import parity_network


sqrt2 = np.sqrt(2.)
sqrt3 = np.sqrt(3.)

paulii = SparsePauliOp('I')
paulix = SparsePauliOp('X')
pauliy = SparsePauliOp('Y')
pauliz = SparsePauliOp('Z')
hadamard = (paulix + pauliz) / sqrt2
p0 = SparsePauliOp(['I', 'Z'], [0.5, 0.5])
p1 = SparsePauliOp(['I', 'Z'], [0.5, -0.5])

sigmaplus = (paulix + 1.j * pauliy) * 0.5  # |0><1|
sigmaminus = (paulix - 1.j * pauliy) * 0.5  # |0><1|
cyc_incr = np.zeros((3, 3), dtype=np.complex128)
cyc_incr[[0, 1, 2], [2, 0, 1]] = 1.
incr = cyc_incr @ np.diagflat([1., 1., 0.])
cincr = np.zeros((6, 6), dtype=np.complex128)
cincr[:3, :3] = np.eye(3, dtype=np.complex128)
cincr[3:, 3:] = incr
ocincr = np.zeros((6, 6), dtype=np.complex128)
ocincr[:3, :3] = incr
ocincr[3:, 3:] = np.eye(3, dtype=np.complex128)

rccx_ctc = QuantumCircuit(3)
rccx_ctc.ry(-np.pi / 4., 1)
rccx_ctc.cx(0, 1)
rccx_ctc.ry(-np.pi / 4., 1)
rccx_ctc.cx(2, 1)
rccx_ctc.ry(np.pi / 4., 1)
rccx_ctc.cx(0, 1)
rccx_ctc.ry(np.pi / 4., 1)

rccx_cct = QuantumCircuit(3)
rccx_cct.ry(-np.pi / 4., 2)
rccx_cct.cx(2, 1)
rccx_cct.cx(1, 2)
rccx_cct.ry(-np.pi / 4., 1)
rccx_cct.cx(0, 1)
rccx_cct.ry(np.pi / 4., 1)
rccx_cct.cx(1, 2)
rccx_cct.cx(2, 1)
rccx_cct.ry(np.pi / 4., 2)

cziy = QuantumCircuit(3)
cziy.h(2)
cziy.t(2)
cziy.cx(1, 2)
cziy.tdg(2)
cziy.cx(0, 2)
cziy.t(2)
cziy.cx(1, 2)
cziy.tdg(2)
cziy.cp(np.pi / 2., 1, 0)
cziy.h(2)

BT = 3
BOSON_NLEVEL = 3

hopping_shape = (2, 2, BT, 2, 2, BT)  # i(r), o(r), l(r), i(r+1), o(r+1), l(r+1)
diag_fn = np.sqrt((np.arange(BT)[:, None, None] + np.arange(1, 3)[None, :, None])
                  / (np.arange(BT)[:, None, None] + np.arange(1, 3)[None, None, :]))

hi1_mat = op_matrix(np.diagflat(diag_fn), hopping_shape, (3, 4, 1))
hi1_mat = op_matrix(cincr, hopping_shape, (1, 0)) @ hi1_mat
hi1_mat = op_matrix(ocincr, hopping_shape, (4, 3)) @ hi1_mat
hi1_mat = op_matrix(sigmaminus, hopping_shape, 2) @ hi1_mat
hi1_mat = op_matrix(pauliz, hopping_shape, 4) @ hi1_mat
hi1_mat = op_matrix(sigmaplus, hopping_shape, 5) @ hi1_mat
hi1_mat += hi1_mat.conjugate().T

hi2_mat = op_matrix(np.diagflat(diag_fn), hopping_shape, (0, 2, 5))
hi2_mat = op_matrix(cincr, hopping_shape, (5, 3)) @ hi2_mat
hi2_mat = op_matrix(ocincr, hopping_shape, (2, 0)) @ hi2_mat
hi2_mat = op_matrix(sigmaminus, hopping_shape, 4) @ hi2_mat
hi2_mat = op_matrix(pauliz, hopping_shape, 2) @ hi2_mat
hi2_mat = op_matrix(sigmaplus, hopping_shape, 1) @ hi2_mat
hi2_mat += hi2_mat.conjugate().T

HoppingTermConfig = namedtuple('HoppingTermConfig',
                               ['qubit_labels', 'boson_ops', 'default_qp', 'gl_states', 'indices'])


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
    elif np.amax(lvals) == 1:
        if qp is None:
            qp = QubitPlacement([('l', site)])
        circuit = QuantumCircuit(1)
        # H_E^(1) + H_E^(2)
        circuit.p(-0.75 * time_step, qp['l'])
        return circuit, qp, qp
    elif BOSON_NLEVEL == 2:
        if qp is None:
            qp = QubitPlacement([('d0', site), ('d1', site)])
        circuit = QuantumCircuit(2)
        # H_E^(1) + H_E^(2) = 1/2 n_l + 1/4 n_l^2
        coeffs = np.zeros((2, 2))
        coeffs[0, 1] = 0.75
        coeffs[1, 0] = 2.
        coeffs = diag_to_iz(coeffs)
        circuit.rz(coeffs[0, 1] * time_step, qp['d0'])
        circuit.rz(coeffs[1, 0] * time_step, qp['d1'])
        circuit.cx(qp['d0'], qp['d1'])
        circuit.rz(coeffs[1, 1] * time_step, qp['d1'])
        circuit.cx(qp['d0'], qp['d1'])
    elif BOSON_NLEVEL == 3:
        if qp is None:
            qp = QubitPlacement([('d', site)])
        circuit = QuantumCircuit(1)
        # H_E^(1) + H_E^(2)
        circuit.append(P1Gate(-0.75 * time_step), [qp['d']])
        circuit.append(P2Gate(-2. * time_step), [qp['d']])

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
            qp = QubitPlacement([('i', site), ('o', site), ('l', site)])
        circuit = QuantumCircuit(3)
        # 1/2 n_l n_o (1 - n_i)
        coeffs = np.zeros((2, 2, 2))
        coeffs[1, 1, 0] = 0.5
        coeffs = diag_to_iz(coeffs)
        circuit.compose(parity_network(coeffs * time_step),
                        qubits=[qp['i', site], qp['o', site], qp['l', site]],
                        inplace=True)

    else:
        if BOSON_NLEVEL == 2:
            if qp is None:
                qp = QubitPlacement([('i', site), ('l', site), ('d0', site), ('d1', site),
                                     ('o', site)])
            circuit = QuantumCircuit(5)
        elif BOSON_NLEVEL == 3:
            if qp is None:
                qp = QubitPlacement([('i', site), ('l', site), ('d', site), ('o', site)])
            circuit = QuantumCircuit(4)
        # 1/2 n_l n_o (1 - n_i)
        circuit.x(qp['i'])
        circuit.compose(rccx_ctc, qubits=[qp['i', site], qp['l', site], qp['o', site]],
                        inplace=True)
        if BOSON_NLEVEL == 2:
            coeffs = np.zeros((2, 2, 2))
            coeffs[0, 1, 1] = 1.
            coeffs[1, 0, 1] = 2.
            coeffs *= 0.5
            coeffs = diag_to_iz(coeffs)
            circuit.compose(parity_network(coeffs * time_step),
                            qubits=[qp['l', site], qp['d0', site], qp['d1', site]],
                            inplace=True)
        elif BOSON_NLEVEL == 3:
            circuit.append(X12Gate(), [qp['d', site]])
            circuit.append(QubitQutritCRxPlusPiGate(), [qp['l', site], qp['d', site]])
            circuit.append(X12Gate(), [qp['d', site]])
            circuit.append(QGate(0.25 * time_step), [qp['d', site]])
            circuit.append(X12Gate(), [qp['d', site]])
            circuit.append(QubitQutritCRxMinusPiGate(), [qp['l', site], qp['d', site]])
            circuit.append(X12Gate(), [qp['d', site]])
            circuit.append(QGate(-0.25 * time_step), [qp['d', site]])
        circuit.rz(-0.5 * time_step, qp['l', site])
        circuit.compose(rccx_ctc.inverse(), qubits=[qp['i', site], qp['l', site], qp['o', site]],
                        inplace=True)
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
            "q": ('l', site + 1),
            "pd": ('d', site),
            "qd": ('d', site + 1)
        }
        indices = {"x'": 0, "x": 1, "y'": 3, "y": 4, "p": 2, "q": 5}
    else:
        qubit_labels = {
            "x'": ('o', site + 1),
            "x": ('i', site + 1),
            "y'": ('o', site),
            "y": ('i', site),
            "p": ('l', site + 1),
            "q": ('l', site),
            "pd": ('d', site + 1),
            "qd": ('d', site)
        }
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

        mask = ~((cyp & cf & (occnum(pboson) == 0)) | (~cyp & cf & (occnum(pboson) == BT - 1)))
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

        mask = ~((cyp & cf & (occnum(tboson) == 0)) | (~cyp & cf & (occnum(tboson) == BT - 1)))
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
            labels = ["p", "pd", "x", "x'", "y'", "q", "qd", "y"]
        case (1, 1):
            labels = ["x", "p", "pd", "x'", "q", "qd", "y'", "y"]
        case (2, 0):
            labels = ["q", "qd", "y", "y'", "x'", "p", "pd", "x"]
        case (2, 1):
            labels = ["y", "q", "qd", "y'", "p", "pd", "x'", "x"]
    if boson_ops['p'][0] == 'id':
        labels.remove("p")
    if boson_ops['p'][0] != 'lambda':
        labels.remove("pd")
    if boson_ops['q'][0] == 'id':
        labels.remove("q")
    if boson_ops['q'][0] != 'lambda':
        labels.remove("qd")

    default_qp = QubitPlacement([qubit_labels[label] for label in labels])

    return HoppingTermConfig(qubit_labels, boson_ops, default_qp, gl_states, indices)


def hopping_term(
    term_type,
    site,
    time_step,
    interaction_x,
    left_flux=None,
    right_flux=None,
    qp=None
):
    config = hopping_term_config(term_type, site, left_flux=left_flux, right_flux=right_flux)

    usvd_circuit, init_p, final_p = hopping_usvd(term_type, site, config=config, qp=qp)
    diag_circuit = hopping_diagonal_term(term_type, site, time_step, interaction_x, config=config,
                                         qp=final_p)[0]

    circuit = QuantumCircuit(init_p.num_qubits)
    circuit.compose(usvd_circuit, inplace=True)
    circuit.compose(diag_circuit, inplace=True)
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

    def rel_phase_toffoli(lf1, lf2, lb, inverse):
        if qpl(lb) < qpl(lf1) and qpl(lb) < qpl(lf2):
            circuit.compose(rccx_cct.inverse() if inverse else rccx_cct,
                            qubits=list(sorted([qpl(lf1), qpl(lf2), qpl(lb)], reverse=True)),
                            inplace=True)
        elif qpl(lb) > qpl(lf1) and qpl(lb) > qpl(lf2):
            circuit.compose(rccx_cct.inverse() if inverse else rccx_cct,
                            qubits=list(sorted([qpl(lf1), qpl(lf2), qpl(lb)])),
                            inplace=True)
        else:
            circuit.compose(rccx_ctc.inverse() if inverse else rccx_ctc,
                            qubits=[qpl(lf1), qpl(lb), qpl(lf2)],
                            inplace=True)

    # Default QP (term_type, site parity):
    # (1, 0): ["p", "pd", "x", "x'", "y'", "q", "qd", "y"]
    # (1, 1): ["x", "p", "pd", "x'", "q", "qd", "y'", "y"]
    # (2, 0): ["q", "qd", "y", "y'", "x'", "p", "pd", "x"]
    # (2, 1): ["y", "q", "qd", "y'", "p", "pd", "x'", "x"]

    if config.boson_ops['q'][0] == 'X':
        circuit.ccx(qpl("y'"), qpl("y"), qpl("q"))
    elif config.boson_ops['q'][0] == 'lambda':
        rel_phase_toffoli("y", "y'", "q", False)
        circuit.append(QubitQutritCRxMinusPiGate(), [qpl("q"), qpl("qd")])
        circuit.append(XminusGate(), [qpl("qd")])
        circuit.append(QubitQutritCRxMinusPiGate(), [qpl("q"), qpl("qd")])
        circuit.append(XplusGate(), [qpl("qd")])
        circuit.rz(-0.5 * np.pi, qpl("q"))
        rel_phase_toffoli("y", "y'", "q", True)

    # Bring y' next to x' and do CX
    match (term_type, site % 2):
        case (1, 1):
            qp = swap("y'", "q")
        case (2, 1):
            qp = swap("x'", "p")

    # In principle we do a CX between y' and x' here, but will cancel with the SWAP that comes right
    # after for many configurations
    # circuit.cx(qpl("y'"), qpl("x'"))

    # Current QP
    # (1, 0): ["p", "pd", "x", "x'", "y'", "q", "qd", "y"]
    # (1, 1): ["x", "p", "pd", "x'", "y'", "q", "qd", "y"]
    # (2, 0): ["q", "qd", "y", "y'", "x'", "p", "pd", "x"]
    # (2, 1): ["y", "q", "qd", "y'", "x'", "pd", "p", "x"]

    if config.boson_ops['p'][0] == 'X':
        # Need y' x p contiguous (order doesn't matter)
        # qp = swap("y'", "x'")  # Canceling the CX above
        circuit.cx(qpl("x'"), qpl("y'"))
        circuit.cx(qpl("y'"), qpl("x'"))
        qp = qp.swap(config.qubit_labels["y'"], config.qubit_labels["x'"])
        circuit.x(qpl("x"))
        circuit.ccx(qpl("y'"), qpl("x"), qpl("p"))
        circuit.x(qpl("x"))
    elif config.boson_ops['p'][0] == 'lambda':
        # y'-p-x desired; contiguous is fine. p must be on the original qubit
        match (term_type, site % 2):
            case (1, 0) | (1, 1) | (2, 0):
                # qp = swap("y'", "x'")
                circuit.cx(qpl("x'"), qpl("y'"))
                circuit.cx(qpl("y'"), qpl("x'"))
                qp = qp.swap(config.qubit_labels["y'"], config.qubit_labels["x'"])
            case (2, 1):
                # Deferred CX from above
                circuit.cx(qpl("y'"), qpl("x'"))
                qp = swap("x'", "p")
                qp = swap("x'", "x")
        circuit.x(qpl("x"))
        rel_phase_toffoli("y'", "x", "p", False)
        circuit.append(QubitQutritCRxMinusPiGate(), [qpl("p"), qpl("pd")])
        circuit.append(XminusGate(), [qpl("pd")])
        circuit.append(QubitQutritCRxMinusPiGate(), [qpl("p"), qpl("pd")])
        circuit.append(XplusGate(), [qpl("pd")])
        circuit.rz(-0.5 * np.pi, qpl("p"))
        rel_phase_toffoli("y'", "x", "p", True)
        circuit.x(qpl("x"))
    else:
        # Deferred CX from above
        circuit.cx(qpl("y'"), qpl("x'"))

    circuit.h(qpl("y'"))

    return circuit, init_p, qp


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
    idx = config.indices["y'"]
    mask_yp = states[:, idx] == 1
    idx = config.indices["x'"]
    states[mask_yp, idx] *= -1
    states[mask_yp, idx] += 1
    mask = mask_yp & (states[:, config.indices["y"]] == 1)
    idx = config.indices["q"]
    if config.boson_ops['q'][0] == 'lambda':
        states[mask, idx] -= 1
        states[mask, idx] %= BT
    elif config.boson_ops['q'][0] == 'X':
        states[mask, idx] *= -1
        states[mask, idx] += 1
    mask = mask_yp & (states[:, config.indices["x"]] == 0)
    idx = config.indices["p"]
    if config.boson_ops['p'][0] == 'lambda':
        states[mask, idx] -= 1
        states[mask, idx] %= BT
    elif config.boson_ops['p'][0] == 'X':
        states[mask, idx] *= -1
        states[mask, idx] += 1

    # Apply the projectors and construct the diagonal operator
    # Start with the diagonal function
    diag_op = np.tile(diag_fn[..., None, None], (1, 1, 1, 2, 2))
    if config.boson_ops['p'][1] == 'id':
        # Diag op is expressed in terms of n_q
        op_dims = ['q']
    elif config.boson_ops['q'][1] == 'id':
        op_dims = ['p']
    elif site % 2 == 0 and term_type == 1:
        op_dims = ['q']
    elif site % 2 == 0 and term_type == 2:
        op_dims = ['p']
    else:
        op_dims = ['p']
    op_dims += ["x", "y", "x'", "y'"]

    states = states[states[:, config.indices["x'"]] == 1]
    # Zx, Zy', P1x'
    diag_op *= np.expand_dims(np.array([0., 1.]), [0, 1, 2, 4])
    diag_op *= np.expand_dims(np.array([1., -1.]), [0, 2, 3, 4])
    diag_op *= np.expand_dims(np.array([1., -1.]), [0, 1, 2, 3])

    for boson, fermion, cval in [('p', 'x', 0), ('q', 'y', 1)]:
        projector = config.boson_ops[boson][1]
        if projector == 'id':
            continue
        mask = states[:, config.indices[fermion]] == cval
        if projector == 'Lambda':
            mask &= states[:, config.indices[boson]] == BT - 1
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

    # QP inherited from Usvd function                      (p incrementer)
    # (1, 0): ["p", "pd", "x", "x'", "y'", "q", "qd", "y"] (id)
    #         ["p", "pd", "x", "y'", "x'", "q", "qd", "y"] (X, lambda)
    # (1, 1): ["x", "p", "pd", "x'", "y'", "q", "qd", "y"] (id)
    #         ["x", "p", "pd", "y'", "x'", "q", "qd", "y"] (X, lambda)
    # (2, 0): ["q", "qd", "y", "y'", "x'", "p", "pd", "x"] (id)
    #         ["q", "qd", "y", "x'", "y'", "p", "pd", "x"] (X, lambda)
    # (2, 1): ["y", "q", "qd", "y'", "x'", "pd", "p", "x"] (id)
    #         ["y", "q", "qd", "x'", "y'", "pd", "p", "x"] (X)
    #         ["y", "q", "qd", "y'", "p", "pd", "x", "x'"] (lambda)

    match (term_type, site % 2, config.boson_ops["p"][1], config.boson_ops["q"][1]):
        case (1, 0, 'id', 'zero'):
            swaps = []
        case (1, 1, 'zero', 'id'):
            swaps = [("q", "y")]
        case (2, 0, 'zero', 'id'):
            swaps = []
        case (2, 1, 'id', 'zero'):
            swaps = [("p", "x")]
            if config.boson_ops["p"][0] == 'lambda':
                swaps.append(("p", "x'"))
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
