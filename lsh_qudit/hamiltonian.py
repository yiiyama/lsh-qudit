from copy import deepcopy
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from qutrit_experiments.gates import (X12Gate, P1Gate, P2Gate, RZ12Gate, QGate, XplusGate,
                                      XminusGate, QubitQutritCRxMinusPiGate,
                                      QubitQutritCRxPlusPiGate)
from .utils import QubitPlacement, op_matrix, physical_states
from .parity_network import full_parity_walk


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

ccix = QuantumCircuit(3)
ccix.h(2)
ccix.cx(0, 2)
ccix.t(2)
ccix.cx(1, 2)
ccix.tdg(2)
ccix.cx(0, 2)
ccix.t(2)
ccix.cx(1, 2)
ccix.tdg(2)
ccix.h(2)

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


def mass_term(sites, time_step, mass_mu, qp=None):
    if qp is None:
        qp = QubitPlacement([('i', site) for site in sites] + [('o', site) for site in sites])

    circuit = QuantumCircuit(2 * len(sites))
    circuit.rz(-mass_mu * time_step, [qp['i', site] for site in sites if site % 2 == 0])
    circuit.rz(-mass_mu * time_step, [qp['o', site] for site in sites if site % 2 == 0])
    circuit.rz(mass_mu * time_step, [qp['i', site] for site in sites if site % 2 == 1])
    circuit.rz(mass_mu * time_step, [qp['o', site] for site in sites if site % 2 == 1])

    return circuit, qp, qp


def ele12_term(sites, time_step, qp=None):
    if qp is None:
        qp = QubitPlacement([('l', site) for site in sites])

    circuit = QuantumCircuit(len(sites))
    if 1 in sites:
        # r=1: nl=0 or 1
        circuit.p(-0.75 * time_step, qp['l', 1])
    # Bulk
    for site in sites:
        if site == 1:
            continue
        circuit.append(P1Gate(-0.75 * time_step), [qp['l', site]])
        circuit.append(P2Gate(-2. * time_step), [qp['l', site]])

    return circuit, qp, qp


def ele3_f_term(sites, time_step, qp=None):
    # 3/4 * (1 - n_i) * n_o
    if qp is None:
        qp = QubitPlacement([('i', site) for site in sites] + [('o', site) for site in sites])

    circuit = QuantumCircuit(2 * len(sites))
    circuit.x(qp['i'])
    circuit.cp(-0.75 * time_step, qp['i'], qp['o'])
    circuit.x(qp['i'])

    return circuit, qp, qp


def ele3_b_r1_op(time_step):
    # i1-o1-l1
    return p0.tensor(p1).tensor(p1).simplify() * 0.5 * time_step


def ele3_b_r1_term(time_step, qp=None):
    # 1/2 * n_l * n_o * (1 - n_i)
    #  r=1: i1-o1-l1 <- may change later
    op = ele3_b_r1_op(time_step)
    if qp is None:
        qp = QubitPlacement([('l', 1), ('o', 1), ('i', 1)])

    op_angles = {pauli.to_label(): coeff for pauli, coeff in zip(op.paulis, op.coeffs)}
    ordered_paulis = [''.join('IZ'[b] for b in bits)
                      for bits in (np.arange(8)[:, None] >> np.arange(3)[None, ::-1]) % 2]
    angles = [op_angles.get(p, 0.) for p in ordered_paulis]

    circuit = full_parity_walk(3, angles)
    return circuit, qp, qp


def ele3_b_bulk_term(sites, time_step, qp=None):
    # 1/2 * n_l * n_o * (1 - n_i)
    #  Bulk boson
    #   Strategy: rel-phase CCX on an ancilla, then CQ*Rz from ancilla to qutrit.
    #   CQ can be implemented with
    #     CX02 - Q(φ/2) - CX02 - Q(-φ/2)
    #   and CX02 is
    #     X12 - CX01 - X12
    if qp is None:
        qp = QubitPlacement([('i', site) for site in sites] + [('o', site) for site in sites]
                            + [('l', site) for site in sites] + [('d', site) for site in sites])

    circuit = QuantumCircuit(4 * len(sites))
    circuit.x(qp['i'])
    for site in sites:
        circuit.compose(ccix, qubits=[qp['i', site], qp['o', site], qp['l', site]],
                        inplace=True)

    for site in sites:
        circuit.append(X12Gate(), [qp['d', site]])
        circuit.append(QubitQutritCRxPlusPiGate(), [qp['l', site], qp['d', site]])
        circuit.append(X12Gate(), [qp['d', site]])
        circuit.append(QGate(0.25 * time_step), [qp['d', site]])
        circuit.append(X12Gate(), [qp['d', site]])
        circuit.append(QubitQutritCRxMinusPiGate(), [qp['l', site], qp['d', site]])
        circuit.append(X12Gate(), [qp['d', site]])
        circuit.append(QGate(-0.25 * time_step), [qp['d', site]])
        circuit.rz(-0.5 * time_step, qp['l', site])

    for site in sites:
        circuit.compose(ccix.inverse(), qubits=[qp['i', site], qp['o', site], qp['l', site]],
                        inplace=True)

    circuit.x(qp['i'])

    return circuit, qp, qp


def hopping_states(hopping_type, left_flux=None, right_flux=None, as_multi=True):
    states = physical_states(left_flux=left_flux, right_flux=right_flux)
    if hopping_type == 1:
        mask = np.logical_not(np.all(np.isclose(hi1_mat[:, states], 0.), axis=0))
    else:
        mask = np.logical_not(np.all(np.isclose(hi2_mat[:, states], 0.), axis=0))
    return physical_states(left_flux=left_flux, right_flux=right_flux, as_multi=as_multi)[mask]


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
        idx = {"x'": 0, "x": 1, "y'": 3, "y": 4, "p": 2, "q": 5}
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
        idx = {"x'": 4, "x": 3, "y'": 1, "y": 0, "p": 5, "q": 2}

    gl_states = physical_states(left_flux=left_flux, right_flux=right_flux, as_multi=True)
    states = gl_states[gl_states[:, idx["x'"]] != gl_states[:, idx["y'"]]]
    states_preproj = states

    def occnum(label, filt=None):
        if filt is None:
            return states[:, idx[label]]
        return states[filt, idx[label]]

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

    return qubit_labels, boson_ops, default_qp, gl_states, idx


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
    qubit_labels, boson_ops, default_qp = config[:3]

    if qp is None:
        qp = default_qp
    init_p = qp

    circuit = QuantumCircuit(init_p.num_qubits)

    def qpl(label):
        return qp[qubit_labels[label]]

    def swap(label1, label2):
        circuit.swap(qpl(label1), qpl(label2))
        return qp.swap(qubit_labels[label1], qubit_labels[label2])

    if boson_ops['q'][0] == 'id':
        pass
    elif boson_ops['q'][0] == 'X':
        circuit.ccx(qpl("y'"), qpl("y"), qpl("q"))
    else:
        circuit.compose(ccix, qubits=[qpl("y'"), qpl("y"), qpl("q")], inplace=True)
        circuit.append(QubitQutritCRxMinusPiGate(), [qpl("q"), qpl("qd")])
        circuit.append(XminusGate(), [qpl("qd")])
        circuit.append(QubitQutritCRxMinusPiGate(), [qpl("q"), qpl("qd")])
        circuit.append(XplusGate(), [qpl("qd")])
        circuit.rz(-0.5 * np.pi, qpl("q"))
        circuit.compose(ccix.inverse(), qubits=[qpl("y'"), qpl("y"), qpl("q")], inplace=True)

    if site % 2 == 1:
        qp = swap("y'", "q")

    if boson_ops['p'][0] == 'id':
        circuit.cx(qpl("y'"), qpl("x'"))
    else:
        circuit.cx(qpl("y'"), qpl("x'"))
        qp = swap("y'", "x'")

        circuit.x(qpl("x"))
        if boson_ops['p'][0] == 'X':
            circuit.ccx(qpl("y'"), qpl("x"), qpl("p"))
        else:
            circuit.compose(ccix, qubits=[qpl("y'"), qpl("x"), qpl("p")], inplace=True)
            circuit.append(QubitQutritCRxMinusPiGate(), [qpl("p"), qpl("pd")])
            circuit.append(XminusGate(), [qpl("pd")])
            circuit.append(QubitQutritCRxMinusPiGate(), [qpl("p"), qpl("pd")])
            circuit.append(XplusGate(), [qpl("pd")])
            circuit.rz(-0.5 * np.pi, qpl("p"))
            circuit.compose(ccix.inverse(), qubits=[qpl("y'"), qpl("x"), qpl("p")], inplace=True)
        circuit.x(qpl("x"))

    circuit.h(qpl("y'"))

    return circuit, init_p, qp


def hopping_diagonal_op(
    term_type,
    site,
    left_flux=None,
    right_flux=None,
    config=None,
):
    if config is None:
        config = hopping_term_config(term_type, site, left_flux=left_flux, right_flux=right_flux)
    qubit_labels, boson_ops, default_qp, gl_states, indices = config

    # Pass the Gauss's law-satisfying states through the decrementers in Usvd
    states = gl_states.copy()
    idx = indices["y'"]
    mask_yp = states[:, idx] == 1
    idx = indices["x'"]
    states[mask_yp, idx] *= -1
    states[mask_yp, idx] += 1
    mask = mask_yp & (states[:, indices["y"]] == 1)
    idx = indices["q"]
    if boson_ops['q'][0] == 'lambda':
        states[mask, idx] -= 1
        states[mask, idx] %= BT
    elif boson_ops['q'][0] == 'X':
        states[mask, idx] *= -1
        states[mask, idx] += 1
    mask = mask_yp & (states[:, indices["x"]] == 0)
    idx = indices["p"]
    if boson_ops['p'][0] == 'lambda':
        states[mask, idx] -= 1
        states[mask, idx] %= BT
    elif boson_ops['p'][0] == 'X':
        states[mask, idx] *= -1
        states[mask, idx] += 1

    # Apply the projectors and construct the diagonal operator
    # Start with the diagonal function
    diag_op = np.tile(diag_fn[..., None, None], (1, 1, 1, 2, 2))
    if boson_ops['p'][1] == 'id':
        # Diag op is expressed in terms of n_q
        op_dims = ['q']
    elif boson_ops['q'][1] == 'id':
        op_dims = ['p']
    elif site % 2 == 0 and term_type == 1:
        op_dims = ['q']
    elif site % 2 == 0 and term_type == 2:
        op_dims = ['p']
    else:
        op_dims = ['p']
    op_dims += ["x", "y", "x'", "y'"]

    states = states[states[:, indices["x'"]] == 1]
    # Zx, Zy', P1x'
    diag_op *= np.expand_dims(np.array([0., 1.]), [0, 1, 2, 4])
    diag_op *= np.expand_dims(np.array([1., -1.]), [0, 2, 3, 4])
    diag_op *= np.expand_dims(np.array([1., -1.]), [0, 1, 2, 3])

    for boson, fermion, cval in [('p', 'x', 0), ('q', 'y', 1)]:
        projector = boson_ops[boson][1]
        if projector == 'id':
            continue
        mask = states[:, indices[fermion]] == cval
        if projector == 'Lambda':
            mask &= states[:, indices[boson]] == BT - 1
        else:
            mask &= states[:, indices[boson]] != 0
        states = states[~mask]

        fdim = op_dims.index(fermion)
        diag_op = np.moveaxis(diag_op, fdim, 1)
        if projector == 'Lambda':
            diag_op[-1, cval] = 0.
        else:
            diag_op[1:, cval] = 0.
        diag_op = np.moveaxis(diag_op, 1, fdim)

    for fermion in ["x", "y"]:
        unique = np.unique(states[:, indices[fermion]])
        if unique.shape[0] == 1:
            fdim = op_dims.index(fermion)
            diag_op = np.moveaxis(diag_op, fdim, 0)[unique[0]]
            op_dims.pop(fdim)

    # Convert the op to Pauli products
    iz = np.array([[1., 1.], [1., -1.]]) / 2.
    if boson_ops[op_dims[0]][0] == 'lambda':
        # Transform to [I, Z01, Z12] basis
        conv = np.array([[1., 1., 1.], [2., -1., -1.], [-1., -1., 2.]]) / 3.
        diag_op = np.tensordot(conv, diag_op, (1, 0))
    else:
        # Transform to [I, Z] basis
        diag_op = np.tensordot(iz, diag_op[:2], (1, 0))
    # Transform the remaining dims to [I, Z] basis
    for idim in range(1, diag_op.ndim):
        diag_op = np.moveaxis(diag_op, idim, 0)
        diag_op = np.moveaxis(np.tensordot(iz, diag_op, (1, 0)), 0, idim)

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

    if config is None:
        config = hopping_term_config(term_type, site, left_flux=left_flux, right_flux=right_flux)
    qubit_labels, boson_ops, default_qp = config[:3]

    # Construct the circuit
    if qp is None:
        qp = default_qp

    circuit = QuantumCircuit(qp.num_qubits)

    def qpl(label):
        return qp[qubit_labels[label]]

    def swap(label1, label2):
        circuit.swap(qpl(label1), qpl(label2))
        return qp.swap(qubit_labels[label1], qubit_labels[label2])

    match (term_type, site % 2):
        case (1, 0):
            if boson_ops["p"][0] == 'id' and boson_ops["q"][0] == 'X' and "x" not in op_dims:
                # Initial order: ["x", "x'", "y'", "q", "y"]
                # op_dims: ["q", "y", "x'", "y'"]
                coeffs = diag_op.transpose(2, 3, 0, 1)
                shape = coeffs.shape
                coeffs = np.array([interaction_x * time_step * c for c in coeffs.reshape(-1)])
                coeffs = coeffs.reshape(shape)
                circ = diag_4qubit_z1_circuit(coeffs)
                circuit.compose(circ, qubits=[qpl(lab) for lab in ["x'", "y'", "q", "y"]],
                                inplace=True)
            elif boson_ops["p"][1] == 'id' and boson_ops["q"][0] == 'X':
                # Initial order: ["p", "pd", "x", "y'", "x'", "q", "y"]
                # op_dims: ["q", "x", "y", "x'", "y'"]
                coeffs = diag_op.transpose(1, 3, 4, 0, 2)
                shape = coeffs.shape
                coeffs = np.array([interaction_x * time_step * c for c in coeffs.reshape(-1)])
                coeffs = coeffs.reshape(shape)
                circ = diag_5qubit_z2_circuit(coeffs)
                qp = swap("x'", "y'")
                circuit.compose(circ, qubits=[qpl(lab) for lab in ["x", "x'", "y'", "q", "y"]],
                                inplace=True)
                qp = swap("x'", "y'")
            elif boson_ops["p"] != ('lambda', 'Lambda') or boson_ops["q"] != ('lambda', 'Lambda'):
                raise NotImplementedError(f'Optimization for boson_ops {boson_ops} not implemented')
            else:
                raise NotImplementedError('General diagonal term H_I^(1)[r even] not implemented')
        case (1, 1):
            if boson_ops["p"][0] == 'X' and boson_ops["q"][1] == 'id':
                # Initial order: ["x", "p", "y'", "x'", "qd", "q", "y"]
                # op_dims: ["p", "x", "y", "x'", "y'"]
                coeffs = diag_op.transpose(1, 0, 4, 3, 2)
                shape = coeffs.shape
                coeffs = np.array([interaction_x * time_step * c for c in coeffs.reshape(-1)])
                coeffs = coeffs.reshape(shape)
                circ = diag_5qubit_z2_circuit(coeffs)
                qp = swap("q", "y")
                circuit.compose(circ, qubits=[qpl(lab) for lab in ["x", "p", "y'", "x'", "y"]],
                                inplace=True)
                qp = swap("q", "y")
            elif boson_ops["p"] != ('lambda', 'Lambda') or boson_ops["q"] != ('lambda', 'Lambda'):
                raise NotImplementedError(f'Optimization for boson_ops {boson_ops} not implemented')
            else:
                raise NotImplementedError('General diagonal term H_I^(1)[r odd] not implemented')
        case (2, 0):
            labels = ["q", "qd", "y", "y'", "x'", "p", "pd", "x"]
        case (2, 1):
            labels = ["y", "q", "qd", "y'", "p", "pd", "x'", "x"]

    return circuit, qp, qp


def diag_3qubit_z0_circuit(coeffs):
    circuit = QuantumCircuit(3)
    circuit.rz(2. * coeffs[1, 0, 0], 0)
    circuit.cx(0, 1)
    circuit.rz(2. * coeffs[1, 1, 0], 1)
    circuit.cx(1, 2)
    circuit.rz(2. * coeffs[1, 1, 1], 2)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.rz(2. * coeffs[1, 0, 1], 2)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    return circuit


def diag_4qubit_z1_circuit(coeffs):
    circuit = QuantumCircuit(4)
    circuit.compose(diag_3qubit_z0_circuit(coeffs[0]), qubits=[1, 2, 3], inplace=True)
    circuit.cx(0, 1)
    circuit.compose(diag_3qubit_z0_circuit(coeffs[1]), qubits=[1, 2, 3], inplace=True)
    circuit.cx(0, 1)
    return circuit


def diag_5qubit_z2_circuit(coeffs):
    circuit = QuantumCircuit(5)
    circuit.compose(diag_3qubit_z0_circuit(coeffs[0, 0]), qubits=[2, 3, 4], inplace=True)
    circuit.cx(1, 2)
    circuit.compose(diag_3qubit_z0_circuit(coeffs[0, 1]), qubits=[2, 3, 4], inplace=True)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.compose(diag_3qubit_z0_circuit(coeffs[1, 0]), qubits=[2, 3, 4], inplace=True)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.compose(diag_3qubit_z0_circuit(coeffs[1, 1]), qubits=[2, 3, 4], inplace=True)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.cx(0, 1)
    return circuit


def hopping_usvd_old(term_type, left_flux=None, right_flux=None, qp=None):
    """Constructs a U_SVD(1) circuit with optional flux specifications.

    Lattice boundary conditions can impose certain constraints on the available LSH flux from the
    left or the right. Certain simplifications to U_SVD are possible depending on the resulting
    reductions of the matrix elements.

    To check for simplifications, we first identify the physical states (satisfying the Gauss's law
    and the boundary conditions) with non-coinciding n_i(r) and n_i(r+1). Then, within the list of
    such states,

    - If n_l(r) is 0 or 1 with 1 occurring only when n_i(r+1)=1, the decrementer is replaced with X,
      and the o(r)-l(r) projector is replaced with [|0)(0|_l(r)]^{1-n_o(r)}.
    - Similarly, if n_l(r+1) is 0 or 1 with 1 occurring only when n_i(r+1)=1, the decrementer is
      replaced with X, and the o(r+1)-l(r+1) projector is replaced with [|0)(0|_l(r+1)]^{n_o(r+1)}.
    - Excluding states with (n_i, n_o, n_l)_{r+1}=(1, 1, 0) or (0, 0, Lambda), if n_o(r) is
      identically 1, the n_i(r+1)-n_o(r) controlled decrementer and the corresponding projector are
      replaced with identity.
    - Similarly, excluding states with (n_i(r+1), n_o(r), n_l(r))=(1, 0, 0) or (0, 1, Lambda), if
      n_o(r+1) is identically 0, the n_i(r+1)-n_o(r+1) controlled decrementer and the corresponding
      projector are replaced with identity.
    """
    if left_flux is not None or right_flux is not None:
        states = physical_states(left_flux=left_flux, right_flux=right_flux, as_multi=True)
        indices = states[np.nonzero(np.logical_not(np.equal(states[:, 0], states[:, 3])))]

        # TODO HERE - I SHOULD MASK ON n_i(r+1)=1 RIGHT?
        lr_decr = 0
        mask = np.equal(indices[:, 3], 1)
        if not np.all(np.equal(indices[mask, 1], 0)):
            pass


        mask = np.equal(indices[:, 3], 1) & np.equal(indices[:, 1], 0)
        lr_decr = np.any(np.not_equal(indices[mask, 2], 0)).astype(int)
        if lr_decr:
            # A full decrementer or X?
            lr_decr += np.any(np.equal(indices[mask, 2], 2)).astype(int)
        # Is there a need for the l(r+1) decrementer?
        mask = np.equal(indices[:, 3], 1) & np.equal(indices[:, 4], 1)
        lrp1_decr = np.any(np.not_equal(indices[mask, 5], 0)).astype(int)
        if lrp1_decr:
            lrp1_decr += np.any(np.equal(indices[mask, 5], 2)).astype(int)

    else:
        lr_decr = 2
        lrp1_decr = 2

    if qp is None:
        labels = []
        if lrp1_decr > 0:
            labels += [('o', 1), ('l', 1)]
            if lrp1_decr == 2:
                labels.append(('a', 1))
        labels += [('i', 1), ('i', 0)]
        if lr_decr > 0:
            if lr_decr == 2:
                labels.append(('a', 1))
            labels += [('l', 0), ('o', 0)]

        qp = QubitPlacement(labels)
    init_p = qp

    circuit = QuantumCircuit(init_p.num_qubits)
    if lrp1_decr == 2:
        circuit.compose(ccix, qubits=[qp['i', 1], qp['o', 1], qp['a', 1]], inplace=True)
        circuit.append(QubitQutritCRxMinusPiGate(), [qp['a', 1], qp['l', 1]])
        circuit.append(XminusGate(), [qp['l', 1]])
        circuit.append(QubitQutritCRxMinusPiGate(), [qp['a', 1], qp['l', 1]])
        circuit.append(XplusGate(), [qp['l', 1]])
        circuit.rz(-0.5 * np.pi, qp['a', 1])
        circuit.compose(ccix.inverse(), qubits=[qp['i', 1], qp['o', 1], qp['a', 1]], inplace=True)
    elif lrp1_decr == 1:
        circuit.ccx(qp['i', 1], qp['o', 1], qp['l', 1])

    if lr_decr == 0:
        circuit.cx(qp['i', 1], qp['i', 0])
    else:
        # circuit.cx(qp['i', 1], qp['i', 0])
        # circuit.swap(qp['i', 1], qp['i', 0])
        circuit.cx(qp['i', 0], qp['i', 1])
        circuit.cx(qp['i', 1], qp['i', 0])
        qp = qp.swap(('i', 1), ('i', 0))

        if lr_decr == 2:
            circuit.compose(ccix, qubits=[qp['i', 1], qp['o', 0], qp['a', 0]], inplace=True)
            circuit.append(QubitQutritCRxMinusPiGate(), [qp['a', 0], qp['l', 0]])
            circuit.append(XminusGate(), [qp['l', 0]])
            circuit.append(QubitQutritCRxMinusPiGate(), [qp['a', 0], qp['l', 0]])
            circuit.append(XplusGate(), [qp['l', 0]])
            circuit.rz(-0.5 * np.pi, qp['a', 0])
            circuit.compose(ccix.inverse(), qubits=[qp['i', 1], qp['o', 0], qp['a', 0]],
                            inplace=True)
        else:
            circuit.ccx(qp['i', 1], qp['o', 0], qp['l', 0])

    circuit.h(qp['i', 1])

    return circuit, init_p, qp


def hi1_r0_usvd(qp=None):
    if qp is None:
        qp = QubitPlacement([('o', 1), ('l', 1), ('i', 1), ('i', 0)])
    init_p = qp

    # CX(i1, i0) +
    circuit = QuantumCircuit(4)
    # # Rel-phase Toffoli (CZiY with o1-i1-l1 as c1-c2-t)
    # circuit.h(qp['l', 1])
    # circuit.t(qp['l', 1])
    # circuit.cx(qp['i', 1], qp['l', 1])
    # circuit.tdg(qp['l', 1])
    # circuit.cx(qp['o', 1], qp['l', 1])
    circuit.cx(qp['i', 1], qp['i', 0])
    # circuit.t(qp['l', 1])
    # circuit.cx(qp['i', 1], qp['l', 1])
    # circuit.tdg(qp['l', 1])
    # circuit.swap(qp['o', 1], qp['l', 1])
    # qp = qp.swap(('o', 1), ('l', 1))
    # circuit.cp(np.pi / 2., qp['i', 1], qp['o', 1])
    # circuit.h(qp['l', 1])
    circuit.ccx(qp['o', 1], qp['i', 1], qp['l', 1])
    circuit.h(qp['i', 1])

    return circuit, init_p, qp


def hi1_diag_r0_op(interaction_x, time_step):
    # i0-i1-l1-o1
    # -P1_i(0) Z_i(1) ( I_l(1) √2*P0_o(1) + P0_l(1) P1_o(1) )
    # diag_fn = diagonal_function_entries(1, left_flux=0)
    # op = np.ones((2, 2, 3, 2, 2, 3))
    # op *= np.expand_dims(np.array([0., 1.]), (1, 2, 3, 4, 5))  # P1_{i(0)}
    # op *= np.expand_dims(np.array([0., 1.]), (0, 2, 3, 4, 5))  # P1_{o(0)}
    # op *= np.expand_dims(np.array([1., 0., 0.]), (0, 1, 3, 4, 5))  # P0_{l(0)}
    # op *= np.expand_dims(np.array([1., -1.]), (0, 1, 2, 4, 5))  # Z_{i(1)}
    # op *= np.expand_dims(np.array([1., -1.]), (0, 1, 2, 3, 5))  # Z_{o(1)}

    spo = -p1.tensor(pauliz).tensor(
        sqrt2 * paulii.tensor(p0) + p0.tensor(p1)
    ).simplify() * interaction_x * time_step
    return spo


def hi1_diag_r0_term(interaction_x, time_step, qp=None):
    op = hi1_diag_r0_op(interaction_x, time_step)
    angles = {pauli.to_label(): 2. * coeff
              for pauli, coeff in zip(op.paulis, op.coeffs)}

    params = [Parameter(f'p{i}') for i in range(4)]
    subpaulis = ['ZII', 'ZIZ', 'ZZI', 'ZZZ']
    subcycle = QuantumCircuit(3)
    subcycle.rz(params[0], 2)
    subcycle.cx(2, 1)
    subcycle.rz(params[2], 1)
    subcycle.cx(1, 0)
    subcycle.rz(params[3], 0)
    subcycle.cx(2, 1)
    subcycle.cx(1, 0)
    subcycle.rz(params[1], 0)
    subcycle.cx(2, 1)
    subcycle.cx(1, 0)
    subcycle.cx(2, 1)
    subcycle.cx(1, 0)

    if qp is None:
        qp = QubitPlacement([('o', 1), ('l', 1), ('i', 1), ('i', 0)])
    init_p = qp

    circuit = QuantumCircuit(4)
    subqubits = [qp['o', 1], qp['l', 1], qp['i', 1]]

    circuit.compose(
        subcycle.assign_parameters(dict(zip(params, [angles['I' + p] for p in subpaulis]))),
        qubits=subqubits,
        inplace=True
    )
    circuit.cx(qp['i', 0], qp['i', 1])
    circuit.compose(
        subcycle.assign_parameters(dict(zip(params, [angles['Z' + p] for p in subpaulis]))),
        qubits=subqubits,
        inplace=True
    )
    circuit.cx(qp['i', 0], qp['i', 1])

    return circuit, init_p, init_p


def hi1_r1_usvd(qp=None):
    if qp is None:
        qp = QubitPlacement([('o', 2), ('l', 2), ('a', 2), ('i', 2), ('i', 1), ('l', 1), ('o', 1)])
    init_p = qp

    circuit = QuantumCircuit(7)
    circuit.compose(ccix, qubits=[qp['i', 2], qp['o', 2], qp['a', 2]], inplace=True)
    circuit.append(QubitQutritCRxMinusPiGate(), [qp['a', 2], qp['l', 2]])
    circuit.append(XminusGate(), [qp['l', 2]])
    circuit.append(QubitQutritCRxMinusPiGate(), [qp['a', 2], qp['l', 2]])
    circuit.append(XplusGate(), [qp['l', 2]])
    circuit.rz(-0.5 * np.pi, qp['a', 2])
    circuit.compose(ccix.inverse(), qubits=[qp['i', 2], qp['o', 2], qp['a', 2]], inplace=True)
    # circuit.cx(qp['i', 2], qp['i', 1])
    # circuit.swap(qp['i', 2], qp['i', 1])
    circuit.cx(qp['i', 1], qp['i', 2])
    circuit.cx(qp['i', 2], qp['i', 1])
    qp = qp.swap(('i', 2), ('i', 1))
    circuit.x(qp['o', 1])
    circuit.ccx(qp['i', 2], qp['o', 1], qp['l', 1])
    circuit.x(qp['o', 1])
    circuit.h(qp['i', 2])

    return circuit, init_p, qp


def hi1_diag_r1_op(interaction_x, time_step):
    # o1-l1-i2-i1-o2
    spo = (p0 - sqrt2 * p1).tensor(p0).tensor(pauliz).tensor(p1).tensor(p0 + p1 / sqrt2)
    spo -= p1.tensor(p1).tensor(pauliz).tensor(p1).tensor(sqrt3 / sqrt2 * p0 + p1)
    spo = spo.simplify() * interaction_x * time_step
    return spo


def hi1_diag_r1_term(interaction_x, time_step, qp=None):
    op = hi1_diag_r1_op(interaction_x, time_step)
    angles = {pauli.to_label(): 2. * coeff
              for pauli, coeff in zip(op.paulis, op.coeffs)}

    if qp is None:
        qp = QubitPlacement([('o', 2), ('a', 2), ('i', 1), ('i', 2), ('l', 1), ('o', 1)])
    init_p = qp

    params = [Parameter(f'p{i}') for i in range(4)]
    subpaulis = ['ZII', 'ZIZ', 'ZZI', 'ZZZ']
    subcycle = QuantumCircuit(3)
    subcycle.rz(params[0], 2)
    subcycle.cx(2, 1)
    subcycle.rz(params[2], 1)
    subcycle.cx(1, 0)
    subcycle.rz(params[3], 0)
    subcycle.cx(2, 1)
    subcycle.cx(1, 0)
    subcycle.rz(params[1], 0)
    subcycle.cx(2, 1)
    subcycle.cx(1, 0)
    subcycle.cx(2, 1)
    subcycle.cx(1, 0)

    circuit = QuantumCircuit(6)
    circuit.swap(qp['o', 2], qp['a', 2])
    qp = qp.swap(('o', 2), ('a', 2))

    subqubits = [qp['o', 2], qp['i', 1], qp['i', 2]]

    circuit.compose(
        subcycle.assign_parameters(dict(zip(params, [angles['II' + p] for p in subpaulis]))),
        qubits=subqubits,
        inplace=True
    )
    circuit.cx(qp['l', 1], qp['i', 2])
    circuit.compose(
        subcycle.assign_parameters(dict(zip(params, [angles['IZ' + p] for p in subpaulis]))),
        qubits=subqubits,
        inplace=True
    )
    circuit.cx(qp['o', 1], qp['l', 1])
    circuit.cx(qp['l', 1], qp['i', 2])
    circuit.compose(
        subcycle.assign_parameters(dict(zip(params, [angles['ZI' + p] for p in subpaulis]))),
        qubits=subqubits,
        inplace=True
    )
    circuit.cx(qp['o', 1], qp['l', 1])
    circuit.cx(qp['l', 1], qp['i', 2])
    circuit.compose(
        subcycle.assign_parameters(dict(zip(params, [angles['ZZ' + p] for p in subpaulis]))),
        qubits=subqubits,
        inplace=True
    )
    circuit.cx(qp['o', 1], qp['l', 1])
    circuit.cx(qp['l', 1], qp['i', 2])
    circuit.cx(qp['o', 1], qp['l', 1])
    circuit.swap(qp['o', 2], qp['a', 2])
    qp = qp.swap(('o', 2), ('a', 2))

    return circuit, init_p, qp


def hi1_r2_usvd(qp=None):
    if qp is None:
        qp = QubitPlacement([('o', 3), ('l', 3), ('i', 3), ('i', 2), ('l', 2), ('a', 2), ('o', 2)])
    init_p = qp

    circuit = QuantumCircuit(7)
    circuit.ccx(qp['i', 3], qp['o', 3], qp['l', 3])
    # circuit.cx(qp['i', 3], qp['i', 2])
    # circuit.swap(qp['i', 3], qp['i', 2])
    circuit.cx(qp['i', 2], qp['i', 3])
    circuit.cx(qp['i', 3], qp['i', 2])
    qp = qp.swap(('i', 3), ('i', 2))
    circuit.x(qp['o', 2])
    circuit.compose(ccix, qubits=[qp['i', 3], qp['o', 2], qp['a', 2]], inplace=True)
    circuit.append(QubitQutritCRxMinusPiGate(), [qp['a', 2], qp['l', 2]])
    circuit.append(XminusGate(), [qp['l', 2]])
    circuit.append(QubitQutritCRxMinusPiGate(), [qp['a', 2], qp['l', 2]])
    circuit.append(XplusGate(), [qp['l', 2]])
    circuit.rz(-0.5 * np.pi, qp['a', 2])
    circuit.compose(ccix.inverse(), qubits=[qp['i', 3], qp['o', 2], qp['a', 2]], inplace=True)
    circuit.x(qp['o', 2])
    circuit.h(qp['i', 3])

    return circuit, init_p, qp


def hi1_diag_r2_op(interaction_x, time_step):
    # o2-l3-o3-i3-i2
    spo = (p0 - sqrt2 * p1).tensor(
        p0.tensor(p0 + p1 / sqrt2) + p1.tensor(sqrt3 / sqrt2 * p0 + p1)
    ).tensor(pauliz).tensor(p1).simplify() * interaction_x * time_step
    return spo


def hi1_diag_r2_term(interaction_x, time_step, qp=None):
    op = hi1_diag_r2_op(interaction_x, time_step)
    angles = {pauli.to_label(): 2. * coeff
              for pauli, coeff in zip(op.paulis, op.coeffs)}

    if qp is None:
        qp = QubitPlacement([('o', 2), ('a', 2), ('i', 1), ('i', 2), ('l', 1), ('o', 1)])
    init_p = qp

    circuit = QuantumCircuit(6)
