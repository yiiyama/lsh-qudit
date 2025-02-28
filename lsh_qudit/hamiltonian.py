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
    decr_ops = {'p': 'lambda', 'q': 'lambda'}
    proj_ops = {'p': 'Lambda', 'q': 'Lambda'}
    bconfs = [('p', 'x', 0), ('q', 'y', 1)]
    # Check for simplifications following the projection on the other side
    results = []
    for iproj in [0, 1]:
        states = states_preproj.copy()
        proj = bconfs[iproj]
        test = bconfs[1 - iproj]

        cyp = occnum("y'") == 1
        cf = occnum(proj[1]) == proj[2]
        if np.all(occnum(proj[0], cf) < 2) and np.all(occnum(proj[0], ~cyp & cf) == 0):
            # Special case of l=0,1
            proj_side = ('X', 'zero')
        else:
            proj_side = ('lambda', 'Lambda')

        mask = ~((cyp & cf & (occnum(proj[0]) == 0)) | (~cyp & cf & (occnum(proj[0]) == BT - 1)))
        if np.all(mask):
            proj_side = (proj_side[0], 'id')
        else:
            states = states[mask]

        cyp = occnum("y'") == 1
        cf = occnum(test[1]) == test[2]
        if not np.any(cf):
            test_side = ('id', 'id')
            results.append((proj_side, test_side))
            continue

        if np.all(occnum(test[0], cf) < 2) and np.all(occnum(test[0], ~cyp & cf) == 0):
            # Special case of l=0,1
            test_side = ('X', 'zero')
        else:
            test_side = ('lambda', 'Lambda')

        mask = ~((cyp & cf & (occnum(test[0]) == 0)) | (~cyp & cf & (occnum(test[0]) == BT - 1)))
        if np.all(mask):
            test_side = (test_side[0], 'id')

        results.append((proj_side, test_side))

    # Evaluate and select the better site to project on according to the following score matrix:
    #                test
    #  |    |id-id X-id λ-id X-0 λ-Λ
    # p|X-id|    0    2    3   6   9
    # r|λ-id|    1    3    4   7  10
    # o|X-0 |    5    6    7  11  12
    # j|λ-Λ |    8    9   10  12  13
    scores = []
    for result in results:
        match result:
            case (('X', 'id'), ('id', 'id')):
                score = 0
            case (('lambda', 'id'), ('id', 'id')):
                score = 1
            case (('X', 'id'), ('X', 'id')):
                score = 2
            case (('lambda', 'id'), ('X', 'id')) | (('X', 'id'), ('lambda', 'id')):
                score = 3
            case (('lambda', 'id'), ('lambda', 'id')):
                score = 4
            case (('X', 'zero'), ('id', 'id')):
                score = 5
            case (('X', 'zero'), ('X', 'id')) | (('X', 'id'), ('X', 'zero')):
                score = 6
            case (('X', 'zero'), ('lambda', 'id')) | (('lambda', 'id'), ('X', 'zero')):
                score = 7
            case (('lambda', 'Lambda'), ('id', 'id')):
                score = 8
            case (('lambda', 'Lambda'), ('X', 'id')) | (('X', 'id'), ('lambda', 'Lambda')):
                score = 9
            case (('lambda', 'Lambda'), ('lambda', 'id')) | (('lambda', 'id'), ('lambda', 'Lambda')):
                score = 10
            case (('X', 'zero'), ('X', 'zero')):
                score = 11
            case (('lambda', 'Lambda'), ('X', 'zero')) | (('X', 'zero'), ('lambda', 'Lambda')):
                score = 12
            case (('lambda', 'Lambda'), ('lambda', 'Lambda')):
                score = 13
            case _:
                raise ValueError('typo?')
        scores.append(score)

    if scores[0] <= scores[1]:
        decr_ops['p'], proj_ops['p'] = results[0][0]
        decr_ops['q'], proj_ops['q'] = results[0][1]
    else:
        decr_ops['p'], proj_ops['p'] = results[1][1]
        decr_ops['q'], proj_ops['q'] = results[1][0]

    return qubit_labels, decr_ops, proj_ops, gl_states, idx


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
    circuit.compose(diag_circuit, qubits=[init_p[lab] for lab in final_p.qubit_labels],
                    inplace=True)
    circuit.compose(usvd_circuit.inverse(), qubits=[init_p[lab] for lab in final_p.qubit_labels],
                    inplace=True)

    return circuit, qp, qp


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
    qubit_labels, decr_ops = config[:2]

    if qp is None:
        labels = ["x", "p", "pd", "x'", "y'", "q", "qd", "y"]
        qp = QubitPlacement([qubit_labels[lab] for lab in labels])

    init_p = qp

    def qpl(label):
        return qp[qubit_labels[label]]

    circuit = QuantumCircuit(init_p.num_qubits)
    if decr_ops['q'] == 'id':
        pass
    elif decr_ops['q'] == 'X':
        circuit.ccx(qpl("y'"), qpl("y"), qpl("q"))
    else:
        circuit.compose(ccix, qubits=[qpl("y'"), qpl("y"), qpl("q")], inplace=True)
        circuit.append(QubitQutritCRxMinusPiGate(), [qpl("q"), qpl("qd")])
        circuit.append(XminusGate(), [qpl("qd")])
        circuit.append(QubitQutritCRxMinusPiGate(), [qpl("q"), qpl("qd")])
        circuit.append(XplusGate(), [qpl("qd")])
        circuit.rz(-0.5 * np.pi, qpl("q"))
        circuit.compose(ccix.inverse(), qubits=[qpl("y'"), qpl("y"), qpl("q")], inplace=True)

    if decr_ops['p'] == 'id':
        circuit.cx(qpl("y'"), qpl("x'"))
    else:
        # circuit.cx(qpl("y'"), qpl("x'"))
        # circuit.swap(qpl("y'"), qpl("x'"))
        circuit.cx(qpl("x'"), qpl("y'"))
        circuit.cx(qpl("y'"), qpl("x'"))
        qp = qp.swap(qubit_labels["y'"], qubit_labels["x'"])

        circuit.x(qpl("x"))
        if decr_ops['p'] == 'X':
            circuit.ccx(qpl("y'"), qpl("x"), qpl("p"))
        else:
            circuit.compose(ccix, qubits=[qpl("y'"), qpl("x"), qpl("p")], inplace=True)
            circuit.append(QubitQutritCRxMinusPiGate(), [qpl("p"), qpl("pd")])
            circuit.append(XminusGate(), [qpl("pd")])
            circuit.append(QubitQutritCRxMinusPiGate(), [qpl("p"), qpl("pd")])
            circuit.append(XplusGate(), [qpl("pd")])
            circuit.rz(-0.5 * np.pi, qpl("p"))
            circuit.compose(ccix.inverse(), qubits=[qpl("y'"), qpl("y"), qpl("p")], inplace=True)
        circuit.x(qpl("x"))

    circuit.h(qpl("y'"))

    return circuit, init_p, qp


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
    # TODO Logic surrounding decr_ops / proj_ops - HI1 r=1 does not agree with the specialized function

    if config is None:
        config = hopping_term_config(term_type, site, left_flux=left_flux, right_flux=right_flux)
    qubit_labels, decr_ops, proj_ops, gl_states, indices = config

    # Construct the full diagonal op as a tensor
    # Start with the diagonal function
    diag_op = np.tile(diag_fn[..., None, None, None], (1, 1, 1, BT, 2, 2))
    diag_op = np.moveaxis(diag_op, (0, 1, 2, 3), (indices[x] for x in ["p", "x", "y", "q"]))
    # p projector
    diag_op = np.moveaxis(diag_op, (indices["p"], indices["x"]), (0, 1))
    if proj_ops['p'] == 'id':
        pass
    elif proj_ops['p'] == 'zero':
        diag_op[1:, 0] = 0.
    else:
        diag_op[-1, 0] = 0.
    diag_op = np.moveaxis(diag_op, (0, 1), (indices["p"], indices["x"]))
    diag_op = np.moveaxis(diag_op, (indices["q"], indices["y"]), (0, 1))
    # q projector
    if proj_ops['q'] == 'id':
        pass
    elif proj_ops['q'] == 'zero':
        diag_op[1:, 1] = 0.
    else:
        diag_op[-1, 1] = 0.
    diag_op = np.moveaxis(diag_op, (0, 1), (indices["q"], indices["y"]))
    # Zx
    diag_op = np.moveaxis(diag_op, indices["x"], 0)
    diag_op[1] *= -1.
    diag_op = np.moveaxis(diag_op, 0, indices["x"])
    # |1)(1|x'
    diag_op = np.moveaxis(diag_op, indices["x'"], 0)
    diag_op[0] = 0.
    diag_op = np.moveaxis(diag_op, 0, indices["x'"])
    # Zy'
    diag_op = np.moveaxis(diag_op, indices["y'"], 0)
    diag_op[1] *= -1.
    diag_op = np.moveaxis(diag_op, 0, indices["y'"])

    # Pass the Gauss's law-satisfying states through the decrementers in Usvd
    states = gl_states.copy()
    idx = indices["y'"]
    mask_yp = states[:, idx] == 1
    idx = indices["x'"]
    states[mask_yp, idx] *= -1
    states[mask_yp, idx] += 1
    if decr_ops['q'] != 'id':
        mask = mask_yp & (states[:, indices["y"]] == 1)
        idx = indices["q"]
        if decr_ops['q'] == 'lambda':
            states[mask, idx] -= 1
            states[mask, idx] %= BT
        elif decr_ops['q'] == 'X':
            states[mask, idx] *= -1
            states[mask, idx] += 1
    if decr_ops['p'] != 'id':
        mask = mask_yp & (states[:, indices["x"]] == 0)
        idx = indices["p"]
        if decr_ops['p'] == 'lambda':
            states[mask, idx] -= 1
            states[mask, idx] %= BT
        elif decr_ops['p'] == 'X':
            states[mask, idx] *= -1
            states[mask, idx] += 1

    usvd_states = states.copy()
    usvd_states[:, indices["y'"]] *= -1
    usvd_states[:, indices["y'"]] += 1
    usvd_states = np.unique(np.concatenate([states, usvd_states], axis=0), axis=0)

    # Identify the degrees of freedom and simplify the diagonal op as necessary
    qubits = {"y'"}
    op_dims = dict(indices)

    # Select states with n_x'=1
    gather = np.nonzero(states[:, indices["x'"]] == 1)[0]
    if gather.shape[0] == 0:
        # Return identity circuit
        pass
    unique = np.unique(states[:, indices["x'"]])
    if unique.shape[0] > 1:
        states = states[gather]
        qubits.add("x'")
    else:
        # If unique, the value must be 1
        diag_op = np.moveaxis(diag_op, op_dims["x'"], 0)[1]
        for key in list(op_dims.keys()):
            if op_dims[key] > op_dims["x'"]:
                op_dims[key] -= 1
        op_dims.pop("x'")

    boson_dofs = [('x', 1, 'p')]
    if np.all(states[:, indices["p"]] == states[:, indices["q"]]):
        boson_dofs.append(('y', 0, 'p'))
        # Take the p-q diagonals and eliminate the q axis
        diag_op = np.moveaxis(diag_op, (op_dims["p"], op_dims["q"]), (0, 1))
        diag_op = diag_op[np.arange(BT), np.arange(BT)]
        diag_op = np.moveaxis(diag_op, 0, op_dims["p"])
        for key in list(op_dims.keys()):
            if op_dims[key] > op_dims["q"]:
                op_dims[key] -= 1
        op_dims.pop("q")
    else:
        boson_dofs.append(('y', 0, 'q'))

    for ctl, val, dof in boson_dofs:
        if decr_ops[dof] == 'lambda':
            gather = np.nonzero((states[:, indices[ctl]] == val)
                                | (states[:, indices[dof]] != BT - 1))[0]
        elif decr_ops[dof] == 'X':
            gather = np.nonzero((states[:, indices[ctl]] == val)
                                | (states[:, indices[dof]] == 0))[0]
        if gather.shape[0] == 0:
            # Return identity
            pass
        unique = np.unique(states[:, indices[dof]])
        if unique.shape[0] > 1:
            states = states[gather]
            qubits.add(dof)
        elif dof in op_dims:
            diag_op = np.moveaxis(diag_op, op_dims[dof], 0)[unique[0]]
            for key in list(op_dims.keys()):
                if op_dims[key] > op_dims[dof]:
                    op_dims[key] -= 1
            op_dims.pop(dof)

    for dof in ['x', 'y']:
        unique = np.unique(states[:, indices[dof]])
        if unique.shape[0] > 1:
            qubits.add(dof)
        else:
            diag_op = np.moveaxis(diag_op, op_dims[dof], 0)[unique[0]]
            for key in list(op_dims.keys()):
                if op_dims[key] > op_dims[dof]:
                    op_dims[key] -= 1
            op_dims.pop(dof)

    # Which degrees of freedom will have qubits assigned?
    dofs = []
    unique_states = {}
    for label in ["p", "x", "y"]:
        unique_states[label] = np.unique(input_states[label])
        if len(unique_states[label]) > 1:
            dofs.append(label)
    if len(np.unique(input_states["q"])) > 1 and 'decrement_q_by_X' in flags:
        dofs.append("q")

    # Construct the full diagonal op as a tensor
    # Dims: y, q, x', y', p, x
    # Start with the diagonal function
    diag_op = diag_fn.transpose(2, 1, 0)
    diag_op = np.tile(diag_op[:, None, None, None, :, :], (1, BT, 2, 2, 1, 1))
    # p projector
    if 'decrement_p_by_X' in flags:
        diag_op[..., 1:, 0] = 0.
    else:
        diag_op[..., -1, 0] = 0.
    # q projector
    if 'decrement_q_by_X' in flags:
        diag_op[1, 1:] = 0.
    else:
        diag_op[1, -1] = 0.
    # Zx
    diag_op[..., 1] *= -1.
    # |1)(1|x'
    diag_op[:, :, 0] = 0.
    # Zy'
    diag_op[:, :, :, 1] *= -1.

    # Project out non-dof dimensions
    indices = (
        unique_states["y"][0] if len(unique_states["y"]) == 1 else slice(None),
        unique_states["p"][0] if len(unique_states["p"]) == 1 else slice(None),
        unique_states["x"][0] if len(unique_states["x"]) == 1 else slice(None)
    )
    if "q" in dofs:
        op = np.repeat(op[None, ...], BT, axis=0)
        # q is in dofs only when decrementing q by X
        op[1:, 1] = 0.
        indices = (slice(None),) + indices
    op = op[indices]

    # Convert the op to Pauli products
    if np.all(input_states["p"] < 2) and np.all(input_states["q"] < 2):
        nq = len(op.shape)
        indices = (slice(0, 2),) * nq
        op = op[indices]

        selections = (np.arange(2 ** nq)[:, None] >> np.arange(nq)[None, ::-1]) % 2
        paulis = np.array([[1, 1], [1, -1]])[selections]
        args = sum(([paulis[:, i], (0, i + 1)] for i in range(nq)), [])
        args.append(list(range(nq + 1)))
        pfilter = np.einsum(*args).reshape(2 ** nq, 2 ** nq)
        angles = {''.join('IZ'[b] for b in sel): np.tensordot(fil, op.reshape(-1), (1, 0))
                  for sel, fil in zip(selections, pfilter)}
    else:
        raise NotImplementedError('Qudit diagonals')

    # Construct the circuit
    if qp is None:
        labels = []
        if "x" in dofs:
            labels.append("x")
        if "p" in dofs:
            labels.append("p")
            if np.any(input_states["p"] > 1):
                labels.append("pd")
        labels += ["y'", "x'"]
        if "q" in dofs:
            labels.append("q")
            if np.any(input_states["q"] > 1):
                labels.append("qd")
        if "y" in dofs:
            labels.append("y")

        qp = QubitPlacement([qubit_labels[lab] for lab in labels])

    circuit = QuantumCircuit(qp.num_qubits)

    return 0, 1, 2


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


def hi1_diag_term(interaction_x, time_step, left_flux=None, right_flux=None, qp=None):
    # State indices that should pass the full H_I projection
    outcome_indices = hopping_states(1, left_flux=left_flux, right_flux=right_flux, as_multi=True)

    # Is there a need for the l(r) projector?
    mask = np.equal(outcome_indices[:, 3], 1) & np.equal(outcome_indices[:, 1], 0)
    if np.all(np.equal(outcome_indices[mask, 2], 0)):
        # If all doubly-controlled states had l(r)=0, we skip the decrementer & incrementer in the
        # SVD circuits altogether and therefore not apply any projection in the diagonal either.
        lr_proj = 'id'
    elif np.all(np.equal(outcome_indices[mask, 2], 1)):
        # If all doubly-controlled states had l(r)=1, we use an X instead of lambda-. We then have
        # to project on l(r)=0.
        lr_proj = 'zero'
    else:
        lr_proj = 'full'

    # Is there a need for the l(r+1) projector?
    mask = np.equal(outcome_indices[:, 3], 1) & np.equal(outcome_indices[:, 4], 1)
    if np.all(np.equal(outcome_indices[mask, 5], 0)):
        # If all doubly-controlled states had l(r+1)=0, we skip the decrementer & incrementer in the
        # SVD circuits altogether and therefore not apply any projection in the diagonal either.
        lrp1_proj = 'id'
    elif np.all(np.equal(outcome_indices[mask, 5], 1)):
        # If all doubly-controlled states had l(r+1)=1, we use an X instead of lambda-. We then have
        # to project on l(r)=0.
        lrp1_proj = 'zero'
    else:
        lrp1_proj = 'full'

    # Next, actually apply the U_SVD transformation to the passing states to determine the
    # possible simplifications on Z_{o(r)} and the diagonal function.
    shape = (2, 2, 3, 2, 2, 3)
    cx = (p0.tensor(paulii) + p1.tensor(paulix)).simplify()
    cc_cyc_decr = np.eye(12, dtype=np.complex128)
    cc_cyc_decr[9:, 9:] = cyc_incr.conjugate().T
    coc_cyc_decr = np.eye(12, dtype=np.complex128)
    coc_cyc_decr[6:9, 6:9] = cyc_incr.conjugate().T
    usvd = (op_matrix(hadamard, shape, 2) @ op_matrix(cx, shape, (2, 5))
            @ op_matrix(coc_cyc_decr, shape, (2, 4, 3)) @ op_matrix(cc_cyc_decr, shape, (2, 1, 0)))

    states = hopping_states(1, left_flux=left_flux, right_flux=right_flux, as_multi=False)
    one_hots = np.zeros((np.prod(shape),) + states.shape)
    one_hots[states, np.arange(len(states))] = 1.
    tr_states = np.array(np.nonzero((usvd @ one_hots).reshape(shape + (-1,)))[:-1]).T

    if np.all(np.equal(tr_states[:, 1], 0)):
        z_or = 1
    elif np.all(np.equal(tr_states[:, 1], 1)):
        z_or = -1
    else:
        z_or = None

    diag_fn = diagonal_function_entries(1, left_flux=left_flux, right_flux=right_flux)

    # Compose the full diagonal term
    # diag_op =


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
    # o2-o2-i2-i1
    spo = (
        (p0 - sqrt2 * p1).tensor(p0).tensor(p0 + p1 / sqrt2)
        - p1.tensor(p1).tensor(sqrt3 / sqrt2 * p0 + p1)
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
