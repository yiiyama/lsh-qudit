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
diag_fn = np.sqrt((np.arange(BT)[None, :, None] + np.arange(1, 3)[:, None, None])
                  / (np.arange(BT)[None, :, None] + np.arange(1, 3)[None, None, :]))
hi1_mat = op_matrix(np.diagflat(diag_fn), hopping_shape, (4, 3, 1))
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

    def occnum(label):
        return states[:, idx[label]]

    flags = []

    if np.all(occnum("p") < 2) and not np.any((occnum("p") == 1) & (occnum("y'") == 0)):
        flags.append('decrement_p_by_X')
    if np.all(occnum("q") < 2) and not np.any((occnum("q") == 1) & (occnum("y'") == 0)):
        flags.append('decrement_q_by_X')
    mask_q = (occnum("y'") == 1) & (occnum("y") == 1) & (occnum("q") == 0)
    mask_q |= (occnum("y'") == 0) & (occnum("y") == 1) & (occnum("q") == BT - 1)
    mask_p = (occnum("y'") == 1) & (occnum("x") == 0) & (occnum("p") == 0)
    mask_p |= (occnum("y'") == 0) & (occnum("x") == 0) & (occnum("p") == BT - 1)
    if np.all(states[np.logical_not(mask_q), idx["x"]] == 1):
        flags.append('no_decrementer_p')
    elif np.all(states[np.logical_not(mask_p), idx["y"]] == 0):
        flags.append('no_decrementer_q')

    states = states[np.logical_not(mask_p | mask_q)]
    filtered_states = {key: states[:, value] for key, value in idx.items()}

    return qubit_labels, flags, filtered_states


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
        qubit_labels, flags = hopping_term_config(term_type, site,
                                                  left_flux=left_flux, right_flux=right_flux)[:2]
    else:
        qubit_labels, flags = config[:2]

    if qp is None:
        labels = ["x", "p", "pd", "x'", "y'", "q", "qd", "y"]
        qp = QubitPlacement([qubit_labels[lab] for lab in labels])

    init_p = qp

    def qpl(label):
        return qp[qubit_labels[label]]

    circuit = QuantumCircuit(init_p.num_qubits)
    if 'no_decrementer_q' in flags:
        pass
    elif 'decrement_q_by_X' in flags:
        circuit.ccx(qpl("y'"), qpl("y"), qpl("q"))
    else:
        circuit.compose(ccix, qubits=[qpl("y'"), qpl("y"), qpl("q")], inplace=True)
        circuit.append(QubitQutritCRxMinusPiGate(), [qpl("q"), qpl("qd")])
        circuit.append(XminusGate(), [qpl("qd")])
        circuit.append(QubitQutritCRxMinusPiGate(), [qpl("q"), qpl("qd")])
        circuit.append(XplusGate(), [qpl("qd")])
        circuit.rz(-0.5 * np.pi, qpl("q"))
        circuit.compose(ccix.inverse(), qubits=[qpl("y'"), qpl("y"), qpl("q")], inplace=True)

    if 'no_decrementer_p' in flags:
        circuit.cx(qpl("y'"), qpl("x'"))
    else:
        # circuit.cx(qpl("y'"), qpl("x'"))
        # circuit.swap(qpl("y'"), qpl("x'"))
        circuit.cx(qpl("x'"), qpl("y'"))
        circuit.cx(qpl("y'"), qpl("x'"))
        qp = qp.swap(qubit_labels["y'"], qubit_labels["x'"])

        circuit.x(qpl("x"))
        if 'decrement_p_by_X' in flags:
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
    if config is None:
        qubit_labels, flags, filtered_states = hopping_term_config(term_type, site,
                                                                   left_flux=left_flux,
                                                                   right_flux=right_flux)
    else:
        qubit_labels, flags, filtered_states = config

    # Pass the filtered states through the decrementers in Usvd
    transformed_states = deepcopy(filtered_states)
    mask_yp = transformed_states["y'"] == 1
    transformed_states["x'"][mask_yp] = 1 - transformed_states["x'"][mask_yp]
    mask = mask_yp & (transformed_states["y"] == 1)
    if 'decrement_q_by_X' in flags:
        transformed_states["q"][mask] = 1 - transformed_states["q"][mask]
    else:
        transformed_states["q"][mask] = (transformed_states["q"][mask] - 1) % BT
    mask = mask_yp & (transformed_states["x"] == 0)
    if 'decrement_p_by_X' in flags:
        transformed_states["p"][mask] = 1 - transformed_states["p"][mask]
    else:
        transformed_states["p"][mask] = (transformed_states["p"][mask] - 1) % BT

    # Which degrees of freedom will have qubits assigned?
    dofs = []
    unique_states = {}
    for label in ["p", "x", "y"]:
        unique_states[label] = np.unique(transformed_states[label])
        if len(unique_states[label]) > 1:
            dofs.append(label)
    if len(np.unique(transformed_states["q"])) > 1 and 'decrement_q_by_X' in flags:
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
    if np.all(transformed_states["p"] < 2) and np.all(transformed_states["q"] < 2):
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
            if np.any(transformed_states["p"] > 1):
                labels.append("pd")
        labels += ["y'", "x'"]
        if "q" in dofs:
            labels.append("q")
            if np.any(transformed_states["q"] > 1):
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
        lr_proj = 'none'
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
        lrp1_proj = 'none'
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
