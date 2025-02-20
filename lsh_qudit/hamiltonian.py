import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qutrit_experiments.gates import (X12Gate, P1Gate, P2Gate, RZ12Gate, QGate, XplusGate,
                                      XminusGate, QubitQutritCRxMinusPiGate,
                                      QubitQutritCRxPlusPiGate)
from .utils import QubitPlacement
from .parity_network import full_parity_walk


sqrt2 = np.sqrt(2.)
sqrt3 = np.sqrt(3.)

paulii = SparsePauliOp('I')
pauliz = SparsePauliOp('Z')
p0 = SparsePauliOp(['I', 'Z'], [0.5, 0.5])
p1 = SparsePauliOp(['I', 'Z'], [0.5, -0.5])

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
                            + [('l', site) for site in sites] + [('a', site) for site in sites])

    circuit = QuantumCircuit(4 * len(sites))
    circuit.x(qp['i'])
    for site in sites:
        circuit.compose(ccix, qubits=[qp['i', site], qp['o', site], qp['a', site]],
                        inplace=True)

    for site in sites:
        circuit.append(X12Gate(), [qp['l', site]])
        circuit.append(QubitQutritCRxPlusPiGate(), [qp['a', site], qp['l', site]])
        circuit.append(X12Gate(), [qp['l', site]])
        circuit.append(QGate(0.25 * time_step), [qp['l', site]])
        circuit.append(X12Gate(), [qp['l', site]])
        circuit.append(QubitQutritCRxMinusPiGate(), [qp['a', site], qp['l', site]])
        circuit.append(X12Gate(), [qp['l', site]])
        circuit.append(QGate(-0.25 * time_step), [qp['l', site]])
        circuit.rz(-0.5 * time_step, qp['a', site])

    for site in sites:
        circuit.compose(ccix.inverse(), qubits=[qp['i', site], qp['o', site], qp['a', site]],
                        inplace=True)

    circuit.x(qp['i'])

    return circuit, qp, qp


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
    spo = -p1.tensor(pauliz).tensor(
        sqrt2 * paulii.tensor(p0) + p0.tensor(p1)
    ).simplify() * interaction_x * time_step
    return spo


def hi1_diag_r0_term(interaction_x, time_step, qp=None):
    op = hi1_diag_r0_op(interaction_x, time_step)
    angles = {pauli.to_label(): 2. * coeff
              for pauli, coeff in zip(op.paulis, op.coeffs)}

    if qp is None:
        qp = QubitPlacement([('o', 1), ('l', 1), ('i', 1), ('i', 0)])
    init_p = qp

    circuit = QuantumCircuit(4)
    circuit.rz(angles['IZII'], qp['i', 1])
    circuit.cx(qp['i', 1], qp['l', 1])
    circuit.rz(angles['IZZI'], qp['l', 1])
    circuit.cx(qp['l', 1], qp['o', 1])
    circuit.rz(angles['IZZZ'], qp['o', 1])
    circuit.cx(qp['l', 1], qp['o', 1])
    circuit.cx(qp['i', 1], qp['l', 1])
    circuit.swap(qp['l', 1], qp['o', 1])
    qp = qp.swap(('l', 1), ('o', 1))
    circuit.cx(qp['i', 1], qp['o', 1])
    circuit.rz(angles['IZIZ'], qp['o', 1])
    circuit.cx(qp['i', 1], qp['o', 1])
    circuit.cx(qp['i', 0], qp['i', 1])
    circuit.cx(qp['i', 1], qp['o', 1])
    circuit.rz(angles['ZZIZ'], qp['o', 1])
    circuit.cx(qp['i', 1], qp['o', 1])
    circuit.swap(qp['o', 1], qp['l', 1])
    qp = qp.swap(('o', 1), ('l', 1))
    circuit.cx(qp['i', 1], qp['l', 1])
    circuit.cx(qp['l', 1], qp['o', 1])
    circuit.rz(angles['ZZZZ'], qp['o', 1])
    circuit.cx(qp['l', 1], qp['o', 1])
    circuit.rz(angles['ZZZI'], qp['l', 1])
    circuit.cx(qp['i', 1], qp['l', 1])
    circuit.rz(angles['ZZII'], qp['i', 1])
    circuit.cx(3, qp['i', 1])

    return circuit, init_p, init_p


def hi1_r1_usvd(qp=None):
    if qp is None:
        qp = QubitPlacement([('l', 2), ('o', 2), ('a', 2), ('i', 2), ('i', 1), ('l', 1), ('o', 1)])
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
    # o1-l1-o2-i2-i1
    spo = (
        (p0 - sqrt2 * p1).tensor(p0).tensor(p0 + p1 / sqrt2)
        - p1.tensor(p1).tensor(sqrt3 / sqrt2 * p0 + p1)
    ).tensor(pauliz).tensor(p1).simplify() * interaction_x * time_step
    return spo


def hi1_diag_r1_term(interaction_x, time_step, qp=None):
    op = hi1_diag_r1_op(interaction_x, time_step)
    angles = {pauli.to_label(): 2. * coeff
              for pauli, coeff in zip(op.paulis, op.coeffs)}

    if qp is None:
        qp = QubitPlacement([('o', 2), ('a', 2), ('i', 1), ('i', 2), ('l', 1), ('o', 1)])
    init_p = qp

    circuit = QuantumCircuit(6)
    circuit.swap(qp['o', 2], qp['a', 2])
    qp = qp.swap(('o', 2), ('a', 2))
    circuit.swap(qp['o', 2], qp['i', 1])
    qp = qp.swap(('o', 2), ('i', 1))
    circuit.swap(qp['o', 2], qp['i', 2])
    qp = qp.swap(('o', 2), ('i', 2))

    circuit.rz(angles['IIIZI'], qp['i', 2])

    circuit.cx(qp['i', 2], qp['o', 2])
    circuit.rz(angles['IIZZI'], qp['o', 2])

    circuit.cx(qp['o', 2], qp['l', 1])
    circuit.rz(angles['IZZZI'], qp['l', 1])

    circuit.cx(qp['l', 1], qp['o', 1])
    circuit.rz(angles['ZZZZI'], qp['o', 1])
    circuit.cx(qp['l', 1], qp['o', 1])

    circuit.cx(qp['i', 2], qp['o', 2])
    circuit.cx(qp['o', 2], qp['l', 1])
    circuit.rz(angles['IZIZI'], qp['l', 1])

    circuit.cx(qp['l', 1], qp['o', 1])
    circuit.rz(angles['ZZIZI'], qp['o', 1])
    circuit.cx(qp['l', 1], qp['o', 1])

    circuit.cx(qp['o', 2], qp['l', 1])
    circuit.cx(qp['i', 2], qp['o', 2])
    circuit.cx(qp['o', 2], qp['l', 1])

    circuit.swap(qp['o', 1], qp['l', 1])
    qp = qp.swap(('o', 1), ('l', 1))

    circuit.cx(qp['o', 2], qp['o', 1])
    circuit.rz(angles['ZIZZI'], qp['o', 1])
    circuit.cx(qp['o', 2], qp['o', 1])

    circuit.cx(qp['i', 2], qp['o', 2])
    circuit.swap(qp['o', 1], qp['o', 2])
    qp = qp.swap(('o', 1), ('o', 2))

    circuit.cx(qp['i', 2], qp['o', 1])
    circuit.rz(angles['ZIIZI'], qp['o', 1])
    circuit.cx(qp['i', 2], qp['o', 1])

    circuit.cx(qp['i', 1], qp['i', 2])  # reflection point

    circuit.cx(qp['i', 2], qp['o', 1])
    circuit.rz(angles['ZIIZZ'], qp['o', 1])
    circuit.cx(qp['i', 2], qp['o', 1])

    circuit.swap(qp['o', 1], qp['o', 2])
    qp = qp.swap(('o', 1), ('o', 2))
    circuit.cx(qp['i', 2], qp['o', 2])

    circuit.cx(qp['o', 2], qp['o', 1])
    circuit.rz(angles['ZIZZZ'], qp['o', 1])
    circuit.cx(qp['o', 2], qp['o', 1])

    circuit.swap(qp['o', 1], qp['l', 1])
    qp = qp.swap(('o', 1), ('l', 1))

    circuit.cx(qp['o', 2], qp['l', 1])
    circuit.cx(qp['i', 2], qp['o', 2])
    circuit.cx(qp['o', 2], qp['l', 1])

    circuit.cx(qp['l', 1], qp['o', 1])
    circuit.rz(angles['ZZIZZ'], qp['o', 1])
    circuit.cx(qp['l', 1], qp['o', 1])

    circuit.rz(angles['IZIZZ'], qp['l', 1])
    circuit.cx(qp['o', 2], qp['l', 1])
    circuit.cx(qp['i', 2], qp['o', 2])

    circuit.cx(qp['l', 1], qp['o', 1])
    circuit.rz(angles['ZZZZZ'], qp['o', 1])
    circuit.cx(qp['l', 1], qp['o', 1])

    circuit.rz(angles['IZZZZ'], qp['l', 1])
    circuit.cx(qp['o', 2], qp['l', 1])

    circuit.rz(angles['IIZZZ'], qp['o', 2])
    circuit.cx(qp['i', 2], qp['o', 2])

    circuit.rz(angles['IIIZZ'], qp['i', 2])

    circuit.cx(qp['i', 1], qp['i', 2])

    circuit.swap(qp['o', 2], qp['i', 2])
    qp = qp.swap(('o', 2), ('i', 2))
    circuit.swap(qp['o', 2], qp['i', 1])
    qp = qp.swap(('o', 2), ('i', 1))
    circuit.swap(qp['o', 2], qp['a', 2])
    qp = qp.swap(('o', 2), ('a', 2))

    return circuit, init_p, qp
