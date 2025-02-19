"""Search the optimal order of the diagonal terms."""
from copy import deepcopy
from collections import defaultdict
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import TransformationPass
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.transpiler.passes.optimization import InverseCancellation


class RestoreQubitOrdering(TransformationPass):
    """Restore qubit ordering after routing."""
    def run(self, dag):
        qreg = dag.qregs['q']
        layout = self.property_set['final_layout']
        if all(f == i for i, f in enumerate(layout.to_permutation(qreg))):
            return dag

        swap_dag = DAGCircuit()
        swap_dag.add_qreg(qreg)
        for node in dag.topological_op_nodes():
            if node.op.name == 'swap':
                swap_dag.apply_operation_front(node.op, qargs=node.qargs)

        for node in swap_dag.topological_op_nodes():
            dag.apply_operation_back(node.op, qargs=node.qargs)
            layout.swap(*node.qargs)

        return dag


def get_opt3_pm(backend, physical_qubits):
    pm = generate_preset_pass_manager(3, backend, initial_layout=physical_qubits,
                                      basis_gates=['id', 'x', 'sx', 'rz', 'cx'])
    pm.routing.append(RestoreQubitOrdering())  # pylint: disable=no-member
    pm.routing.append(InverseCancellation([SwapGate()]))  # pylint: disable=no-member
    return pm


def _make_sequences(ops_by_weight, op_coeff, circuit, indent=0):
    print(f'{" " * indent}_make_sequences({op_coeff[0]})')
    op, coeff = op_coeff
    current_weight = op.count('Z')
    ops_by_weight = deepcopy(ops_by_weight)
    ops_by_weight[current_weight].remove(op_coeff)

    control = op.index('Z')
    cxs = []
    while (target := op.find('Z', control + 1)) != -1:
        qc = circuit.num_qubits - control - 1
        qt = circuit.num_qubits - target - 1
        circuit.cx(qc, qt)
        cxs.append((qc, qt))
        control = target
    circuit.rz(2. * coeff, circuit.num_qubits - control - 1)
    for qc, qt in cxs[::-1]:
        circuit.cx(qc, qt)

    circuits = []
    weight = max(1, current_weight - 1)
    while weight > 0 and not ops_by_weight[weight]:
        weight -= 1
    for next_op_coeff in ops_by_weight[weight]:
        circuits.extend(_make_sequences(ops_by_weight, next_op_coeff, circuit.copy(), indent + 1))
    for next_op_coeff in ops_by_weight[current_weight]:
        circuits.extend(_make_sequences(ops_by_weight, next_op_coeff, circuit.copy(), indent + 1))
    weight = current_weight + 1
    while weight <= circuit.num_qubits and not ops_by_weight[weight]:
        weight += 1
    for next_op_coeff in ops_by_weight[weight]:
        circuits.extend(_make_sequences(ops_by_weight, next_op_coeff, circuit.copy(), indent + 1))

    if circuits:
        return circuits

    return [circuit]


def get_all_orderings(spo):
    spo = spo.simplify()
    ops_by_weight = defaultdict(list)
    for pauli, coeff in zip(spo.paulis, spo.coeffs):
        pstr = pauli.to_label()
        try:
            coeff = coeff.real
        except AttributeError:
            pass
        ops_by_weight[pstr.count('Z')].append((pstr, coeff))
    ops_by_weight[0].clear()

    circuits = []

    weight = 1
    while not ops_by_weight[weight]:
        weight += 1
    for op_coeff in ops_by_weight[weight]:
        circuits.extend(_make_sequences(ops_by_weight, op_coeff, QuantumCircuit(spo.num_qubits)))

    return circuits


def diag_propagator_circuit(spo, backend, physical_qubits):
    circuits = get_all_orderings(spo)
    print(f'Transpiling {len(circuits)} circuits')
    pm = get_opt3_pm(backend, physical_qubits)
    tcircuits = pm.run(circuits)
    min_metric = None
    best_circuit = None
    for circuit, tcircuit in zip(circuits, tcircuits):
        metric = (tcircuit.count_ops()['ecr'], tcircuit.depth())
        if min_metric is None or metric < min_metric:
            min_metric = metric
            best_circuit = circuit

    return best_circuit


def _full_parity_walk_sub(num_qubits, angles):
    circ = QuantumCircuit(num_qubits)
    if num_qubits == 1:
        return circ

    circ.compose(_full_parity_walk_sub(num_qubits - 1, angles[::2]),
                 qubits=list(range(1, num_qubits)), inplace=True)

    if num_qubits == 2:
        circ.cx(1, 0)
        circ.rz(2. * angles[3], 0)
    elif num_qubits == 3:
        for i1 in range(2):
            circ.cx(1, 0)
            circ.cx(2, 1)
            circ.rz(2. * angles[5 + 2 * i1], 0)
    elif num_qubits == 4:
        for i1 in range(2):
            circ.cx(1, 0)
            circ.cx(2, 1)
            circ.cx(3, 2)
            circ.rz(2. * angles[9 + 4 * i1], 0)
            circ.cx(1, 0)
            circ.cx(2, 1)
            circ.rz(2. * angles[11 + 4 * i1], 0)

    return circ


def full_parity_walk(num_qubits, angles):
    circ = QuantumCircuit(num_qubits)

    for iq in range(num_qubits):
        circ.rz(2. * angles[2 ** iq], iq)

    for nq in range(2, num_qubits + 1):
        circ.compose(_full_parity_walk_sub(nq, angles), qubits=list(range(nq)), inplace=True)

    for iq in range(num_qubits - 1):
        circ.cx(iq + 1, iq)
    return circ


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


sqrt2 = np.sqrt(2.)
sqrt3 = np.sqrt(3.)

paulii = SparsePauliOp('I')
pauliz = SparsePauliOp('Z')
p0 = SparsePauliOp(['I', 'Z'], [0.5, 0.5])
p1 = SparsePauliOp(['I', 'Z'], [0.5, -0.5])


def hi1r0_diag(interaction_x, time_step):
    # o0-i0-i1-o1
    #          l1
    # -P1_i(0) Z_i(1) ( √2*P0_o(1) I_l(1) + P1_o(1) P0_l(1) )
    spo = -p1.tensor(pauliz).tensor(
        sqrt2 * p0.tensor(paulii) + p1.tensor(p0)
    ).simplify() * interaction_x * time_step
    angles = {pauli.to_label(): 2. * coeff
              for pauli, coeff in zip(spo.paulis, spo.coeffs)}

    circ = QuantumCircuit(4)
    circ.rz(angles['IZII'], 2)
    circ.cx(2, 1)
    circ.rz(angles['IZZI'], 1)
    circ.cx(1, 0)
    circ.rz(angles['IZZZ'], 0)
    circ.cx(1, 0)
    circ.cx(2, 1)
    circ.cx(2, 0)
    circ.rz(angles['IZIZ'], 0)
    circ.cx(2, 0)
    circ.cx(3, 2)
    circ.cx(2, 0)
    circ.rz(angles['ZZIZ'], 0)
    circ.cx(2, 0)
    circ.cx(2, 1)
    circ.cx(1, 0)
    circ.rz(angles['ZZZZ'], 0)
    circ.cx(1, 0)
    circ.rz(angles['ZZZI'], 1)
    circ.cx(2, 1)
    circ.rz(angles['ZZII'], 2)
    circ.cx(3, 2)

    return spo, circ


def hi1r1_diag(interaction_x, time_step):
    # i1-o1-l1-o2-i2
    # P1_i(1) f(o(1), l(1), o(2)) Z_i(2)
    # Will write the SPO and the circuit in this "physical ordering" but reorder the qubits
    # to i-o-l-i-o-l before returning
    spo = p1.tensor(
        (p0 - sqrt2 * p1).tensor(p0).tensor(p0 + p1 / sqrt2)
        - p1.tensor(p1).tensor(sqrt3 / sqrt2 * p0 + p1)
    ).tensor(pauliz).simplify() * interaction_x * time_step
    angles = {pauli.to_label(): 2. * coeff
              for pauli, coeff in zip(spo.paulis, spo.coeffs)}

    circ = QuantumCircuit(5)
    circ.rz(angles['IIIIZ'], 0)

    circ.cx(0, 1)
    circ.rz(angles['IIIZZ'], 1)

    circ.cx(1, 2)
    circ.rz(angles['IIZZZ'], 2)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.rz(angles['IIZIZ'], 2)

    circ.cx(0, 1)
    circ.cx(2, 3)
    circ.rz(angles['IZZIZ'], 3)
    circ.cx(1, 2)
    circ.cx(0, 1)
    circ.cx(2, 3)
    circ.rz(angles['IZIZZ'], 3)
    circ.cx(1, 2)
    circ.cx(2, 3)
    circ.rz(angles['IZZZZ'], 3)
    circ.cx(1, 2)
    circ.cx(2, 3)
    circ.rz(angles['IZIIZ'], 3)

    circ.cx(4, 3)

    circ.rz(angles['ZZIIZ'], 3)
    circ.cx(2, 3)
    circ.cx(1, 2)
    circ.rz(angles['ZZZZZ'], 3)
    circ.cx(2, 3)
    circ.cx(1, 2)
    circ.rz(angles['ZZIZZ'], 3)
    circ.cx(2, 3)
    circ.cx(0, 1)
    circ.cx(1, 2)
    circ.rz(angles['ZZZIZ'], 3)
    circ.cx(2, 3)
    circ.cx(0, 1)

    # circ.cx(4, 3)
    # circ.swap(4, 3)
    circ.cx(3, 4)
    circ.cx(4, 3)
    circ.cx(3, 2)

    circ.rz(angles['ZIZIZ'], 2)
    circ.cx(1, 2)
    circ.cx(0, 1)
    circ.rz(angles['ZIZZZ'], 2)
    circ.cx(1, 2)

    # circ.cx(3, 2)
    # circ.swap(3, 2)
    circ.cx(2, 3)
    circ.cx(3, 2)
    circ.cx(2, 1)

    circ.rz(angles['ZIIZZ'], 1)
    circ.cx(0, 1)

    # circ.cx(2, 1)
    # circ.swap(2, 1)
    circ.cx(1, 2)
    circ.cx(2, 1)
    circ.cx(1, 0)

    circ.rz(angles['ZIIIZ'], 0)

    circ.cx(1, 0)
    circ.swap(2, 1)
    circ.swap(3, 2)
    circ.swap(4, 3)

    paulis = [pauli.to_label() for pauli in spo.paulis]
    paulis = [p[:3] + p[:2:-1] for p in paulis]
    spo = SparsePauliOp(paulis, spo.coeffs)
    tmp = QuantumCircuit(5)
    tmp.compose(circ, qubits=[1, 0, 2, 3, 4], inplace=True)
    circ = tmp

    return spo, circ
