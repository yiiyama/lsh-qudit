"""Search the optimal order of the diagonal terms."""
from copy import deepcopy
from collections import defaultdict
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate
from qiskit.transpiler import TransformationPass, generate_preset_pass_manager


class RestoreQubitOrdering(TransformationPass):
    """Restore qubit ordering after routing."""
    def run(self, dag):
        swap = SwapGate()
        qreg = dag.qregs['q']
        layout = self.property_set['final_layout']
        perm = layout.to_permutation(qreg)
        while True:
            inv_perm = {initial: final for final, initial in enumerate(perm) if final != initial}
            if not inv_perm:
                break
            min_moved = min(inv_perm.keys())
            current_pos = inv_perm[min_moved]
            perm[current_pos] = perm[current_pos - 1]
            perm[current_pos - 1] = min_moved
            dag.apply_operation_back(swap, (qreg[current_pos - 1], qreg[current_pos]), (),
                                     check=False)
            layout.swap(current_pos - 1, current_pos)

        self.property_set['final_layout'] = layout

        return dag


def get_opt3_pm(backend, physical_qubits):
    pm = generate_preset_pass_manager(3, backend, initial_layout=physical_qubits)
    pm.routing.append(RestoreQubitOrdering())  # pylint: disable=no-member
    return pm


def _make_sequences(ops_by_weight, op_coeff, circuit):
    op, coeff = op_coeff
    current_weight = op.count('Z')
    ops_by_weight = deepcopy(ops_by_weight)
    ops_by_weight[current_weight].remove(op_coeff)

    control = op[::-1].index('Z')
    cxs = []
    while (target := op[::-1].find('Z', control + 1)) != -1:
        circuit.cx(control, target)
        cxs.append((control, target))
        control = target
    circuit.rz(2. * coeff, control)
    for control, target in cxs[::-1]:
        circuit.cx(control, target)

    circuits = []
    weight = max(1, current_weight - 1)
    while weight > 0 and not ops_by_weight[weight]:
        weight -= 1
    for next_op_coeff in ops_by_weight[weight]:
        circuits.extend(_make_sequences(ops_by_weight, next_op_coeff, circuit.copy()))
    for next_op_coeff in ops_by_weight[current_weight]:
        circuits.extend(_make_sequences(ops_by_weight, next_op_coeff, circuit.copy()))
    weight = current_weight + 1
    while weight <= circuit.num_qubits and not ops_by_weight[weight]:
        weight += 1
    for next_op_coeff in ops_by_weight[weight]:
        circuits.extend(_make_sequences(ops_by_weight, next_op_coeff, circuit.copy()))

    if circuits:
        return circuits

    return [circuit]


def diag_propagator_circuit(spo, backend, physical_qubits):
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
