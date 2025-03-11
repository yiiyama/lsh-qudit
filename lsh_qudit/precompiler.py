"""Precompiler pass to ensure certain gate cancellations take place."""
import numpy as np
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import CXGate, RYGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass


class LSHPrecompiler(TransformationPass):
    """Gate cancellation pass for CX-SWAP, RCCX-SWAP, and clambda+- pairs.

    Our application has specific simplifying gate patterns that may be missed by the qiskit
    transpiler. We therefore insert this pass as the first node of the "init" transpiler stage.
    Specifically, we cancel:

    - Incrementer-decrementer pair: In transition from one hopping term to another)
    - RCCX-SWAP: Occurs in the U_SVD circuit
    - CX-SWAP: Happens occasionally

    Sequential Toffoli gates are handled by the default transpiler properly.

    When using certain transmons as qutrits, we further replace the remaining uncancelled SWAPs with
    three CXs to avoid qubits getting routed by the transpiler.
    """
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        for node in dag.topological_op_nodes():
            op_name = node.op.name
            if any(op_name.startswith(name) for name in [r'cc$\lambda^{-}$', r'c$\lambda^{-}$c']):
                if (op_name.endswith('_dg') and len((suc := list(dag.op_successors(node)))) == 1
                        and suc[0].op.name == op_name[:-3]):
                    dag.remove_op_node(node)
                    dag.remove_op_node(suc[0])
                else:
                    subdag = circuit_to_dag(node.op.definition)
                    mapping = {qubit: qarg for qubit, qarg in zip(subdag.qubits, node.qargs)}
                    dag.substitute_node_with_dag(node, subdag, wires=mapping)

        for node in dag.topological_op_nodes():
            if node.op.name.startswith('rccx_cct'):
                self._replace_swap_rccx_cct(dag, node)
            elif node.op.name.startswith('rccx_ctc'):
                # _replace_swap_rccx_ctc can replace the inverse rccx node; check existence first
                try:
                    dag.node(node._node_id)
                except IndexError:
                    continue
                self._replace_swap_rccx_ctc(dag, node)

        gates_to_decompose = (
            'rccx_cct', 'rccx_cct_dg', 'rccx_ctc', 'rccx_ctc_dg',
            r'c$\lambda^{-}$', r'c$\lambda^{-}$_dg'
        )
        for node in dag.topological_op_nodes():
            if node.op.name not in gates_to_decompose:
                continue
            subdag = circuit_to_dag(node.op.definition)
            mapping = {qubit: qarg for qubit, qarg in zip(subdag.qubits, node.qargs)}
            dag.substitute_node_with_dag(node, subdag, wires=mapping)

        # We want the qubits fixed to the initial layout
        # -> Replace remaining SWAPs with 3 CXs so the downstream passes don't try to
        # route the qubits

        return dag

    def _replace_swap_rccx_cct(self, dag: DAGCircuit, rccx_node: DAGOpNode):
        if rccx_node.op.name.endswith('_dg'):
            outward = dag.op_successors
            inward = dag.op_predecessors
        else:
            outward = dag.op_predecessors
            inward = dag.op_successors
        # Find the adjacent node on the target wire
        target_qubit = rccx_node.qargs[2]
        try:
            test = next(tn for tn in outward(rccx_node) if target_qubit in tn.qargs)
        except StopIteration:
            return

        # Trace outward the 1Q run
        t_1q_nodes = []
        while len(test.qargs) == 1:
            t_1q_nodes.append(test)
            if not (tests := list(outward(test))):
                # We hit the beginning/end of the circuit
                return
            test = tests[0]

        if test.op.name != 'swap' or set(test.qargs) != set(rccx_node.qargs[1:]):
            return

        swap_node = test

        # Find the successor of the SWAP on the C2 qubit
        test = next(tn for tn in inward(test) if rccx_node.qargs[1] in tn.qargs)

        # Trace inward the 1Q run
        c2_1q_nodes = []
        while len(test.qargs) == 1:
            c2_1q_nodes.append(test)
            test = list(inward(test))[0]

        if test != rccx_node:
            return

        # Remove the original 1Q and SWAP gates
        for node in t_1q_nodes + c2_1q_nodes + [swap_node]:
            dag.remove_op_node(node)

        # Substitute the RCCX node with the decomposed and simplified RCCX-SWAP-1Q sequence
        subdag = DAGCircuit()
        qreg = QuantumRegister(3)  # Maps to c1, c2, t
        subdag.add_qreg(qreg)
        if rccx_node.op.name.endswith('_dg'):
            apply_outward = subdag.apply_operation_back
            apply_inward = subdag.apply_operation_front
            angle = np.pi / 4.
        else:
            apply_outward = subdag.apply_operation_front
            apply_inward = subdag.apply_operation_back
            angle = -np.pi / 4.
        # Apply the 1Q gates on t to the new c2, and vice versa
        apply_outward(RYGate(angle), [qreg[1]])
        for node in t_1q_nodes:
            apply_outward(node.op, [qreg[1]])
        for node in c2_1q_nodes:
            apply_inward(node.op, [qreg[2]])
        apply_inward(CXGate(), [qreg[1], qreg[2]])
        apply_inward(RYGate(angle), [qreg[1]])
        apply_inward(CXGate(), [qreg[0], qreg[1]])
        apply_inward(RYGate(-angle), [qreg[1]])
        apply_inward(CXGate(), [qreg[1], qreg[2]])
        apply_inward(CXGate(), [qreg[2], qreg[1]])
        apply_inward(RYGate(-angle), [qreg[2]])

        dag.substitute_node_with_dag(rccx_node, subdag)

    def _replace_swap_rccx_ctc(self, dag: DAGCircuit, rccx_node: DAGOpNode):
        if rccx_node.op.name.endswith('_dg'):
            outward = dag.op_successors
            inward = dag.op_predecessors
        else:
            outward = dag.op_predecessors
            inward = dag.op_successors
        # Find the adjacent node on the target wire
        target_qubit = rccx_node.qargs[1]
        try:
            test = next(tn for tn in outward(rccx_node) if target_qubit in tn.qargs)
        except StopIteration:
            return

        # Trace back the 1Q run
        t_1q_nodes = []
        while len(test.qargs) == 1:
            t_1q_nodes.append(test)
            if not (tests := list(outward(test))):
                # We hit the beginning/end of the circuit
                return
            test = tests[0]

        if test.op.name != 'swap' or not set(test.qargs) < set(rccx_node.qargs):
            return

        swap_node = test
        control_qubit = next(bit for bit in test.qargs if bit != target_qubit)
        idle_qubit = next(bit for bit in test.qargs if bit not in [control_qubit, target_qubit])

        # Find the successor of the SWAP on the C qubit
        test = next(tn for tn in inward(test) if control_qubit in tn.qargs)

        # Trace forward the 1Q run
        c2_1q_nodes = []
        while len(test.qargs) == 1:
            c2_1q_nodes.append(test)
            test = list(inward(test))[0]

        if test != rccx_node:
            return

        # Remove the original 1Q and SWAP gates
        for node in t_1q_nodes + c2_1q_nodes + [swap_node]:
            dag.remove_op_node(node)

        # If the qargs order is going to change, transpose the inverse gate
        # (because RCCX has asymmetric relative phases between C1 and C2)
        if rccx_node.qargs[0] != control_qubit:
            inverse_node = next(tn for tn in inward(rccx_node) if tn.op.name.startswith('rccx_ctc'))
            subdag = DAGCircuit()
            qreg = QuantumRegister(3)  # Maps to c1, t, c2
            subdag.add_qreg(qreg)
            if rccx_node.op.name.endswith('_dg'):
                apply_inward = subdag.apply_operation_front
                angle = np.pi / 4.
            else:
                apply_inward = subdag.apply_operation_back
                angle = -np.pi / 4.
            # Apply the 1Q gates on t to the new c, and vice versa
            apply_inward(RYGate(angle), [qreg[1]])
            apply_inward(CXGate(), [qreg[0], qreg[1]])
            apply_inward(RYGate(angle), [qreg[1]])
            apply_inward(CXGate(), [qreg[2], qreg[1]])
            apply_inward(RYGate(-angle), [qreg[1]])
            apply_inward(CXGate(), [qreg[0], qreg[1]])
            apply_inward(RYGate(-angle), [qreg[1]])

            wires = {qreg[0]: control_qubit, qreg[1]: target_qubit, qreg[2]: idle_qubit}
            dag.substitute_node_with_dag(inverse_node, subdag, wires=wires)

        # Substitute the RCCX node with the decomposed and simplified RCCX-SWAP-1Q sequence
        subdag = DAGCircuit()
        qreg = QuantumRegister(3)  # Maps to c1, t, c2
        subdag.add_qreg(qreg)
        if rccx_node.op.name.endswith('_dg'):
            apply_outward = subdag.apply_operation_back
            apply_inward = subdag.apply_operation_front
            angle = np.pi / 4.
        else:
            apply_outward = subdag.apply_operation_front
            apply_inward = subdag.apply_operation_back
            angle = -np.pi / 4.
        # Apply the 1Q gates on t to the new c, and vice versa
        apply_outward(RYGate(angle), [qreg[0]])
        for node in t_1q_nodes:
            apply_outward(node.op, [qreg[0]])
        for node in c2_1q_nodes:
            apply_inward(node.op, [qreg[1]])
        apply_inward(CXGate(), [qreg[0], qreg[1]])
        apply_inward(CXGate(), [qreg[1], qreg[0]])
        apply_inward(RYGate(angle), [qreg[1]])
        apply_inward(CXGate(), [qreg[2], qreg[1]])
        apply_inward(RYGate(-angle), [qreg[1]])
        apply_inward(CXGate(), [qreg[0], qreg[1]])
        apply_inward(RYGate(-angle), [qreg[1]])

        wires = {qreg[0]: control_qubit, qreg[1]: target_qubit, qreg[2]: idle_qubit}
        dag.substitute_node_with_dag(rccx_node, subdag, wires=wires)
