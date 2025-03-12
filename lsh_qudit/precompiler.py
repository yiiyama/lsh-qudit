"""Precompiler pass to ensure certain gate cancellations take place."""
import numpy as np
from qiskit.circuit import QuantumRegister
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.library import CCZGate, CXGate, HGate, RYGate
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass

from .parity_network import parity_walk_up, parity_walk_down, parity_walk_upr, parity_walk_downr
from .utils import diag_to_iz


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
        # Incrementer-decrementer pair (including CCX-CCX)
        for node in dag.topological_op_nodes():
            op_name = node.op.name
            if any(op_name.startswith(name) for name in [r'cc$\lambda^{-}$', r'c$\lambda^{-}$c']):
                if (op_name.endswith('_dg') and len((suc := list(dag.op_successors(node)))) == 1
                        and suc[0].op.name == op_name[:-3]):
                    dag.remove_op_node(node)
                    dag.remove_op_node(suc[0])
                else:
                    subdag = circuit_to_dag(node.op.definition)
                    dag.substitute_node_with_dag(node, subdag)

            elif op_name == 'ccx':
                try:
                    suc = list(dag.op_successors(node))
                except IndexError:  # if this node was removed with the preceding ccx
                    continue
                if len(suc) == 1 and suc[0].op.name == 'ccx' and suc[0].qargs == node.qargs:
                    dag.remove_op_node(node)
                    dag.remove_op_node(suc[0])

        # Replace all CCXs with H-CCZ-H
        for node in dag.topological_op_nodes():
            if node.op.name == 'ccx':
                subdag = DAGCircuit()
                qreg = QuantumRegister(3)
                subdag.add_qreg(qreg)
                subdag.apply_operation_back(HGate(), [qreg[2]])
                subdag.apply_operation_back(CCZGate(), [qreg[0], qreg[1], qreg[2]])
                subdag.apply_operation_back(HGate(), [qreg[2]])
                dag.substitute_node_with_dag(node, subdag)

        # RCCX-SWAP/CX and CCZ-SWAP/CX
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
            elif node.op.name == 'ccz':
                if not self._replace_swapcx_ccz(dag, node, 1):
                    self._replace_swapcx_ccz(dag, node, -1)

        # Decompose the remaining custom gates to expose the CXs
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

        # SWAP-CX
        for node in dag.topological_op_nodes():
            pass

        # We want the qubits fixed to the initial layout
        # -> Replace remaining SWAPs with 3 CXs so the downstream passes don't try to
        # route the qubits

        return dag

    def _replace_swap_rccx_cct(self, dag: DAGCircuit, rccx_node: DAGOpNode):
        # Find the adjacent node on the target wire
        target_qubit = rccx_node.qargs[2]
        nodes_on_t = list(dag.nodes_on_wire(target_qubit, only_ops=True))
        rccx_node_idx = nodes_on_t.index(rccx_node)

        if rccx_node.op.name.endswith('_dg'):
            outward = dag.op_successors
            inward = dag.op_predecessors
            out_incr = 1
            if rccx_node_idx == len(nodes_on_t) - 1:
                return
        else:
            outward = dag.op_predecessors
            inward = dag.op_successors
            out_incr = -1
            if rccx_node_idx == 0:
                return

        test = nodes_on_t[rccx_node_idx + out_incr]
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
        nodes_on_c2 = list(dag.nodes_on_wire(rccx_node.qargs[1], only_ops=True))
        swap_node_idx = nodes_on_c2.index(swap_node)

        # Find the successor of the SWAP on the C2 qubit
        test = nodes_on_c2[swap_node_idx - out_incr]

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
        target_qubit = rccx_node.qargs[1]
        nodes_on_t = list(dag.nodes_on_wire(target_qubit, only_ops=True))
        rccx_node_idx = nodes_on_t.index(rccx_node)

        if rccx_node.op.name.endswith('_dg'):
            outward = dag.op_successors
            inward = dag.op_predecessors
            out_incr = 1
            if rccx_node_idx == len(nodes_on_t) - 1:
                return
        else:
            outward = dag.op_predecessors
            inward = dag.op_successors
            out_incr = -1
            if rccx_node_idx == 0:
                return

        test = nodes_on_t[rccx_node_idx + out_incr]
        # Trace back the 1Q run
        t_1q_nodes = []
        while len(test.qargs) == 1:
            t_1q_nodes.append(test)
            if not (tests := list(outward(test))):
                # We hit the beginning/end of the circuit
                return
            test = tests[0]

        if test.op.name != 'swap' or set(test.qargs) - set(rccx_node.qargs):
            return

        swap_node = test
        control_qubit = next(bit for bit in swap_node.qargs if bit != target_qubit)
        idle_qubit = next(bit for bit in rccx_node.qargs
                          if bit not in [target_qubit, control_qubit])
        nodes_on_c = list(dag.nodes_on_wire(control_qubit, only_ops=True))
        swap_node_idx = nodes_on_c.index(swap_node)

        # Find the successor of the SWAP on the C2 qubit
        test = nodes_on_c[swap_node_idx - out_incr]

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

    def _replace_swapcx_ccz(self, dag: DAGCircuit, ccz_node: DAGOpNode, direction: int):
        qreg = list(dag.qregs.values())[0]
        ordered_qubits = sorted(ccz_node.qargs, key=qreg.index)
        nodes_on_cent = list(dag.nodes_on_wire(ordered_qubits[1], only_ops=True))
        ccz_node_idx = nodes_on_cent.index(ccz_node)
        if direction > 0:
            outward = dag.op_successors
            inward = dag.op_predecessors
            out_incr = 1
            if ccz_node_idx == len(nodes_on_cent) - 1:
                return
        else:
            outward = dag.op_predecessors
            inward = dag.op_successors
            out_incr = -1
            if ccz_node_idx == 0:
                return

        test = nodes_on_cent[ccz_node_idx + out_incr]
        # Trace outward the 1Q run
        qcent_1q_nodes = []
        while len(test.qargs) == 1:
            qcent_1q_nodes.append(test)
            if not (tests := list(outward(test))):
                # We hit the beginning/end of the circuit
                return False
            test = tests[0]

        if test.op.name not in ['cx', 'swap'] or set(test.qargs) - set(ccz_node.qargs):
            return False

        twoq_node = test
        other_wire = next(bit for bit in twoq_node.qargs if bit != ordered_qubits[1])
        other_wire_idx = next(idx for idx, bit in enumerate(ccz_node.qargs) if bit == other_wire)
        nodes_on_other = list(dag.nodes_on_wire(other_wire, only_ops=True))
        twoq_node_idx = nodes_on_other.index(twoq_node)

        # Move inward on the other wire
        test = nodes_on_other[twoq_node_idx - out_incr]

        # Trace inward the 1Q run
        qother_1q_nodes = []
        while len(test.qargs) == 1:
            qother_1q_nodes.append(test)
            test = list(inward(test))[0]

        if test != ccz_node:
            return False

        subdag = DAGCircuit()
        qreg = QuantumRegister(3)
        subdag.add_qreg(qreg)
        if direction > 0:
            apply_outward = subdag.apply_operation_back
            apply_inward = subdag.apply_operation_front
        else:
            apply_outward = subdag.apply_operation_front
            apply_inward = subdag.apply_operation_back

        hccz = np.zeros((2, 2, 2))
        hccz[1, 1, 1] = np.pi
        angles = diag_to_iz(hccz)

        if twoq_node.op.name == 'cx':
            if any(not scc.commute_nodes(twoq_node, node)
                    for node in qcent_1q_nodes + qother_1q_nodes):
                return False

            ctrl = next(idx for idx, bit in enumerate(ccz_node.qargs) if bit == twoq_node.qargs[0])
            targ = next(idx for idx, bit in enumerate(ccz_node.qargs) if bit == twoq_node.qargs[1])

            # All 1Q commute; move them to the other side of CX and align the CCZ decomposition
            for node in qcent_1q_nodes:
                apply_outward(node.op, [qreg[1]])
            for node in qother_1q_nodes:
                apply_inward(node.op, [qreg[other_wire_idx]])
            apply_inward(CXGate(), [qreg[ctrl], qreg[targ]])

            match (direction, ctrl, targ):
                case (1, 0, 1):
                    pwalk = parity_walk_upr
                case (1, 1, 0):
                    pwalk = parity_walk_down
                case (1, 1, 2):
                    pwalk = parity_walk_up
                case (1, 2, 1):
                    pwalk = parity_walk_downr
                case (-1, 0, 1):
                    pwalk = parity_walk_up
                case (-1, 1, 0):
                    pwalk = parity_walk_downr
                case (-1, 1, 2):
                    pwalk = parity_walk_upr
                case (-1, 2, 1):
                    pwalk = parity_walk_down
                case _:
                    raise ValueError('Wrong direction value?')
        else:
            # Move all 1Q nodes to the other side of the SWAP
            for node in qcent_1q_nodes:
                apply_outward(node.op, [qreg[other_wire_idx]])
            for node in qother_1q_nodes:
                apply_inward(node.op, [qreg[1]])

            match (direction, other_wire_idx):
                case (1, 0):
                    pwalk = parity_walk_upr
                case (1, 2):
                    pwalk = parity_walk_downr
                case (-1, 0):
                    pwalk = parity_walk_up
                case (-1, 2):
                    pwalk = parity_walk_down
                case _:
                    raise ValueError('Wrong direction value?')

            apply_inward(CXGate(), [qreg[other_wire_idx], qreg[1]])
            apply_inward(CXGate(), [qreg[1], qreg[other_wire_idx]])
            apply_inward(CXGate(), [qreg[other_wire_idx], qreg[1]])

        # pylint: disable-next=used-before-assignment
        pwalk_dag = circuit_to_dag(pwalk(angles, singles_front=direction > 0))
        if direction > 0:
            subdag = pwalk_dag.compose(subdag, inplace=False)
        else:
            subdag.compose(pwalk_dag, inplace=True)

        # Remove the original 1Q and SWAP gates
        for node in qcent_1q_nodes + qother_1q_nodes + [twoq_node]:
            dag.remove_op_node(node)

        dag.substitute_node_with_dag(ccz_node, subdag)
        return True
