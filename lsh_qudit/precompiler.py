"""Precompiler pass to ensure certain gate cancellations take place."""
import numpy as np
from qiskit.circuit import QuantumRegister
from qiskit.circuit.commutation_library import SessionCommutationChecker as scc
from qiskit.circuit.library import CCZGate, CXGate, HGate, RYGate, RZGate
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
    - CP-SWAP: Occurs after the H_E[3] circuit
    - CX-SWAP: Happens occasionally

    Sequential Toffoli gates are handled by the default transpiler properly.

    When using certain transmons as qutrits, we further replace the remaining uncancelled SWAPs with
    three CXs to avoid qubits getting routed by the transpiler.
    """
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # Replace all CCXs with H-CCZ-H and CCX+/-s with RCCX-CX+--RCCX
        for node in dag.topological_op_nodes():
            if node.op.name == 'ccx':
                subdag = DAGCircuit()
                qreg = QuantumRegister(3)
                subdag.add_qreg(qreg)
                subdag.apply_operation_back(HGate(), [qreg[2]])
                subdag.apply_operation_back(CCZGate(), [qreg[0], qreg[1], qreg[2]])
                subdag.apply_operation_back(HGate(), [qreg[2]])
                dag.substitute_node_with_dag(node, subdag)
            elif node.op.name in ('ccxminus', 'ccxplus'):
                subdag = circuit_to_dag(node.op.definition)
                dag.substitute_node_with_dag(node, subdag)

        # RCCX-SWAP/CX and CCZ-SWAP/CX
        while True:
            for node in dag.topological_op_nodes():
                if node.op.name == 'rccx':
                    ordered_qargs = sorted(node.qargs, key=lambda bit: dag.find_bit(bit).index)
                    if node.qargs[2] == ordered_qargs[1]:
                        if self._replace_swap_rccx_ctc(dag, node):
                            break
                    else:
                        if self._replace_swap_rccx_cct(dag, node):
                            break
                if (node.op.name == 'ccz'
                        and (self._replace_swapcx_ccz(dag, node, 1)
                             or self._replace_swapcx_ccz(dag, node, -1))):
                    break
            else:
                break

        # Decompose the remaining custom gates to expose the CXs
        for node in dag.topological_op_nodes():
            if node.op.name == 'rccx':
                ordered_qargs = sorted(node.qargs, key=lambda bit: dag.find_bit(bit).index)
                if node.qargs[2] == ordered_qargs[1]:
                    subdag = circuit_to_dag(node.op.definition)
                else:
                    # Compile for linear topology
                    subdag = DAGCircuit()
                    qreg = QuantumRegister(3)
                    subdag.add_qreg(qreg)
                    subdag.apply_operation_back(RYGate(-np.pi / 4.), [qreg[2]])
                    subdag.apply_operation_back(CXGate(), [qreg[2], qreg[1]])
                    subdag.apply_operation_back(CXGate(), [qreg[1], qreg[2]])
                    subdag.apply_operation_back(RYGate(-np.pi / 4.), [qreg[1]])
                    subdag.apply_operation_back(CXGate(), [qreg[0], qreg[1]])
                    subdag.apply_operation_back(RYGate(np.pi / 4.), [qreg[1]])
                    subdag.apply_operation_back(CXGate(), [qreg[1], qreg[2]])
                    subdag.apply_operation_back(CXGate(), [qreg[2], qreg[1]])
                    subdag.apply_operation_back(RYGate(np.pi / 4.), [qreg[2]])
            elif node.op.name in ('cxminus', 'cxplus', 'cq'):
                subdag = circuit_to_dag(node.op.definition)
            else:
                continue
            mapping = {qubit: qarg for qubit, qarg in zip(subdag.qubits, node.qargs)}
            dag.substitute_node_with_dag(node, subdag, wires=mapping)

        # SWAP-SWAP/CX and CP-SWAP/CX
        while True:
            for node in dag.topological_op_nodes():
                if (node.op.name in ('swap', 'cp')
                        and (self._replace_swapcx_swapcp(dag, node, 1)
                             or self._replace_swapcx_swapcp(dag, node, -1))):
                    break
            else:
                break

        # We want the qubits fixed to the initial layout
        # -> Resolve the remaining CCZs and SWAPs so the downstream passes don't try to route the
        # qubits
        for node in dag.topological_op_nodes():
            if node.op.name == 'swap':
                subdag = DAGCircuit()
                qreg = QuantumRegister(2)
                subdag.add_qreg(qreg)
                subdag.apply_operation_back(CXGate(), [qreg[0], qreg[1]])
                subdag.apply_operation_back(CXGate(), [qreg[1], qreg[0]])
                subdag.apply_operation_back(CXGate(), [qreg[0], qreg[1]])
                dag.substitute_node_with_dag(node, subdag)
            elif node.op.name == 'ccz':
                hccz = np.zeros((2, 2, 2))
                hccz[1, 1, 1] = np.pi
                subdag = circuit_to_dag(parity_walk_up(diag_to_iz(hccz)))
                ordered_qargs = sorted(node.qargs, key=lambda bit: dag.find_bit(bit).index)
                dag.substitute_node_with_dag(node, subdag,
                                             wires=dict(zip(subdag.qubits, ordered_qargs)))
            elif node.op.name == 'cp':
                phi = node.op.params[0]
                subdag = DAGCircuit()
                qreg = QuantumRegister(2)
                subdag.add_qreg(qreg)
                subdag.apply_operation_back(RZGate(phi / 2.), [qreg[0]])
                subdag.apply_operation_back(RZGate(phi / 2.), [qreg[1]])
                subdag.apply_operation_back(CXGate(), [qreg[0], qreg[1]])
                subdag.apply_operation_back(RZGate(-phi / 2.), [qreg[1]])
                subdag.apply_operation_back(CXGate(), [qreg[0], qreg[1]])
                dag.substitute_node_with_dag(node, subdag)

        return dag

    def _replace_swapcx_swapcp(self, dag: DAGCircuit, swapcp_node: DAGOpNode, out_incr: int):
        nodes_on_q0 = list(dag.nodes_on_wire(swapcp_node.qargs[0], only_ops=True))
        swap_node_idx = nodes_on_q0.index(swapcp_node)
        if out_incr > 0:
            outward = dag.op_successors
            inward = dag.op_predecessors
            out_incr = 1
            if swap_node_idx == len(nodes_on_q0) - 1:
                return False
        else:
            outward = dag.op_predecessors
            inward = dag.op_successors
            out_incr = -1
            if swap_node_idx == 0:
                return False

        test = nodes_on_q0[swap_node_idx + out_incr]
        # Trace outward the 1Q run
        q0_1q_nodes = []
        while len(test.qargs) == 1:
            q0_1q_nodes.append(test)
            if not (tests := list(outward(test))):
                # We hit the beginning/end of the circuit
                return False
            test = tests[0]

        if test.op.name not in ['cx', 'swap'] or set(test.qargs) != set(swapcp_node.qargs):
            return False

        twoq_node = test
        nodes_on_q1 = list(dag.nodes_on_wire(swapcp_node.qargs[1], only_ops=True))
        twoq_node_idx = nodes_on_q1.index(twoq_node)

        # Move inward on the other wire
        test = nodes_on_q1[twoq_node_idx - out_incr]

        # Trace inward the 1Q run
        q1_1q_nodes = []
        while len(test.qargs) == 1:
            q1_1q_nodes.append(test)
            test = list(inward(test))[0]

        if test != swapcp_node:
            return False

        subdag = DAGCircuit()
        qreg = QuantumRegister(2)
        subdag.add_qreg(qreg)
        if out_incr > 0:
            apply_outward = subdag.apply_operation_back
            apply_inward = subdag.apply_operation_front
        else:
            apply_outward = subdag.apply_operation_front
            apply_inward = subdag.apply_operation_back

        if swapcp_node.op.name == 'swap':
            # Move 1Q gates to the other side of swap
            for node in q0_1q_nodes:
                apply_outward(node.op, [qreg[1]])
            for node in q1_1q_nodes:
                apply_inward(node.op, [qreg[0]])

            if twoq_node.op.name == 'cx':
                ctrl = next(idx for idx, bit in enumerate(swapcp_node.qargs)
                            if bit == twoq_node.qargs[0])
                targ = 1 - ctrl
                apply_outward(CXGate(), [qreg[ctrl], qreg[targ]])
                apply_outward(CXGate(), [qreg[targ], qreg[ctrl]])
        else:  # cp
            phi = swapcp_node.op.params[0]
            if twoq_node.op.name == 'cx':
                ctrl = next(idx for idx, bit in enumerate(swapcp_node.qargs)
                            if bit == twoq_node.qargs[0])
                targ = 1 - ctrl
                apply_outward(RZGate(phi / 2.), [qreg[ctrl]])
                apply_outward(RZGate(phi / 2.), [qreg[targ]])
                apply_outward(CXGate(), [qreg[ctrl], qreg[targ]])
                apply_outward(RZGate(-phi / 2.), [qreg[targ]])

                for nodes, qubit in [(list(q0_1q_nodes), qreg[0]),
                                     (list(q1_1q_nodes[::-1]), qreg[1])]:
                    cp_side = 0 if out_incr > 0 else -1
                    cx_side = -1 if out_incr > 0 else 0
                    cp_side_nodes = []
                    cx_side_nodes = []
                    while nodes:
                        if scc.commute_nodes(swapcp_node, nodes[cp_side]):
                            cp_side_nodes.append(nodes.pop(cp_side))
                        elif scc.commute_nodes(twoq_node, nodes[cx_side]):
                            cx_side_nodes.append(nodes.pop(cx_side))
                        else:
                            break
                    if nodes:
                        # Has non-commuting 1Q gate in between
                        return False

                    for node in cp_side_nodes[::-1]:
                        apply_inward(node.op, [qubit])
                    for node in cx_side_nodes[::-1]:
                        apply_outward(node.op, [qubit])
            else:
                # Move the 1Q gates to the other side of the swap
                for node in q0_1q_nodes:
                    apply_outward(node.op, [qreg[1]])
                for node in q1_1q_nodes:
                    apply_inward(node.op, [qreg[0]])
                apply_inward(CXGate(), [qreg[0], qreg[1]])
                apply_inward(CXGate(), [qreg[1], qreg[0]])
                apply_inward(RZGate(-phi / 2.), [qreg[1]])
                apply_inward(CXGate(), [qreg[0], qreg[1]])
                apply_inward(RZGate(phi / 2.), [qreg[0]])
                apply_inward(RZGate(phi / 2.), [qreg[1]])

        # Remove the original 1Q and SWAP/CX gates
        for node in q0_1q_nodes + q1_1q_nodes + [twoq_node]:
            dag.remove_op_node(node)

        dag.substitute_node_with_dag(swapcp_node, subdag)
        return True

    def _replace_swap_rccx_cct(self, dag: DAGCircuit, rccx_node: DAGOpNode):
        # Find the adjacent node on the target wire
        target_qubit = rccx_node.qargs[2]
        nodes_on_t = list(dag.nodes_on_wire(target_qubit, only_ops=True))
        rccx_node_idx = nodes_on_t.index(rccx_node)

        if any(node.op.name == 'rccx' for node in dag.op_successors(rccx_node)):
            outward = dag.op_predecessors
            inward = dag.op_successors
            if rccx_node_idx == 0:
                return False
            # Will explore to the left
            out_incr = -1
        else:
            outward = dag.op_successors
            inward = dag.op_predecessors
            if rccx_node_idx == len(nodes_on_t) - 1:
                return False
            # Will explore to the right
            out_incr = 1

        test = nodes_on_t[rccx_node_idx + out_incr]
        # Trace outward the 1Q run
        t_1q_nodes = []
        while len(test.qargs) == 1:
            t_1q_nodes.append(test)
            if not (tests := list(outward(test))):
                # We hit the beginning/end of the circuit
                return False
            test = tests[0]

        if test.op.name != 'swap' or set(test.qargs) != set(rccx_node.qargs[1:]):
            return False

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
            return False

        # Remove the original 1Q and SWAP gates
        for node in t_1q_nodes + c2_1q_nodes + [swap_node]:
            dag.remove_op_node(node)

        # Substitute the RCCX node with the decomposed and simplified RCCX-SWAP-1Q sequence
        subdag = DAGCircuit()
        qreg = QuantumRegister(3)  # Maps to c1, c2, t
        subdag.add_qreg(qreg)
        if out_incr > 0:
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
        apply_inward(CXGate(), [qreg[2], qreg[1]])
        apply_inward(RYGate(angle), [qreg[1]])
        apply_inward(CXGate(), [qreg[0], qreg[1]])
        apply_inward(RYGate(-angle), [qreg[1]])
        apply_inward(CXGate(), [qreg[1], qreg[2]])
        apply_inward(CXGate(), [qreg[2], qreg[1]])
        apply_inward(RYGate(-angle), [qreg[2]])

        dag.substitute_node_with_dag(rccx_node, subdag)
        return True

    def _replace_swap_rccx_ctc(self, dag: DAGCircuit, rccx_node: DAGOpNode):
        target_qubit = rccx_node.qargs[2]
        nodes_on_t = list(dag.nodes_on_wire(target_qubit, only_ops=True))
        rccx_node_idx = nodes_on_t.index(rccx_node)

        try:
            inverse_node = next(node for node in dag.op_successors(rccx_node)
                                if node.op.name == 'rccx')
        except StopIteration:
            inverse_node = next(node for node in dag.op_predecessors(rccx_node)
                                if node.op.name == 'rccx')
            outward = dag.op_successors
            inward = dag.op_predecessors
            out_incr = 1
            if rccx_node_idx == len(nodes_on_t) - 1:
                return False
            # Will explore to the right
            out_incr = 1
        else:
            outward = dag.op_predecessors
            inward = dag.op_successors
            if rccx_node_idx == 0:
                return False
            # Will explore to the left
            out_incr = -1

        test = nodes_on_t[rccx_node_idx + out_incr]
        # Trace back the 1Q run
        t_1q_nodes = []
        while len(test.qargs) == 1:
            t_1q_nodes.append(test)
            if not (tests := list(outward(test))):
                # We hit the beginning/end of the circuit
                return False
            test = tests[0]

        if test.op.name != 'swap' or set(test.qargs) - set(rccx_node.qargs):
            return False

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
            return False

        # Remove the original 1Q and SWAP gates
        for node in t_1q_nodes + c2_1q_nodes + [swap_node]:
            dag.remove_op_node(node)

        # Substitute the RCCX node with the decomposed and simplified RCCX-SWAP-1Q sequence
        subdag = DAGCircuit()
        qreg = QuantumRegister(3)  # Maps to c1, t, c2
        subdag.add_qreg(qreg)
        if out_incr > 0:
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

        # Substitute the inverse node
        subdag = DAGCircuit()
        qreg = QuantumRegister(3)  # Maps to c1, t, c2
        subdag.add_qreg(qreg)
        if out_incr > 0:
            apply_inward = subdag.apply_operation_front
            angle = np.pi / 4.
        else:
            apply_inward = subdag.apply_operation_back
            angle = -np.pi / 4.
        apply_inward(RYGate(angle), [qreg[1]])
        apply_inward(CXGate(), [qreg[0], qreg[1]])
        apply_inward(RYGate(angle), [qreg[1]])
        apply_inward(CXGate(), [qreg[2], qreg[1]])
        apply_inward(RYGate(-angle), [qreg[1]])
        apply_inward(CXGate(), [qreg[0], qreg[1]])
        apply_inward(RYGate(-angle), [qreg[1]])

        wires = {qreg[0]: control_qubit, qreg[1]: target_qubit, qreg[2]: idle_qubit}
        dag.substitute_node_with_dag(inverse_node, subdag, wires=wires)
        return True

    def _replace_swapcx_ccz(self, dag: DAGCircuit, ccz_node: DAGOpNode, out_incr: int):
        qreg = list(dag.qregs.values())[0]
        ordered_qargs = sorted(ccz_node.qargs, key=qreg.index)
        nodes_on_cent = list(dag.nodes_on_wire(ordered_qargs[1], only_ops=True))
        ccz_node_idx = nodes_on_cent.index(ccz_node)
        if out_incr > 0:
            outward = dag.op_successors
            inward = dag.op_predecessors
            if ccz_node_idx == len(nodes_on_cent) - 1:
                return False
        else:
            outward = dag.op_predecessors
            inward = dag.op_successors
            if ccz_node_idx == 0:
                return False

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
        other_wire = next(bit for bit in twoq_node.qargs if bit != ordered_qargs[1])
        other_wire_idx = next(idx for idx, bit in enumerate(ordered_qargs) if bit == other_wire)
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
        if out_incr > 0:
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

            ctrl = next(idx for idx, bit in enumerate(ordered_qargs) if bit == twoq_node.qargs[0])
            targ = next(idx for idx, bit in enumerate(ordered_qargs) if bit == twoq_node.qargs[1])

            # All 1Q commute; move them to the other side of CX and align the CCZ decomposition
            for node in qcent_1q_nodes:
                apply_outward(node.op, [qreg[1]])
            for node in qother_1q_nodes:
                apply_inward(node.op, [qreg[other_wire_idx]])
            apply_inward(CXGate(), [qreg[ctrl], qreg[targ]])

            match (out_incr, ctrl, targ):
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
                    raise ValueError('Wrong out_incr value? (dir, ctrl, targ)='
                                     f'{(out_incr, ctrl, targ)}')
        else:
            # Move all 1Q nodes to the other side of the SWAP
            for node in qcent_1q_nodes:
                apply_outward(node.op, [qreg[other_wire_idx]])
            for node in qother_1q_nodes:
                apply_inward(node.op, [qreg[1]])

            match (out_incr, other_wire_idx):
                case (1, 0):
                    pwalk = parity_walk_upr
                case (1, 2):
                    pwalk = parity_walk_downr
                case (-1, 0):
                    pwalk = parity_walk_up
                case (-1, 2):
                    pwalk = parity_walk_down
                case _:
                    raise ValueError('Wrong out_incr value? (dir, other_wire_idx)='
                                     f'{(out_incr, other_wire_idx)}')

            apply_inward(CXGate(), [qreg[other_wire_idx], qreg[1]])
            apply_inward(CXGate(), [qreg[1], qreg[other_wire_idx]])
            apply_inward(CXGate(), [qreg[other_wire_idx], qreg[1]])

        # pylint: disable-next=used-before-assignment
        pwalk_dag = circuit_to_dag(pwalk(angles, singles_front=out_incr > 0))
        if out_incr > 0:
            pwalk_dag.compose(subdag, inplace=True)
            subdag = pwalk_dag
            qreg = list(subdag.qregs.values())[0]
        else:
            subdag.compose(pwalk_dag, inplace=True)

        # Remove the original 1Q and SWAP gates
        for node in qcent_1q_nodes + qother_1q_nodes + [twoq_node]:
            dag.remove_op_node(node)

        dag.substitute_node_with_dag(ccz_node, subdag,
                                     wires=dict(zip(qreg, ordered_qargs)))
        return True
