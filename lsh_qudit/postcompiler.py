"""Postcompiler pass to perform additional gate cancellations."""
import numpy as np
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import XGate, RZGate
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler import TransformationPass


class LSHPostcompiler(TransformationPass):
    """Gate cancellation pass for CZ-X-CZ."""
    def run(self, dag: DAGCircuit) -> DAGCircuit:
        # CZ-X-CZ / CX-X-CX
        while True:
            for node in dag.topological_op_nodes():
                if node.op.name == 'cz':
                    suc = list(dag.op_successors(node))
                    if len(suc) != 2:
                        continue
                    try:
                        suc_x = next(n for n in suc if n.op.name == 'x')
                    except StopIteration:
                        continue
                    try:
                        suc_cz = next(n for n in suc if n.op.name == 'cz')
                    except StopIteration:
                        continue
                    if set(suc_cz.qargs) != set(node.qargs):
                        continue
                    # We have CZ-X-CZ
                    dag.remove_op_node(suc_cz)
                    ix = next(i for i, q in enumerate(node.qargs) if q == suc_x.qargs[0])
                    subdag = DAGCircuit()
                    qreg = QuantumRegister(2)
                    subdag.add_qreg(qreg)
                    subdag.apply_operation_back(RZGate(np.pi), [qreg[1 - ix]])
                    dag.substitute_node_with_dag(node, subdag)
                    break
                elif node.op.name == 'cx':
                    suc = list(dag.op_successors(node))
                    if len(suc) != 2:
                        continue
                    try:
                        suc_x = next(n for n in suc if n.op.name == 'x')
                    except StopIteration:
                        continue
                    if suc_x.qargs[0] != node.qargs[0]:
                        continue
                    try:
                        suc_cx = next(n for n in suc if n.op.name == 'cx')
                    except StopIteration:
                        continue
                    if suc_cx.qargs != node.qargs:
                        continue
                    # We have CX-X-CX
                    dag.remove_op_node(suc_cx)
                    subdag = DAGCircuit()
                    qreg = QuantumRegister(2)
                    subdag.add_qreg(qreg)
                    subdag.apply_operation_back(XGate(), [qreg[1]])
                    dag.substitute_node_with_dag(node, subdag)
                    break
            else:
                break

        return dag
