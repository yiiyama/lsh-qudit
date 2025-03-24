"""Functions to identify the best layout of the LSH SU(2) circuit onto a heavy-hex QPU."""
from typing import Optional
import numpy as np
import rustworkx as rx
from qiskit.providers import Backend
from .utils import QubitPlacement


def layout_candidates(
    backend: Backend,
    qp: QubitPlacement,
    hint: Optional[dict[tuple[str, int], int]] = None,
    readout_rel_weight: float = 1.
) -> list[tuple[list[int], float]]:
    """Return all possible layouts with associated scores.

    Args:
        coupling_map: backend.coupling_map.
        qp: Logical qubit placement.
        hint: Assignment hint dict of form {qubit_label: physical qubit}.
        readout_rel_weight: Relative importance of the readout error with respect to 2Q gate errors.

    Returns:
        List of physical qubit ids to be passed to the transpiler.
    """
    # Express the backend topology as an undirected graph with error rates as node and edge data
    props = backend.properties()
    physical_graph = rx.PyGraph()
    physical_graph.add_nodes_from([(iq, np.log(readout_rel_weight * props.readout_error(iq)))
                                   for iq in range(backend.num_qubits)])
    for gate_prop in props.gates:
        if gate_prop.gate in ['cz', 'ecr']:
            qubits = gate_prop.qubits
            if len(physical_graph.edge_indices_from_endpoints(*qubits)) == 0:
                gate_error = props.gate_error(gate_prop.gate, qubits)
                physical_graph.add_edge(qubits[0], qubits[1], np.log(gate_error))

    # Construct the logical graph
    logical_graph = rx.PyGraph()
    node_ids = logical_graph.add_nodes_from(qp.qubit_labels)
    last_node_id = -1
    for inode, label in enumerate(qp.qubit_labels):
        if label[0].startswith('d'):
            continue
        node_id = node_ids[inode]
        if last_node_id >= 0:
            logical_graph.add_edge(last_node_id, node_id, None)
        last_node_id = node_id

    def label_match(name, site):
        def match(data):
            return data == (name, site)
        return match

    for node_id, label in zip(node_ids, qp.qubit_labels):
        if label[0] in ('d', 'd0'):
            l_node_id = logical_graph.filter_nodes(label_match('l', label[1]))[0]
            logical_graph.add_edge(node_id, l_node_id, None)
        elif label[0].startswith('d'):
            didx = int(label[0][1:])
            lower_node_id = logical_graph.filter_nodes(label_match(f'd{didx - 1}', label[1]))[0]
            logical_graph.add_edge(node_id, lower_node_id, None)

    # Loop over the list of subgraph isomorphisms and find the best mapping
    if hint is not None:
        def node_matcher(phys_data, logi_data):
            if (assignment := hint.get(logi_data)) is None:
                return True
            return phys_data[0] == assignment
    else:
        node_matcher = None

    mappings = rx.vf2_mapping(physical_graph, logical_graph, node_matcher=node_matcher,
                              subgraph=True, induced=False)
    candidates = []
    for mapping in mappings:
        subgraph = physical_graph.subgraph(list(mapping.keys()))
        score = sum(logerr for _, logerr in subgraph.nodes()) + sum(subgraph.edges())
        reverse_map = {logical: physical for physical, logical in mapping.items()}
        candidates.append(([reverse_map[il] for il in range(qp.num_qubits)], score))

    return candidates


def layout_heavy_hex(
    backend: Backend,
    qp: QubitPlacement,
    hint: Optional[dict[tuple[str, int], int]] = None,
    readout_rel_weight: float = 1.
) -> list[int]:
    """Return the physical qubit layout of the qubit graph using qubits in the coupling map.

    Args:
        coupling_map: backend.coupling_map.
        qp: Logical qubit placement.
        hint: Assignment hint dict of form {qubit_label: physical qubit}.
        readout_rel_weight: Relative importance of the readout error with respect to 2Q gate errors.

    Returns:
        List of physical qubit ids to be passed to the transpiler.
    """
    candidates = layout_candidates(backend, qp, hint=hint, readout_rel_weight=readout_rel_weight)
    return min(candidates, key=lambda x: x[1])[0]
