"""Gray-Synth algorithm for compiling phase polynomials assuming all-to-all connection."""
import numpy as np
from qiskit.circuit import ParameterExpression, QuantumCircuit
from .utils import is_zero


def gray_synth(
    angles: np.ndarray,
    parallel: bool = False
) -> QuantumCircuit:
    r"""Synthesize an Rz+CX circuit that encodes a diagonal unitary using ancilla-free Gray-Synth.

    The unitary encoded by the returned circuit is

    .. math::

        U(\theta) = \sum_{\mathbf{x} \in \{0,1\}^{n_q}}
            \exp \left(-i \theta [\mathbf{x}] \bigotimes_{i=0}^{n_q-1} Z^{x_i}\right).

    This function automatically performs an optimization if there is one qubit whose 0th entry is
    identically null by making this qubit the target of all CXs (the qubit plays the role of the
    ancilla of standard Gray-Synth).

    The depth of the synthesized circuit can be reduced by factor 2 (asymptotically) with a cost of
    two extra CXs. Set the parallel option to True to synthesize this depth-optimized version of the
    circuit.

    Args:
        angles: A (2,)*nq shaped array of Rz angles.
        parallel: Performs depth optimization.

    Returns:
        A quantum circuit encoding a diagonal unitary.
    """
    num_qubits = len(angles.shape)
    circuit = QuantumCircuit(num_qubits)

    terminal = num_qubits == 1
    # Check for a constant-Z qubit
    for target in range(num_qubits):
        if np.allclose(np.moveaxis(angles, -1 - target, 0)[0], 0.):
            terminal = True
            break

    if parallel and num_qubits > 3:
        # Create a second target and run two parallel synthesis
        subtarget = (target - 1) % num_qubits
        circuit.cx(target, subtarget)
        angles_target_zi = np.moveaxis(angles,
                                       [-1 - target, -1 - subtarget], [0, 1])[1, 0]
        angles_target_zz = np.moveaxis(angles,
                                       [-1 - target, -1 - subtarget], [0, 1])[1, 1]
        sequence_zi = gray_synth_with_ancilla(angles_target_zi, 0)
        sequence_zz = gray_synth_with_ancilla(angles_target_zz, 1)
        qubits = list(range(num_qubits))
        qubits.remove(target)
        qubits.remove(subtarget)
        for (control_zi, rz_angle_zi), (control_zz, rz_angle_zz) in zip(sequence_zi, sequence_zz):
            if control_zi is not None:
                circuit.cx(qubits[control_zi], target)
            if control_zz is not None:
                circuit.cx(qubits[control_zz], subtarget)
            if rz_angle_zi is not None:
                circuit.rz(rz_angle_zi, target)
            if rz_angle_zz is not None:
                circuit.rz(rz_angle_zz, subtarget)

        circuit.cx(target, subtarget)
    else:
        angles_target_z = np.moveaxis(angles, -1 - target, 0)[1]
        qubits = list(range(num_qubits))
        qubits.remove(target)
        for control, rz_angle in gray_synth_with_ancilla(angles_target_z):
            if control is not None:
                circuit.cx(qubits[control], target)
            if rz_angle is not None:
                circuit.rz(rz_angle, target)

    if terminal:
        return circuit

    angles = np.moveaxis(angles, -1 - target, 0)[0]
    cycle = gray_synth(angles, parallel=parallel)
    qubits = list(range(num_qubits))
    qubits.remove(target)
    circuit.compose(cycle, qubits=qubits, inplace=True)

    return circuit


def gray_synth_with_ancilla(
    angles: np.ndarray,
    start_qubit: int = 0
) -> list[tuple[int | None, float | ParameterExpression | None]]:
    """Return a Gray-code cycle to be synthesized into a circuit.

    Args:
        angles: A (2,)*nq shaped array of Rz angles.
        start_qubit: Index of the least significant bit of the Gray code.

    Returns:
        A list of 2-tuples (CX control qubit, Rz angle or None).
    """
    num_data_qubits = len(angles.shape)
    if num_data_qubits == 0:
        return [(None, 2. * angles)]
    if num_data_qubits > 8:
        raise NotImplementedError('Gray code sequence for nq > 8 unimplemented')
    idx = np.arange(2 ** num_data_qubits, dtype=np.uint8)[:, None]
    gray_code = np.unpackbits(idx ^ (idx >> 1), axis=1)[:, -num_data_qubits:]
    # End with 00..0
    gray_code = np.roll(gray_code, -1, axis=0)
    # Roll for nonzero start_qubit
    gray_code = np.roll(gray_code, -start_qubit, axis=1)

    sequence = []
    prev = np.zeros(num_data_qubits, dtype=np.uint8)
    for state in gray_code:
        control = num_data_qubits - 1 - int(np.nonzero(prev != state)[0][0])
        if is_zero((angle := angles[tuple(state)])):
            sequence.append((control, None))
        else:
            sequence.append((control, 2. * angle))
        prev = state

    if all(rz_angle is None for _, rz_angle in sequence[:-1]):
        return [(None, sequence[-1][1])]

    return sequence
