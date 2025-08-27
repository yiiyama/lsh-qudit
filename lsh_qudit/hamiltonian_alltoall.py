"""Functions to generate circuits for components of the LSH SU(2) Hamiltonian (all-to-all)."""
from dataclasses import dataclass
from numbers import Number
from typing import Optional
import numpy as np
from qiskit.circuit import ParameterExpression, QuantumCircuit, QuantumRegister, Qubit
from .set_rccx_inverse import RCCXGate
from .constants import BOSON_TRUNC, BOSONIC_QUBITS
from .agl import physical_states, boundary_conditions
from .utils import QubitPlacement, diag_to_iz
from .parity_network import parity_network

REG_NAMES = ['i', 'o', 'a'] + [f'l{i}' for i in range(BOSONIC_QUBITS)]


def make_circuit(num_sites: int, *reg_names):
    """Helper function to construct a circuit with named qubits."""
    if not reg_names:
        reg_names = REG_NAMES
    regs = {name: QuantumRegister(num_sites, name) for name in reg_names}
    return QuantumCircuit(*regs.values()), regs


def embed_site_circuit(
    full_circ: QuantumCircuit,
    site_circ: QuantumCircuit,
    sites: int | dict[int, int]
):
    """Compose a site circuit onto the full circuit"""
    if isinstance(sites, int):
        sites = [sites]
    full_regs = {reg.name: reg for reg in full_circ.qregs}
    qubit_mapping = []
    for qubit in site_circ.qubits:
        reg, idx = site_circ.find_bit(qubit).registers[0]
        qubit_mapping.append(full_regs[reg.name][sites[idx]])
    full_circ.compose(site_circ, qubits=qubit_mapping, inplace=True)
    return full_circ


def mass_term_site(
    parity: int,
    mass_mu: Number | ParameterExpression,
    time_step: Number | ParameterExpression
) -> tuple[QuantumCircuit, QubitPlacement, QubitPlacement]:
    r"""Local mass term circuit.

    Mass term is given by
    $$
    \begin{align*}
    H_{M}(r) &= H_{M}^{(1)}(r) + H_{M}^{(2)}(r)\,, \\
    H_{M}^{(1)}(r) &= \frac{\mu}{2}(-)^{r+1}Z_{i(r)}\,,\\
    H_{M}^{(2)}(r) &= \frac{\mu}{2}(-)^{r+1}Z_{o(r)}\,
    \end{align*}
    $$
    and has alternating signs depending on the parity of the site number.
    """
    sign = -1 + 2 * parity
    circuit, regs = make_circuit(1, 'i', 'o')
    circuit.rz(sign * mass_mu * time_step, regs['i'])
    circuit.rz(sign * mass_mu * time_step, regs['o'])
    return circuit


def mass_term(
    num_sites: int,
    mass_mu: Number | ParameterExpression,
    time_step: Number | ParameterExpression
) -> QuantumCircuit:
    """Mass term for the full lattice."""
    circuit, _ = make_circuit(num_sites, 'i', 'o')
    for isite in range(num_sites):
        site_circ = mass_term_site(isite % 2, mass_mu, time_step)
        embed_site_circuit(circuit, site_circ, isite)

    return circuit


def electric_12_term_site(
    time_step: Number | ParameterExpression,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1
) -> tuple[QuantumCircuit, QubitPlacement, QubitPlacement]:
    r"""Local first two electric terms.

    First two electric Hamiltonian terms are
    $$
    \begin{align}
    H_{E}^{(1)} & = \frac{1}{2}n_{l}(r)\,,
    \\
    H_{E}^{(2)} & = \frac{1}{4}n_{l}(r)^{2}\,,
    \end{align}
    $$
    """
    gl_states = physical_states(max_left_flux=max_left_flux, max_right_flux=max_right_flux,
                                as_multi=True)
    lvals = np.unique(gl_states[:, 2])
    if lvals.shape[0] == 1:
        # No n_l DOF -> H_E(1) and (2) are constants
        return QuantumCircuit(0)
    if np.amax(lvals) == 1:
        circuit, regs = make_circuit(1, 'l0')
        # H_E^(1) + H_E^(2)
        circuit.p(-0.75 * time_step, regs['l0'][0])
        return circuit

    circuit, regs = make_circuit(1, *[f'l{i}' for i in range(BOSONIC_QUBITS)])
    # H_E^(1) + H_E^(2) = 1/2 n_l + 1/4 n_l^2
    if BOSON_TRUNC == 3 and BOSONIC_QUBITS == 2:
        # Optimization specific for BOSON_TRUNC=3: there should never be a state (1, 1), so we
        # can just make this a sum of local Zs
        circuit.p(-0.75 * time_step, regs['l0'][0])
        circuit.p(-2. * time_step, regs['l1'][0])
    else:
        # Otherwise construct a parity network circuit
        coeffs = np.zeros(2 ** BOSONIC_QUBITS)
        nl = np.arange(BOSON_TRUNC)
        coeffs[:BOSON_TRUNC] = 0.5 * nl + 0.25 * np.square(nl)
        coeffs = coeffs.reshape((2,) * BOSONIC_QUBITS)
        circuit.compose(parity_network(diag_to_iz(coeffs) * time_step), inplace=True)

    return circuit


def electric_12_term(
    num_sites: int,
    time_step: Number | ParameterExpression,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1
) -> QuantumCircuit:
    """First two electric Hamiltonian for the full lattice.

    Note that the electric Hamiltonian has terms only for sites [0, N-2].
    """
    conditions = boundary_conditions(num_sites, max_left_flux=max_left_flux,
                                     max_right_flux=max_right_flux)
    circuit, _ = make_circuit(num_sites, *[f'l{i}' for i in range(BOSONIC_QUBITS)])
    for isite in range(num_sites - 1):
        site_circ = electric_12_term_site(time_step, **conditions[isite])
        embed_site_circuit(circuit, site_circ, isite)

    return circuit


def electric_3f_term_site(
    time_step: Number | ParameterExpression
) -> tuple[QuantumCircuit, QubitPlacement, QubitPlacement]:
    r"""Fermionic part of the local electric (3) Hamiltonian.

    $$
    H_{E,f}^{(3)} = \frac{3}{4} (1 - n_{i}(r)) n_{o}(r)
    $$
    """
    # 3/4 * (1 - n_i) * n_o
    circuit, regs = make_circuit(1, 'i', 'o')
    circuit.x(regs['i'][0])
    circuit.cp(-0.75 * time_step, regs['i'][0], regs['o'][0])
    circuit.x(regs['i'][0])
    return circuit


def electric_3f_term(
    num_sites: int,
    time_step: Number | ParameterExpression
) -> QuantumCircuit:
    """Fermionic part of the electric (3) Hamiltonian for the full lattice.

    Note that the electric Hamiltonian has terms only for sites [0, N-2].
    """
    circuit, _ = make_circuit(num_sites - 1, 'i', 'o')
    site_circ = electric_3f_term_site(time_step)
    for isite in range(num_sites - 1):
        embed_site_circuit(circuit, site_circ, isite)
    return circuit


def electric_3b_term_site(
    time_step: Number | ParameterExpression,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1
) -> tuple[QuantumCircuit, QubitPlacement, QubitPlacement]:
    r"""Fermion-boson part of the local electric (3) Hamiltonian.

    $$
    H_{E,b}^{(3)} = \frac{1}{2} n_{l}(r) n_{o}(r) (1 - n_{i}(r))
    $$
    """
    gl_states = physical_states(max_left_flux=max_left_flux, max_right_flux=max_right_flux,
                                as_multi=True)
    lvals = np.unique(gl_states[:, 2])
    if lvals.shape[0] == 1:
        if lvals[0] == 0:
            return QuantumCircuit(0)

        circuit, regs = make_circuit(1, 'i', 'o')
        circuit.x(regs['i'][0])
        circuit.cp(-0.5 * lvals[0] * time_step, regs['i'][0], regs['o'][1])
        circuit.x(regs['i'][0])

    elif np.amax(lvals) == 1:
        circuit, regs = make_circuit(1, 'i', 'o', 'l0')
        coeffs = np.zeros((2, 2, 2))
        coeffs[1, 1, 0] = 0.5
        coeffs = diag_to_iz(coeffs)
        circuit.compose(parity_network(coeffs * time_step), inplace=True)

    else:
        circuit, regs = make_circuit(1)
        circuit.x(regs['i'][0])
        circuit.append(RCCXGate(), [regs['i'][0], regs['o'][0], regs['a'][0]])
        # 1/2 * n_l as a (2,) * nq + (2,) array
        coeffs = np.zeros((2 ** BOSONIC_QUBITS, 2))
        coeffs[:BOSON_TRUNC, 1] = 0.5 * np.arange(BOSON_TRUNC)
        coeffs = coeffs.reshape((2,) * (BOSONIC_QUBITS + 1))
        circuit.compose(parity_network(diag_to_iz(coeffs) * time_step),
                        qubits=[regs['a'][0]] + [regs[f'l{i}'][0] for i in range(BOSONIC_QUBITS)],
                        inplace=True)
        circuit.append(RCCXGate(), [regs['i'][0], regs['o'][0], regs['a'][0]])
        circuit.x(regs['i'][0])

    return circuit


def electric_3b_term(
    num_sites: int,
    time_step: Number | ParameterExpression,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1
) -> QuantumCircuit:
    """Fermion-boson part of the electric (3) Hamiltonian for the full lattice.

    Note that the electric Hamiltonian has terms only for sites [0, N-2].
    """
    conditions = boundary_conditions(num_sites, max_left_flux=max_left_flux,
                                     max_right_flux=max_right_flux)

    circuit, _ = make_circuit(num_sites - 1)
    for isite in range(num_sites - 1):
        site_circ = electric_3b_term_site(time_step, **conditions[isite])
        embed_site_circuit(circuit, site_circ, isite)

    return circuit


@dataclass
class HoppingTermConfig:
    """Parameters that determine the structure of the hopping term Hamiltonian circuit."""
    qubits: dict[str, Qubit]
    boson_ops: dict[str, tuple[str, str]]
    gl_states: np.ndarray
    indices: dict[str, int]


def hopping_term_config(
    term_type: int,
    max_left_flux: int,
    max_right_flux: int
) -> HoppingTermConfig:
    """Return the x, y, p, q indices and simplifications.

    Lattice boundary conditions can impose certain constraints on the available LSH flux from the
    left or the right. Certain simplifications to U_SVD are possible depending on the resulting
    reductions of the matrix elements.

    To check for simplifications, we first identify the physical states (satisfying the Gauss's law
    and the boundary conditions) with non-coincident n_i(r) and n_i(r+1). Then, within the list of
    such states,

    - If n_l(r) is 0 or 1 with n_o(r)=0, n_l(r)=1 occurring only when n_i(r+1)=1, the decrementer is
      replaced with X, and the o(r)-l(r) projector is replaced with [|0)(0|_l(r)]^{1-n_o(r)}.
    - Similarly, if n_l(r+1) is 0 or 1 with n_o(r+1)=1, n_l(r+1)=1 occurring only when n_i(r+1)=1,
      the decrementer is replaced with X, and the o(r+1)-l(r+1) projector is replaced with
      [|0)(0|_l(r+1)]^{n_o(r+1)}.
    - If n_o(r)=0, n_l(r)=Lambda never happens in the diagonal term, the projector is replaced with
      identity.
    - If n_o(r+1)=1, n_l(r+1)=Lambda never happens in the diagonal term, the projector is replaced
      with identity.
    - Excluding states with (n_i, n_o, n_l)_{r+1}=(1, 1, 0) or (0, 0, Lambda), if n_o(r) is
      identically 1, the n_i(r+1)-n_o(r) controlled decrementer and the corresponding projector are
      replaced with identity.
    - Similarly, excluding states with (n_i(r+1), n_o(r), n_l(r))=(1, 0, 0) or (0, 1, Lambda), if
      n_o(r+1) is identically 0, the n_i(r+1)-n_o(r+1) controlled decrementer and the corresponding
      projector are replaced with identity.
    """
    regs = {name: QuantumRegister(2, name)
            for name in ['i', 'o', 'a'] + [f'l{i}' for i in range(BOSONIC_QUBITS)]}
    if term_type == 1:
        qubits = {
            "x'": regs['i'][0],
            "x": regs['o'][0],
            "y'": regs['i'][1],
            "y": regs['o'][1],
            "p": regs['a'][0],
            "q": regs['a'][1]
        }
        qubits |= {f"pd{i}": regs[f'l{i}'][0] for i in range(BOSONIC_QUBITS)}
        qubits |= {f"qd{i}": regs[f'l{i}'][1] for i in range(BOSONIC_QUBITS)}

        indices = {"x'": 0, "x": 1, "y'": 3, "y": 4, "p": 2, "q": 5}
    else:
        qubits = {
            "x'": regs['o'][1],
            "x": regs['i'][1],
            "y'": regs['o'][0],
            "y": regs['i'][0],
            "p": regs['a'][1],
            "q": regs['a'][0]
        }
        qubits |= {f"pd{i}": regs[f'l{i}'][1] for i in range(BOSONIC_QUBITS)}
        qubits |= {f"qd{i}": regs[f'l{i}'][0] for i in range(BOSONIC_QUBITS)}

        indices = {"x'": 4, "x": 3, "y'": 1, "y": 0, "p": 5, "q": 2}

    gl_states = physical_states(max_left_flux=max_left_flux, max_right_flux=max_right_flux,
                                num_sites=2, as_multi=True)
    states = gl_states[gl_states[:, indices["x'"]] != gl_states[:, indices["y'"]]]
    states_preproj = states

    def occnum(label, filt=None):
        if filt is None:
            return states[:, indices[label]]
        return states[filt, indices[label]]

    # Default: decrement p and q by lambda and project |Lambda) out
    controls = {'p': ('x', 0), 'q': ('y', 1)}
    # Check for simplifications following the projection on the other side
    ops_candidates = []
    for pboson in ['p', 'q']:
        states = states_preproj.copy()
        tboson = 'q' if pboson == 'p' else 'p'
        pfermion, pval = controls[pboson]
        tfermion, tval = controls[tboson]

        cyp = occnum("y'") == 1
        cf = occnum(pfermion) == pval
        if np.all(occnum(pboson, cf) < 2) and np.all(occnum(pboson, ~cyp & cf) == 0):
            # Special case of l=0,1
            ops = {pboson: ('X', 'zero')}
        else:
            ops = {pboson: ('lambda', 'Lambda')}

        mask = ~((cyp & cf & (occnum(pboson) == 0))
                 | (~cyp & cf & (occnum(pboson) == BOSON_TRUNC - 1)))
        if np.all(mask):
            ops[pboson] = (ops[pboson][0], 'id')
        else:
            states = states[mask]

        cyp = occnum("y'") == 1
        cf = occnum(tfermion) == tval
        if not np.any(cf):
            ops[tboson] = ('id', 'id')
            ops_candidates.append(ops)
            continue

        if np.all(occnum(tboson, cf) < 2) and np.all(occnum(tboson, ~cyp & cf) == 0):
            # Special case of l=0,1
            ops[tboson] = ('X', 'zero')
        else:
            ops[tboson] = ('lambda', 'Lambda')

        mask = ~((cyp & cf & (occnum(tboson) == 0))
                 | (~cyp & cf & (occnum(tboson) == BOSON_TRUNC - 1)))
        if np.all(mask):
            ops[tboson] = (ops[tboson][0], 'id')

        ops_candidates.append(ops)

    # Evaluate and select the better site to project on according to the following score matrix:
    #                test
    #  |    |id-id X-id λ-id X-0 λ-Λ
    # p|X-id|    0    2    3   6   9
    # r|λ-id|    1    3    4   7  10
    # o|X-0 |    5    6    7  11  12
    # j|λ-Λ |    8    9   10  12  13
    op_combs = [('id', 'id'), ('X', 'id'), ('lambda', 'id'), ('X', 'zero'), ('lambda', 'Lambda')]
    proj_keys = {c: i for i, c in enumerate(op_combs[1:])}
    test_keys = {c: i for i, c in enumerate(op_combs)}
    scores = [
        [0, 2, 3, 6, 9],
        [1, 3, 4, 7, 10],
        [5, 6, 7, 11, 12],
        [8, 9, 10, 12, 13]
    ]
    s0 = scores[proj_keys[ops_candidates[0]['p']]][test_keys[ops_candidates[0]['q']]]
    s1 = scores[proj_keys[ops_candidates[1]['q']]][test_keys[ops_candidates[1]['p']]]
    if s0 <= s1:
        boson_ops = ops_candidates[0]
    else:
        boson_ops = ops_candidates[1]

    for boson in ['p', 'q']:
        if boson_ops[boson][0] != 'lambda':
            if boson_ops[boson][0] == 'id':
                qubits.pop(boson)
            else:
                qubits[boson] = qubits[f"{boson}d0"]
            for il in range(BOSONIC_QUBITS):
                qubits.pop(f"{boson}d{il}")

    return HoppingTermConfig(qubits, boson_ops, gl_states, indices)


def hopping_term_site(
    term_type: int,
    parity: int,
    interaction_x: Number | ParameterExpression,
    time_step: Number | ParameterExpression,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    with_barrier: bool = False
):
    """Local hopping term."""
    config = hopping_term_config(term_type, max_left_flux, max_right_flux)

    usvd_circuit = hopping_usvd(term_type, parity, config=config)
    diag_circuit = hopping_diagonal_term(term_type, parity, interaction_x, time_step, config=config)

    circuit, _ = make_circuit(2)
    circuit.compose(usvd_circuit, inplace=True)
    if with_barrier:
        circuit.barrier()
    circuit.compose(diag_circuit, inplace=True)
    if with_barrier:
        circuit.barrier()
    circuit.compose(usvd_circuit.inverse(), inplace=True)

    return circuit


def hopping_term(
    num_sites: int,
    site_parity: int,
    term_type: int,
    interaction_x: Number | ParameterExpression,
    time_step: Number | ParameterExpression,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    with_barrier: bool = False
):
    """Full lattice hopping term for a given site parity (0 or 1) and term type (1 or 2)."""
    conditions = boundary_conditions(num_sites, max_left_flux=max_left_flux,
                                     max_right_flux=max_right_flux, num_local=2)
    circuit, _ = make_circuit(num_sites)
    for isite in range(site_parity, num_sites - 1, 2):
        site_circ = hopping_term_site(term_type, isite, interaction_x, time_step,
                                      with_barrier=with_barrier, **conditions[isite])
        embed_site_circuit(circuit, site_circ, [isite, isite + 1])

    return circuit


def hopping_usvd(
    term_type: int,
    parity: int,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    config: Optional[HoppingTermConfig] = None
):
    if config is None:
        config = hopping_term_config(term_type, max_left_flux, max_right_flux)

    circuit, _ = make_circuit(2)

    if (term_type, parity) in [(1, 0), (2, 1)]:
        # x' as the SVD key qubit
        circuit.x(config.qubits["x'"])
        circuit.x(config.qubits["x"])
        if config.boson_ops['p'][0] == 'X':
            circuit.ccx(config.qubits["x'"], config.qubits["x"], config.qubits["p"])
        elif config.boson_ops['p'][0] == 'lambda':
            _hopping_usvd_decrementer(circuit, config.qubits, ["x'", "x"], "p")
        circuit.x(config.qubits["x"])
        circuit.cx(config.qubits["x'"], config.qubits["y'"])
        if config.boson_ops['q'][0] == 'X':
            circuit.ccx(config.qubits["x'"], config.qubits["y"], config.qubits["q"])
        elif config.boson_ops['q'][0] == 'lambda':
            _hopping_usvd_decrementer(circuit, config.qubits, ["x'", "y"], "q")
        circuit.x(config.qubits["x'"])
        circuit.h(config.qubits["x'"])
    else:
        # y' as the SVD key qubit
        if config.boson_ops['q'][0] == 'X':
            circuit.ccx(config.qubits["y'"], config.qubits["y"], config.qubits["q"])
        elif config.boson_ops['q'][0] == 'lambda':
            _hopping_usvd_decrementer(circuit, config.qubits, ["y'", "y"], "q")
        circuit.cx(config.qubits["y'"], config.qubits["x'"])
        circuit.x(config.qubits["x"])
        if config.boson_ops['p'][0] == 'X':
            circuit.ccx(config.qubits["y'"], config.qubits["x"], config.qubits["p"])
        elif config.boson_ops['q'][0] == 'lambda':
            _hopping_usvd_decrementer(circuit, config.qubits, ["y'", "x"], "p")
        circuit.x(config.qubits["x"])
        circuit.h(config.qubits["y'"])

    return circuit


def _hopping_usvd_decrementer(circuit, qubits, fermions, boson):
    circuit.append(RCCXGate(), [qubits[q] for q in fermions + [boson]])
    if BOSON_TRUNC == 2 ** BOSONIC_QUBITS:
        raise NotImplementedError('QFT-based decrementer')
    elif BOSON_TRUNC == 3 and BOSONIC_QUBITS == 2:
        circuit.cx(qubits[boson], qubits[f'{boson}d0'])
        circuit.ccx(qubits[boson], qubits[f'{boson}d0'], qubits[f'{boson}d1'])
        circuit.ccx(qubits[boson], qubits[f'{boson}d1'], qubits[f'{boson}d0'])
    else:
        raise NotImplementedError('No decrementer circuit for'
                                  f' BOSON_TRUNC={BOSON_TRUNC} and'
                                  f' BOSONIC_QUBITS={BOSONIC_QUBITS}')
    circuit.append(RCCXGate(), [qubits[q] for q in fermions + [boson]])


def hopping_diagonal_op(
    term_type: int,
    parity: int,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    config: Optional[HoppingTermConfig] = None
):
    """Compute the diagonal term for the given term type and site number as a sum of Pauli products.
    """
    if config is None:
        config = hopping_term_config(term_type, max_left_flux, max_right_flux)

    # Pass the Gauss's law-satisfying states through the decrementers in Usvd
    states = config.gl_states.copy()
    if (term_type, parity) in [(1, 0), (2, 1)]:
        idx = config.indices["x'"]
        mask_fp = states[:, idx] == 0
        idx = config.indices["y'"]
        states[mask_fp, idx] *= -1
        states[mask_fp, idx] += 1
    else:
        idx = config.indices["y'"]
        mask_fp = states[:, idx] == 1
        idx = config.indices["x'"]
        states[mask_fp, idx] *= -1
        states[mask_fp, idx] += 1

    mask = mask_fp & (states[:, config.indices["y"]] == 1)
    idx = config.indices["q"]
    if config.boson_ops['q'][0] == 'lambda':
        states[mask, idx] -= 1
        states[mask, idx] %= BOSON_TRUNC
    elif config.boson_ops['q'][0] == 'X':
        states[mask, idx] *= -1
        states[mask, idx] += 1
    mask = mask_fp & (states[:, config.indices["x"]] == 0)
    idx = config.indices["p"]
    if config.boson_ops['p'][0] == 'lambda':
        states[mask, idx] -= 1
        states[mask, idx] %= BOSON_TRUNC
    elif config.boson_ops['p'][0] == 'X':
        states[mask, idx] *= -1
        states[mask, idx] += 1

    # Apply the projectors and construct the diagonal operator
    # Start with the diagonal function
    diag_fn = np.sqrt((np.arange(BOSON_TRUNC)[:, None, None] + np.arange(1, 3)[None, :, None])
                      / (np.arange(BOSON_TRUNC)[:, None, None] + np.arange(1, 3)[None, None, :]))
    diag_op = np.tile(diag_fn[..., None, None], (1, 1, 1, 2, 2))
    if config.boson_ops['p'][1] == 'id':
        # Diag op is expressed in terms of n_q
        axis_names = ['q']
    elif config.boson_ops['q'][1] == 'id':
        axis_names = ['p']
    elif (term_type, parity) == (1, 0):
        axis_names = ['q']
    elif (term_type, parity) == (2, 0):
        axis_names = ['p']
    else:
        axis_names = ['p']
    axis_names += ["x", "y", "x'", "y'"]

    if (term_type, parity) in [(1, 0), (2, 1)]:
        # Project onto y' == 0 and multiply Zx'
        states = states[states[:, config.indices["y'"]] == 0]
        diag_op *= np.expand_dims(np.array([1., 0.]), [0, 1, 2, 3])
        diag_op *= np.expand_dims(np.array([1., -1.]), [0, 1, 2, 4])
    else:
        # Project onto x' == 1 and multiply Zy'
        states = states[states[:, config.indices["x'"]] == 1]
        diag_op *= np.expand_dims(np.array([0., 1.]), [0, 1, 2, 4])
        diag_op *= np.expand_dims(np.array([1., -1.]), [0, 1, 2, 3])

    # Zx
    diag_op *= np.expand_dims(np.array([1., -1.]), [0, 2, 3, 4])

    for boson, fermion, cval in [('p', 'x', 0), ('q', 'y', 1)]:
        projector = config.boson_ops[boson][1]
        if projector == 'id':
            continue
        mask = states[:, config.indices[fermion]] == cval
        if projector == 'Lambda':
            mask &= states[:, config.indices[boson]] == BOSON_TRUNC - 1
        else:
            mask &= states[:, config.indices[boson]] != 0
        states = states[~mask]

        fdim = axis_names.index(fermion)
        diag_op = np.moveaxis(diag_op, fdim, 1)
        if projector == 'Lambda':
            diag_op[-1, cval] = 0.
        else:
            diag_op[1:, cval] = 0.
        diag_op = np.moveaxis(diag_op, 1, fdim)

    for fermion in ["x", "y"]:
        unique = np.unique(states[:, config.indices[fermion]])
        if unique.shape[0] == 1:
            fdim = axis_names.index(fermion)
            diag_op = np.moveaxis(diag_op, fdim, 0)[unique[0]]
            axis_names.pop(fdim)

    if config.boson_ops[axis_names[0]][0] != 'lambda':
        diag_op = diag_op[:2]

    return diag_op, axis_names


def hopping_diagonal_term(
    term_type: int,
    parity: int,
    interaction_x: Number | ParameterExpression,
    time_step: Number | ParameterExpression,
    max_left_flux: int = BOSON_TRUNC - 1,
    max_right_flux: int = BOSON_TRUNC - 1,
    config: Optional[HoppingTermConfig] = None
):
    """Construct the circuit for the diagonal term.

    Using the hopping term config for the given boundary condition, identify the qubits that appear
    in the circuit, compute the Z rotation angles, and compose the parity network circuit.
    """
    if config is None:
        config = hopping_term_config(term_type, max_left_flux, max_right_flux)
    diag_op, axis_names = hopping_diagonal_op(term_type, parity, max_left_flux=max_left_flux,
                                              max_right_flux=max_right_flux, config=config)
    # Multiply the diag_op with the physical parameters (can be Parameters)
    shape = diag_op.shape
    diag_op = np.array([interaction_x * time_step * c for c in diag_op.reshape(-1)])
    diag_op = diag_op.reshape(shape)

    circuit, _ = make_circuit(2)
    circ = parity_network(diag_to_iz(diag_op))
    qubit_mapping = [config.qubits[name] for name in axis_names[::-1]]
    circuit.compose(circ, qubits=qubit_mapping, inplace=True)

    return circuit
