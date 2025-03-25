"""Transpilation subroutine."""
from typing import Optional
from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler import PassManager, StagedPassManager, generate_preset_pass_manager
from qiskit.transpiler.passes.optimization import Collect2qBlocks, ConsolidateBlocks
from .layout import layout_heavy_hex
from .precompiler import lsh_qudit_precompiler
from .postcompiler import LSHPostcompiler
from .utils import QubitPlacement


def get_layout(
    circuit: QuantumCircuit | list[QuantumCircuit],
    qp: QubitPlacement,
    backend: Backend
) -> list[int]:
    """Precompile the circuit and find the best layout."""
    precompile_pm = lsh_qudit_precompiler()
    precompiled = precompile_pm.run(circuit)
    if isinstance(precompiled, list):
        precompiled = max(precompiled, key=lambda c: c.depth())
    return layout_heavy_hex(backend, precompiled, qp)


def transpile_lsh_circuit(
    circuit: QuantumCircuit | list[QuantumCircuit],
    backend: Backend,
    layout: Optional[list[int]] = None,
    qp: Optional[QubitPlacement] = None
) -> QuantumCircuit:
    """Precompile, layout, transpile, postcompile.

    Some passes in the preset PM conflicts with the postcompiler and must be removed.
    """
    if layout is None:
        layout = get_layout(circuit, qp, backend)

    preset_pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                             initial_layout=layout)
    stage_pms = {}
    for stage in preset_pm.expanded_stages:
        if (pm := getattr(preset_pm, stage, None)) is not None:
            stage_pms[stage] = pm
    stage_pms['precompile'] = lsh_qudit_precompiler()
    stage_pms['postcompile'] = PassManager([LSHPostcompiler()])

    idx = next(idx for idx, pass_set in enumerate(stage_pms['init']._tasks)
               if isinstance(pass_set[0], Collect2qBlocks))
    stage_pms['init'].remove(idx)
    idx = next(idx for idx, pass_set in enumerate(stage_pms['init']._tasks)
               if isinstance(pass_set[0], ConsolidateBlocks))
    stage_pms['init'].remove(idx)
    stage_pms['optimization']._tasks[1][0].tasks = stage_pms['optimization']._tasks[1][0].tasks[2:]

    stages = ('precompile',) + preset_pm.stages + ('postcompile',)
    pm = StagedPassManager(stages=stages, **stage_pms)

    return pm.run(circuit)
