"""Transpilation subroutine."""
from qiskit.circuit import QuantumCircuit
from qiskit.providers import Backend
from qiskit.transpiler import PassManager, StagedPassManager, generate_preset_pass_manager
from qiskit.transpiler.passes.optimization import Collect2qBlocks, ConsolidateBlocks
from .layout import layout_heavy_hex
from .precompiler import lsh_qudit_precompiler
from .postcompiler import LSHPostcompiler
from .utils import QubitPlacement


def transpile_lsh_circuit(
    circuit: QuantumCircuit,
    qp: QubitPlacement,
    backend: Backend
) -> QuantumCircuit:
    """Precompile, layout, transpile, postcompile.

    Some passes in the preset PM conflicts with the postcompiler and must be removed.
    """
    precompile_pm = lsh_qudit_precompiler()
    precompiled = precompile_pm.run(circuit)
    layout = layout_heavy_hex(backend, precompiled, qp)

    preset_pm = generate_preset_pass_manager(optimization_level=3, backend=backend,
                                             initial_layout=layout)
    postcompile_pm = PassManager([LSHPostcompiler()])

    stage_pms = {}
    for stage in preset_pm.expanded_stages:
        if (pm := getattr(preset_pm, stage, None)) is not None:
            stage_pms[stage] = pm
    stage_pms['precompile'] = precompile_pm
    stage_pms['postcompile'] = postcompile_pm

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
