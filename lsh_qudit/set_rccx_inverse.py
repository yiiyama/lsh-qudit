"""Set the inverse method for RCCXGate."""
from qiskit.circuit.library import RCCXGate


def rccx_inverse(self, annotated=False):  # pylint: disable=unused-argument
    return RCCXGate()


RCCXGate.inverse = rccx_inverse
