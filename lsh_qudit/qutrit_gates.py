"""Qutrit gates."""
from collections.abc import Sequence
from enum import Enum, IntEnum, auto
from typing import Optional, Union
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Barrier, Gate, Parameter
from qiskit.circuit.library import ECRGate, HGate, RZGate, SXGate, XGate, ZGate, RCCXGate
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary as sel
# from qiskit.circuit.parameterexpression import ParameterValueType
#  For some reason Pylance fails to recognize imported ParameterValueType as a valid type alias so
#  I'm redefining it here
from qiskit.circuit.parameterexpression import ParameterExpression
ParameterValueType = Union[ParameterExpression, float]


class GateType(Enum):
    """Gate type descriptors."""
    PULSE = auto()
    VIRTUAL = auto()
    COMPOSITE = auto()


class QutritGate(Gate):
    """Qutrit gate base class."""
    gate_type = None

    def __init__(
        self,
        name: str,
        as_qutrit: tuple[bool, ...],
        params: list[ParameterValueType],
        label: Optional[str] = None
    ):
        super().__init__(name, len(as_qutrit), params, label=label)
        self.as_qutrit = as_qutrit


class QutritPulseGate(QutritGate):
    """Pulse gate base class."""
    gate_type = GateType.PULSE


class QutritVirtualGate(QutritGate):
    """Pulse gate base class."""
    gate_type = GateType.VIRTUAL


class QutritCompositeGate(QutritGate):
    """Pulse gate base class."""
    gate_type = GateType.COMPOSITE


# pylint: disable=too-few-public-methods
class StaticProperties:
    """Mixin for qutrit gates with properties defined at class level. Copies of the properties will
    be set on instances through __init__()."""
    def __init__(self, params, *args, label=None, **kwargs):
        if params is None:
            params = []
        else:
            params = list(params)
        # pylint: disable=no-member
        super().__init__(self.__class__.gate_name, self.__class__.as_qutrit, params, *args,
                         label=label, **kwargs)


class OneQutritPulseGate(StaticProperties, QutritPulseGate):
    """Single qutrit pulse gate base class."""
    as_qutrit = (True,)


class OneQutritVirtualGate(StaticProperties, QutritVirtualGate):
    """Single qutrit virtual gate base class."""
    as_qutrit = (True,)


class OneQutritCompositeGate(StaticProperties, QutritCompositeGate):
    """Single qutrit composite gate base class."""
    as_qutrit = (True,)


class QutritQubitCompositeGate(StaticProperties, QutritCompositeGate):
    """Qutrit-qubit composite gate base class."""
    as_qutrit = (True, False)


class QubitQutritCompositeGate(StaticProperties, QutritCompositeGate):
    """Qubit-qutrit composite gate base class."""
    as_qutrit = (False, True)


class X12Gate(OneQutritPulseGate):
    """The single-qubit X gate on EF subspace."""
    gate_name = 'x12'

    def __init__(self, label: Optional[str] = None):
        """Create new X12 gate."""
        super().__init__([], label=label)

    def _define(self):
        qr = QuantumRegister(1, 'q')
        qc = QuantumCircuit(qr, name=self.name)
        qc.append(U12Gate(np.pi, 0., np.pi), [qr[0]])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return X12Gate()

    def __eq__(self, other):
        return isinstance(other, X12Gate)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.array([
            [1., 0., 0.],
            [0., 0., 1.],
            [0., 1., 0.]
        ], dtype=dtype or complex)


class SX12Gate(OneQutritPulseGate):
    """The single-qubit Sqrt(X) gate on EF subspace."""
    gate_name = 'sx12'

    def __init__(self, label: Optional[str] = None):
        """Create new SX12 gate."""
        super().__init__([], label=label)

    def _define(self):
        qr = QuantumRegister(1, 'q')
        qc = QuantumCircuit(qr, name=self.name)
        qc.append(U12Gate(np.pi / 2., -np.pi / 2., np.pi / 2.), [qr[0]])
        qc.append(P0Gate(-np.pi / 4.), [qr[0]])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return SX12dgGate()

    def __eq__(self, other):
        return isinstance(other, SX12Gate)


class SX12dgGate(OneQutritPulseGate):
    """The single-qubit Sqrt(X) gate on EF subspace."""
    gate_name = 'sx12dg'

    def __init__(self, label: Optional[str] = None):
        """Create new SX12 gate."""
        super().__init__([], label=label)

    def _define(self):
        qr = QuantumRegister(1, 'q')
        qc = QuantumCircuit(qr, name=self.name)
        qc.append(U12Gate(np.pi / 2., np.pi / 2., -np.pi / 2.), [qr[0]])
        qc.append(P0Gate(np.pi / 4.), [qr[0]])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return SX12Gate()

    def __eq__(self, other):
        return isinstance(other, SX12dgGate)


class RZ12Gate(OneQutritVirtualGate):
    """The RZ gate on EF subspace."""
    gate_name = 'rz12'

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new RZ12 gate."""
        super().__init__([phi], label=label)

    def _define(self):
        qr = QuantumRegister(1, 'q')
        qc = QuantumCircuit(qr, name=self.name)
        qc.append(U12Gate(0., np.pi, 0.), [qr[0]])
        qc.append(P0Gate(np.pi / 2.), [qr[0]])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        r"""Return inverted U gate.

        :math:`U(\theta,\phi,\lambda)^{\dagger} =U(-\theta,-\lambda,-\phi)`)
        """
        return RZ12Gate(-self.params[0])

    def __eq__(self, other):
        return isinstance(other, RZ12Gate) and self._compare_parameters(other)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        phi = self.params[0]
        return np.diagflat([1., np.exp(-0.5j * phi), np.exp(0.5j * phi)]).astype(dtype or complex)


class SetF12Gate(OneQutritVirtualGate):
    """Set the qutrit frequency to a specific value."""
    gate_name = 'set_f12'

    def __init__(self, freq: ParameterValueType, label: Optional[str] = None):
        """Create new SetF12 gate."""
        super().__init__([freq], label=label)

    def __eq__(self, other):
        return isinstance(other, SetF12Gate) and self._compare_parameters(other)


class U12Gate(OneQutritCompositeGate):
    """U gate composed of SX12 and RZ12."""
    gate_name = 'u12'

    def __init__(
        self,
        theta: ParameterValueType,
        phi: ParameterValueType,
        lam: ParameterValueType,
        label: Optional[str] = None
    ):
        super().__init__([theta, phi, lam], label=label)

    def inverse(self, annotated: bool = False):
        r"""Return inverted U gate.

        :math:`U(\theta,\phi,\lambda)^{\dagger} =U(-\theta,-\lambda,-\phi)`)
        """
        return U12Gate(-self.params[0], -self.params[2], -self.params[1])

    def __eq__(self, other):
        return isinstance(other, U12Gate) and self._compare_parameters(other)


class XplusGate(OneQutritCompositeGate):
    """X+ gate."""
    gate_name = 'xplus'

    def __init__(self, label: Optional[str] = None):
        super().__init__([], label=label)

    def _define(self):
        qc = QuantumCircuit(QuantumRegister(1, 'q'))
        qc.append(X12Gate(), [0])
        qc.x(0)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return XminusGate()

    def __eq__(self, other):
        return isinstance(other, XplusGate)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.array([
            [0., 0., 1.],
            [1., 0., 0.],
            [0., 1., 0.]
        ], dtype=dtype or complex)


class XminusGate(OneQutritCompositeGate):
    """X- gate."""
    gate_name = 'xminus'

    def __init__(self, label: Optional[str] = None):
        super().__init__([], label=label)

    def _define(self):
        qc = QuantumCircuit(QuantumRegister(1, 'q'))
        qc.x(0)
        qc.append(X12Gate(), [0])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return XplusGate()

    def __eq__(self, other):
        return isinstance(other, XminusGate)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.array([
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 0.]
        ], dtype=dtype or complex)


class X02Gate(OneQutritCompositeGate):
    """X02 gate."""
    gate_name = 'x02'

    def __init__(self, label: Optional[str] = None):
        super().__init__([], label=label)

    def _define(self):
        qc = QuantumCircuit(QuantumRegister(1, 'q'))
        qc.x(0)
        qc.append(X12Gate(), [0])
        qc.x(0)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return X02Gate()

    def __eq__(self, other):
        return isinstance(other, X02Gate)


class P0Gate(OneQutritCompositeGate):
    """P0 gate."""
    gate_name = 'p0'

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new P0 gate."""
        super().__init__([phi], label=label)

    def inverse(self, annotated: bool = False):
        return P0Gate(-self.params[0])

    def __eq__(self, other):
        return isinstance(other, P0Gate) and self._compare_parameters(other)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        phi = self.params[0]
        return np.diagflat([np.exp(1.j * phi), 1., 1.]).astype(dtype or complex)


class P1Gate(OneQutritCompositeGate):
    """P1 gate."""
    gate_name = 'p1'

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new P1 gate."""
        super().__init__([phi], label=label)

    def _define(self):
        phi = self.params[0]
        qc = QuantumCircuit(QuantumRegister(1, 'q'))
        qc.rz(2. * phi / 3., 0)
        qc.append(RZ12Gate(-2. * phi / 3.), [0])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return P1Gate(-self.params[0])

    def __eq__(self, other):
        return isinstance(other, P1Gate) and self._compare_parameters(other)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        phi = self.params[0]
        return np.diagflat([1., np.exp(1.j * phi), 1.]).astype(dtype or complex)


class P2Gate(OneQutritCompositeGate):
    """P2 gate."""
    gate_name = 'p2'

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new P2 gate."""
        super().__init__([phi], label=label)

    def _define(self):
        phi = self.params[0]
        qc = QuantumCircuit(QuantumRegister(1, 'q'))
        qc.rz(2. * phi / 3., 0)
        qc.append(RZ12Gate(4. * phi / 3.), [0])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return P2Gate(-self.params[0])

    def __eq__(self, other):
        return isinstance(other, P2Gate) and self._compare_parameters(other)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        phi = self.params[0]
        return np.diagflat([1., 1., np.exp(1.j * phi)]).astype(dtype or complex)


class QGate(OneQutritCompositeGate):
    """Q gate."""
    gate_name = 'q'

    def __init__(self, phi: ParameterValueType, label: Optional[str] = None):
        """Create new Q gate."""
        super().__init__([phi], label=label)

    def _define(self):
        phi = self.params[0]
        qc = QuantumCircuit(QuantumRegister(1, 'q'))
        qc.rz(2 * phi, 0)
        qc.append(RZ12Gate(2 * phi), [0])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return QGate(-self.params[0])

    def __eq__(self, other):
        return isinstance(other, QGate) and self._compare_parameters(other)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        phi = self.params[0]
        return np.diagflat(np.exp(1.j * phi * np.arange(3))).astype(dtype or complex)


class CrossResonanceGate(Gate):
    """CR gate with a control qutrit and target qubit."""
    gate_name = 'cr'
    gate_type = GateType.PULSE

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        block_unitaries: Optional[np.ndarray] = None,
        label: Optional[str] = None
    ):
        if params is None:
            params = []
        else:
            params = list(params)
        super().__init__(self.gate_name, 2, params, label=label)
        self.block_unitaries = block_unitaries


class CrossResonancePlusGate(CrossResonanceGate):
    """CR+ gate."""
    gate_name = 'crp'


class CrossResonanceMinusGate(CrossResonanceGate):
    """CR- gate."""
    gate_name = 'crm'


class QutritQubitCXType(IntEnum):
    """Type of qutrit-qubit generalized CX gate."""
    X = 2
    X12 = 0
    REVERSE = -1
    CRCR = -2


class RCRGate(Gate):
    """Repeated cross resonance gate."""
    gate_name = 'rcr'
    gate_type = GateType.COMPOSITE

    def __init__(
        self,
        params: list[ParameterValueType],
        label: Optional[str] = None
    ):
        super().__init__('rcr', 2, params, label=label)


class CRCRGate(QutritQubitCompositeGate):
    """Cycled RCR gate.

    CRCR angles:
    Type X(2) -> 2 * (θ_0 + θ_1 - 2*θ_2)
    Type X12(0) -> 2 * (θ_1 + θ_2 - 2*θ_0)
    """
    gate_name = 'crcr'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params, label=label)


class QutritQubitCXGate(QutritQubitCompositeGate):
    """CX gate with a control qutrit and target qubit."""
    gate_name = 'qutrit_qubit_cx'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params, label=label)


class QutritQubitCZGate(QutritQubitCompositeGate):
    """CZ gate with a control qutrit and target qubit."""
    gate_name = 'qutrit_qubit_cz'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(params, label=label)


class QutritMCGate(QutritCompositeGate):
    """Multi-controlled gate using qutrits."""
    def __init__(
        self,
        name: str,
        num_controls: int,
        target_gate: type[Gate],
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        if params is None:
            params = []
        else:
            params = list(params)
        as_qutrit = (False,) + (True,) * (num_controls - 1) + (False,)
        super().__init__(name, as_qutrit, params, label=label)
        self.target_gate = target_gate


class QutritCCXGate(QutritMCGate):
    """Toffoli gate using qutrits."""
    gate_name = 'qutrit_toffoli'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(self.gate_name, 2, XGate, params=params, label=label)


class QutritCCZGate(QutritMCGate):
    """CCZ gate using qutrits."""
    gate_name = 'qutrit_ccz'

    def __init__(
        self,
        params: Optional[Sequence[ParameterValueType]] = None,
        label: Optional[str] = None
    ):
        super().__init__(self.gate_name, 2, ZGate, params=params, label=label)


class QubitQutritCRxPlusPiGate(QubitQutritCompositeGate):
    """Controled Rx01 gate."""
    gate_name = 'qubit_qutrit_crx_pluspi'

    def __init__(
        self,
        label: Optional[str] = None
    ):
        super().__init__([], label=label)

    def inverse(self, annotated: bool = False):
        return QubitQutritCRxMinusPiGate()

    def __eq__(self, other):
        return isinstance(other, QubitQutritCRxPlusPiGate)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 0., 0., -1.j, 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., -1.j, 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.]
        ], dtype=dtype or complex)


class QubitQutritCRxMinusPiGate(QubitQutritCompositeGate):
    """Controled Rx01 gate."""
    gate_name = 'qubit_qutrit_crx_minuspi'

    def __init__(
        self,
        label: Optional[str] = None
    ):
        super().__init__([], label=label)

    def inverse(self, annotated: bool = False):
        return QubitQutritCRxPlusPiGate()

    def __eq__(self, other):
        return isinstance(other, QubitQutritCRxMinusPiGate)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        return np.array([
            [1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1.j, 0., 0.],
            [0., 0., 1., 0., 0., 0.],
            [0., 1.j, 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 1.]
        ], dtype=dtype or complex)


class CQGate(QubitQutritCompositeGate):
    """Controled Q gate."""
    gate_name = 'cq'

    def __init__(
        self,
        phi: ParameterValueType,
        label: Optional[str] = None
    ):
        super().__init__([phi], label=label)

    def _define(self):
        phi = self.params[0]
        qr = QuantumRegister(2, 'q')
        qc = QuantumCircuit(qr, name=self.name)
        qc.append(X12Gate(), [1])
        qc.append(QubitQutritCRxPlusPiGate(), [0, 1])
        qc.append(X12Gate(), [1])
        qc.append(QGate(-0.5 * phi), [1])
        qc.append(X12Gate(), [1])
        qc.append(QubitQutritCRxMinusPiGate(), [0, 1])
        qc.append(X12Gate(), [1])
        qc.append(QGate(0.5 * phi), [1])
        qc.rz(phi, 0)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return CQGate(-self.params[0])

    def __eq__(self, other):
        return isinstance(other, CQGate) and self._compare_parameters(other)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        phi = self.params[0]
        return np.diagflat(
                [1., 1., 1., np.exp(1.j * phi), 1., np.exp(2.j * phi)]
            ).astype(dtype or complex)


class CXminusGate(QubitQutritCompositeGate):
    """Controlled X- gate."""
    gate_name = 'cxminus'

    def __init__(
        self,
        label: Optional[str] = None
    ):
        super().__init__([], label=label)

    def _define(self):
        qr = QuantumRegister(2, 'q')
        qc = QuantumCircuit(qr, name=self.name)
        qc.append(QubitQutritCRxMinusPiGate(), [0, 1])
        qc.append(XminusGate(), [1])
        qc.append(QubitQutritCRxMinusPiGate(), [0, 1])
        qc.append(XplusGate(), [1])
        qc.rz(-0.5 * np.pi, 0)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return CXplusGate()

    def __eq__(self, other):
        return isinstance(other, CXminusGate)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        arr = np.diagflat([1., 1., 1., 0., 0., 0.]).astype(dtype or complex)
        arr[[1, 3], [3, 5]] = 1.
        arr[5, 1] = 1.j
        return arr


class CXplusGate(QubitQutritCompositeGate):
    """Controlled X+ gate."""
    gate_name = 'cxplus'

    def __init__(
        self,
        label: Optional[str] = None
    ):
        super().__init__([], label=label)

    def _define(self):
        qr = QuantumRegister(2, 'q')
        qc = QuantumCircuit(qr, name=self.name)
        qc.rz(0.5 * np.pi, 0)
        qc.append(XminusGate(), [1])
        qc.append(QubitQutritCRxPlusPiGate(), [0, 1])
        qc.append(XplusGate(), [1])
        qc.append(QubitQutritCRxPlusPiGate(), [0, 1])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return CXminusGate()

    def __eq__(self, other):
        return isinstance(other, CXplusGate)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("unable to avoid copy while creating an array as requested")
        arr = np.diagflat([1., 1., 1., 0., 0., 0.]).astype(dtype or complex)
        arr[[3, 5], [1, 3]] = 1.
        arr[1, 5] = -1.j
        return arr


class CCXminusGate(Gate):
    """Doubly controlled Xminus gate using an ancilla qubit."""
    def __init__(self):
        super().__init__('ccxminus', 4, [])

    def _define(self):
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr, name=self.name)
        qc.append(RCCXGate(), [0, 1, 2])
        qc.append(CXminusGate(), [2, 3])
        qc.append(RCCXGate(), [0, 1, 2])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return CCXplusGate()

    def __eq__(self, other):
        return isinstance(other, CCXminusGate)


class CCXplusGate(Gate):
    """Doubly controlled Xminus gate using an ancilla qubit."""
    def __init__(self):
        super().__init__('ccxplus', 4, [])

    def _define(self):
        qr = QuantumRegister(4)
        qc = QuantumCircuit(qr, name=self.name)
        qc.append(RCCXGate(), [0, 1, 2])
        qc.append(CXplusGate(), [2, 3])
        qc.append(RCCXGate(), [0, 1, 2])
        self.definition = qc

    def inverse(self, annotated: bool = False):
        return CCXminusGate()

    def __eq__(self, other):
        return isinstance(other, CCXplusGate)


q = QuantumRegister(1, 'q')
_theta = Parameter('theta')
_phi = Parameter('phi')
_lam = Parameter('lam')
qasm_def = QuantumCircuit(q, global_phase=(_lam + _phi - np.pi) / 2)
qasm_def.append(RZ12Gate(_lam), [0])
qasm_def.append(SX12Gate(), [0])
qasm_def.append(RZ12Gate(_theta + np.pi), [0])
qasm_def.append(SX12Gate(), [0])
qasm_def.append(RZ12Gate(_phi + 3 * np.pi), [0])
qasm_def.append(P0Gate((_lam + _phi - np.pi) / 2.), [0])
sel.add_equivalence(U12Gate(_theta, _phi, _lam), qasm_def)

q = QuantumRegister(1, 'q')
qasm_def = QuantumCircuit(q)
qasm_def.append(X12Gate(), [0])
qasm_def.x(0)
sel.add_equivalence(XplusGate(), qasm_def)

q = QuantumRegister(1, 'q')
qasm_def = QuantumCircuit(q)
qasm_def.x(0)
qasm_def.append(X12Gate(), [0])
sel.add_equivalence(XminusGate(), qasm_def)

q = QuantumRegister(1, 'q')
qasm_def = QuantumCircuit(q)
qasm_def.x(0)
qasm_def.append(X12Gate(), [0])
qasm_def.x(0)
sel.add_equivalence(X02Gate(), qasm_def)

q = QuantumRegister(1, 'q')
qasm_def = QuantumCircuit(q)
qasm_def.append(X12Gate(), [0])
qasm_def.x(0)
qasm_def.append(X12Gate(), [0])
sel.add_equivalence(X02Gate(), qasm_def)

q = QuantumRegister(1, 'q')
_phi = Parameter('phi')
qasm_def = QuantumCircuit(q)
qasm_def.rz(-_phi * 4. / 3., 0)
qasm_def.append(RZ12Gate(-_phi * 2. / 3.), [0])
sel.add_equivalence(P0Gate(_phi), qasm_def)

q = QuantumRegister(1, 'q')
_phi = Parameter('phi')
qasm_def = QuantumCircuit(q)
qasm_def.rz(_phi * 2. / 3., 0)
qasm_def.append(RZ12Gate(-_phi * 2. / 3.), [0])
sel.add_equivalence(P1Gate(_phi), qasm_def)

q = QuantumRegister(1, 'q')
_phi = Parameter('phi')
qasm_def = QuantumCircuit(q)
qasm_def.rz(_phi * 2. / 3., 0)
qasm_def.append(RZ12Gate(_phi * 4. / 3.), [0])
sel.add_equivalence(P2Gate(_phi), qasm_def)

q = QuantumRegister(1, 'q')
_phi = Parameter('phi')
qasm_def = QuantumCircuit(q)
qasm_def.rz(_phi * 2., 0)
qasm_def.append(RZ12Gate(_phi * 2.), [0])
sel.add_equivalence(QGate(_phi), qasm_def)

# Compact version of reverse CX
q = QuantumRegister(2, 'q')
qasm_def = QuantumCircuit(q)
for gate, qargs in [
    (HGate(), [1]),
    (QutritQubitCZGate(), [0, 1]),
    (HGate(), [1])
]:
    qasm_def.append(gate, qargs)
sel.add_equivalence(QutritQubitCXGate(), qasm_def)

# Compact version of reverse CZ
q = QuantumRegister(2, 'q')
qasm_def = QuantumCircuit(q)
for gate, qargs in [
    (XplusGate(), [0]),
    (XGate(), [1]),
    (ECRGate(), [1, 0]),
    (XGate(), [1]),
    (ECRGate(), [1, 0]),
    (X12Gate(), [0]),
    (RZGate(np.pi / 3.), [0]),
    (RZ12Gate(-np.pi / 3.), [0]),
    (RZGate(np.pi), [1])
]:
    qasm_def.append(gate, qargs)
sel.add_equivalence(QutritQubitCZGate(), qasm_def)

q = QuantumRegister(3, 'q')
qasm_def = QuantumCircuit(q)
for gate, qargs in [
    (Barrier(3), q),
    (XminusGate(label='qutrit_toffoli_begin'), q[1:2]),
    (Barrier(2), q[:2]),
    (RZGate(-np.pi / 2.), q[:1]),
    (ECRGate(), q[:2]),
    (XGate(), q[:1]),
    (RZGate(-np.pi), q[1:2]),
    (SXGate(), q[1:2]),
    (RZGate(-np.pi), q[1:2]),
    (Barrier(3), q),
    (QutritQubitCXGate(), q[1:]),
    (Barrier(3), q),
    (XGate(), q[:1]),
    (ECRGate(), q[:2]),
    (RZGate(np.pi / 2.), q[:1]),
    (RZGate(-np.pi / 3.), q[1:2]),
    (RZ12Gate(-2. * np.pi / 3.), q[1:2]),
    (SXGate(), q[1:2]),
    (Barrier(2), q[:2]),
    (XplusGate(label='qutrit_toffoli_end'), q[1:2]),
    (Barrier(3), q)
]:
    qasm_def.append(gate, qargs)
sel.add_equivalence(QutritCCXGate(), qasm_def)

q = QuantumRegister(3, 'q')
qasm_def = QuantumCircuit(q)
for gate, qargs in [
    (Barrier(3), q),
    (XminusGate(label='qutrit_toffoli_begin'), q[1:2]),
    (Barrier(2), q[:2]),
    (RZGate(-np.pi / 2.), q[:1]),
    (ECRGate(), q[:2]),
    (XGate(), q[:1]),
    (RZGate(-np.pi), q[1:2]),
    (SXGate(), q[1:2]),
    (RZGate(-np.pi), q[1:2]),
    (Barrier(3), q),
    (QutritQubitCZGate(), q[1:]),
    (Barrier(3), q),
    (XGate(), q[:1]),
    (ECRGate(), q[:2]),
    (RZGate(np.pi / 2.), q[:1]),
    (RZGate(-np.pi / 3.), q[1:2]),
    (RZ12Gate(-2. * np.pi / 3.), q[1:2]),
    (SXGate(), q[1:2]),
    (Barrier(2), q[:2]),
    (XplusGate(label='qutrit_toffoli_end'), q[1:2]),
    (Barrier(3), q)
]:
    qasm_def.append(gate, qargs)
sel.add_equivalence(QutritCCZGate(), qasm_def)

q = QuantumRegister(2, 'q')
qasm_def = QuantumCircuit(q)
qasm_def.x(q[0])
qasm_def.ecr(q[0], q[1])
qasm_def.rx(np.pi / 2., q[1])
sel.add_equivalence(QubitQutritCRxPlusPiGate(), qasm_def)

q = QuantumRegister(2, 'q')
qasm_def = QuantumCircuit(q)
qasm_def.ecr(q[0], q[1])
qasm_def.x(q[0])
qasm_def.rx(-np.pi / 2., q[1])
sel.add_equivalence(QubitQutritCRxMinusPiGate(), qasm_def)
