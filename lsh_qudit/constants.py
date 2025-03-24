"""Generic constants."""
import numpy as np


BOSON_TRUNC = 3
BOSONIC_QUBITS = 2

sqrt2 = np.sqrt(2.)
sqrt3 = np.sqrt(3.)

pauliz = np.diagflat([1., -1.]).astype(np.complex128)
sigmaplus = np.array([[0., 1.], [0., 0.]], dtype=np.complex128)
sigmaminus = np.array([[0., 0.], [1., 0.]], dtype=np.complex128)
hadamard = np.array([[1., 1.], [1., -1.]]) / np.sqrt(2.)
cx = np.eye(4, dtype=np.complex128)
cx[2:, 2:] = np.array([[0., 1.], [1., 0.]])
ocx = np.eye(4, dtype=np.complex128)
ocx[:2, :2] = np.array([[0., 1.], [1., 0.]])

if BOSONIC_QUBITS == 'qutrit':
    BDIM = 3
else:
    BDIM = 2 ** BOSONIC_QUBITS
cyc_incr = np.zeros((BDIM, BDIM), dtype=np.complex128)
cyc_incr[[0, 1, 2], [2, 0, 1]] = 1.
diags = np.zeros(BDIM)
diags[:BOSON_TRUNC - 1] = 1.
incrp = cyc_incr @ np.diagflat(diags)
cincrp = np.zeros((2 * BDIM, 2 * BDIM), dtype=np.complex128)
eyep_diag = np.zeros(BDIM)
eyep_diag[:BOSON_TRUNC] = 1.
eyep = np.diagflat(eyep_diag)
cincrp[:BDIM, :BDIM] = eyep
cincrp[BDIM:, BDIM:] = incrp
ocincrp = np.zeros((2 * BDIM, 2 * BDIM), dtype=np.complex128)
ocincrp[:BDIM, :BDIM] = incrp
ocincrp[BDIM:, BDIM:] = eyep
ccdecr = np.eye(4 * BDIM, dtype=np.complex128)
ccdecr[3 * BDIM:, 3 * BDIM:] = cyc_incr.conjugate().T
cocdecr = np.eye(4 * BDIM, dtype=np.complex128)
cocdecr[2 * BDIM:3 * BDIM, 2 * BDIM:3 * BDIM] = cyc_incr.conjugate().T
occdecr = np.eye(4 * BDIM, dtype=np.complex128)
occdecr[BDIM:2 * BDIM, BDIM:2 * BDIM] = cyc_incr.conjugate().T
ococdecr = np.eye(4 * BDIM, dtype=np.complex128)
ococdecr[:BDIM, :BDIM] = cyc_incr.conjugate().T
