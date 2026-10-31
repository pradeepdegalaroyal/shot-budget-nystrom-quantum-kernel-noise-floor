"""Statevector simulation of the Qiskit ``ZZFeatureMap`` (linear entanglement).

The map is ``reps`` repetitions of ``H^n`` followed by a diagonal phase
``phi(b) = sum_i 2 x_i b_i + sum_(i,i+1) 2 (pi - x_i)(pi - x_(i+1)) (b_i xor b_(i+1))``.
Because every layer after the Hadamards is diagonal, the state of a batch of
inputs is obtained with one dense Walsh-Hadamard product per repetition.
"""

from __future__ import annotations

import numpy as np

REPS = 2


def _bits(n: int) -> np.ndarray:
    """Return the (2**n, n) little-endian bit table (qubit 0 = least significant)."""
    idx = np.arange(2**n)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)


def _walsh_hadamard(n: int) -> np.ndarray:
    """Normalised n-qubit Hadamard matrix."""
    bits = _bits(n).astype(np.int64)
    idx = np.arange(2**n)
    parity = np.zeros((2**n, 2**n), dtype=np.int64)
    for q in range(n):
        parity ^= ((idx[:, None] >> q) & 1) & bits[None, :, q]
    return (1 - 2 * parity) / np.sqrt(2.0**n)


def zz_states(x: np.ndarray, reps: int = REPS) -> np.ndarray:
    """Statevectors of the ZZ feature map for a batch of inputs.

    Args:
        x: array of shape (N, n) of feature-map angles (one column per qubit).
        reps: number of feature-map repetitions.

    Returns:
        Complex array of shape (N, 2**n); row ``k`` is the state ``|psi(x_k)>``.
    """
    x = np.atleast_2d(np.asarray(x, dtype=float))
    n_pts, n = x.shape
    bits = _bits(n).astype(float)
    w = _walsh_hadamard(n)
    phase = 2.0 * x @ bits.T
    for i in range(n - 1):
        xor = np.abs(bits[:, i] - bits[:, i + 1])
        phase += 2.0 * ((np.pi - x[:, i]) * (np.pi - x[:, i + 1]))[:, None] * xor[None, :]
    diag = np.exp(1j * phase)
    psi = np.full((n_pts, 2**n), 1.0 / np.sqrt(2.0**n), dtype=complex)
    psi = psi * diag
    for _ in range(reps - 1):
        psi = (psi @ w) * diag
    return psi


def exact_kernel(states_a: np.ndarray, states_b: np.ndarray | None = None) -> np.ndarray:
    """Fidelity kernel ``k(x, y) = |<psi(x)|psi(y)>|^2`` between two state batches."""
    states_b = states_a if states_b is None else states_b
    overlap = states_a.conj() @ states_b.T
    return np.abs(overlap) ** 2
