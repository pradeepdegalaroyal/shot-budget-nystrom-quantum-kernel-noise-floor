"""Finite-shot and gate-noise model of a compute-uncompute kernel estimate.

A kernel entry ``k(x, y)`` is the probability of reading all zeros after
``U(y)^dagger U(x) |0>``. With ``G`` noisy two-qubit gates, each followed by a
two-qubit depolarising channel of probability ``p2``, the circuit is modelled as
a global depolarising channel of fidelity ``f = (1 - p2) ** G``:

    p_obs = f * k + (1 - f) / 2**n.

``s`` shots give a binomial count; the estimator inverts the affine map (a
calibrated rescaling), so noise costs shots through the factor ``1 / f**2``
in the variance rather than through bias.
"""

from __future__ import annotations

import numpy as np

from .hardware import two_qubit_gate_count


def depolarising_fidelity(n_qubits: int, p2: float) -> float:
    """Global survival factor ``(1 - p2) ** G`` of a compute-uncompute circuit."""
    return float((1.0 - p2) ** two_qubit_gate_count(n_qubits, compute_uncompute=True))


def sample_kernel_block(
    exact: np.ndarray,
    shots: int,
    n_qubits: int,
    p2: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw a shot-noisy, rescaled estimate of an exact kernel block.

    Args:
        exact: exact kernel entries (any shape).
        shots: shots per entry (>= 1).
        n_qubits: register size, sets the depolarising floor ``1 / 2**n``.
        p2: two-qubit depolarising probability per gate.
        rng: random generator.

    Returns:
        Array with the shape of ``exact``; unbiased for ``exact`` (not clipped).
    """
    f = depolarising_fidelity(n_qubits, p2)
    floor = (1.0 - f) / 2.0**n_qubits
    p_obs = np.clip(f * exact + floor, 0.0, 1.0)
    counts = rng.binomial(shots, p_obs)
    return (counts / shots - floor) / f


def entry_std(k_hat: np.ndarray, shots: int, n_qubits: int, p2: float) -> float:
    """Typical standard deviation of one rescaled kernel-entry estimate."""
    f = depolarising_fidelity(n_qubits, p2)
    floor = (1.0 - f) / 2.0**n_qubits
    p = np.clip(f * np.clip(k_hat, 0.0, 1.0) + floor, 0.0, 1.0)
    return float(np.sqrt(np.mean(p * (1.0 - p)) / shots) / f)
