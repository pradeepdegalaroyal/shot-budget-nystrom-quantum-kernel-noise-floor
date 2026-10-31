"""Hardware-facing checks: two-qubit gate counts and an Aer noise validation."""

from __future__ import annotations

import numpy as np

from .feature_map import REPS


def two_qubit_gate_count(n_qubits: int, compute_uncompute: bool = True) -> int:
    """CX count of the transpiled ZZ feature map (``reps=2``, linear entanglement).

    Each of the ``n - 1`` neighbouring ZZ interactions costs two CX per
    repetition, so one map costs ``2 * REPS * (n - 1)`` and the compute-uncompute
    circuit twice that. :func:`transpiled_cx_count` verifies the formula.
    """
    one_map = 2 * REPS * (n_qubits - 1)
    return 2 * one_map if compute_uncompute else one_map


def _feature_map(n_qubits: int):
    from qiskit.circuit.library import zz_feature_map

    return zz_feature_map(n_qubits, reps=REPS, entanglement="linear")


def transpiled_cx_count(n_qubits: int) -> int:
    """CX count of one feature map after Qiskit transpilation to {cx, u}."""
    from qiskit import transpile

    circ = transpile(_feature_map(n_qubits), basis_gates=["cx", "u"], optimization_level=0)
    return int(circ.count_ops().get("cx", 0))


def aer_overlap_probability(
    x: np.ndarray, y: np.ndarray, p2: float, n_qubits: int | None = None
) -> float:
    """Probability of the all-zero outcome of ``U(y)^dagger U(x)|0>`` from Aer.

    Uses the density-matrix simulator with a two-qubit depolarising error of
    probability ``p2`` on every CX (no optimisation, so gates are not cancelled).
    """
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel, depolarizing_error

    n = int(n_qubits or len(x))
    fm = _feature_map(n)
    qc = QuantumCircuit(n)
    qc.compose(fm.assign_parameters(list(map(float, x))), inplace=True)
    qc.compose(fm.assign_parameters(list(map(float, y))).inverse(), inplace=True)
    noise = NoiseModel()
    if p2 > 0:
        noise.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    sim = AerSimulator(method="density_matrix", noise_model=noise)
    circ = transpile(qc, basis_gates=["cx", "u"], optimization_level=0)
    circ.save_density_matrix()
    rho = sim.run(circ).result().data()["density_matrix"]
    return float(np.real(np.asarray(rho)[0, 0]))
