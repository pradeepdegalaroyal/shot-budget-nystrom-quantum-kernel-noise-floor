"""Cross-checks of the numpy simulator and the global-depolarising model against Qiskit/Aer."""

from __future__ import annotations

import numpy as np
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

from .data import make_split
from .feature_map import REPS, exact_kernel, zz_states
from .hardware import aer_overlap_probability, transpiled_cx_count, two_qubit_gate_count
from .shots import depolarising_fidelity


def statevector_error(n_qubits: int, n_points: int = 4, seed: int = 0) -> float:
    """Largest ``| |<a|b>|^2_numpy - |<a|b>|^2_qiskit |`` over random point pairs."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 3.0, (n_points, n_qubits))
    fm = zz_feature_map(n_qubits, reps=REPS, entanglement="linear")
    ref = np.array([Statevector(fm.assign_parameters(list(r))).data for r in x])
    return float(np.abs(exact_kernel(zz_states(x)) - exact_kernel(ref)).max())


def noise_model_error(n_qubits: int, p2: float, n_pairs: int, seed: int = 0) -> dict:
    """Compare the closed-form noisy all-zero probability with Aer density-matrix runs.

    Inputs are real (PCA-reduced, scaled) breast-cancer points so the circuits
    resemble those of the experiments.
    """
    sp = make_split("breast_cancer", n_qubits, 40, 40, seed)
    x = 0.1 * sp.x_train
    rng = np.random.default_rng(seed)
    f = depolarising_fidelity(n_qubits, p2)
    states = zz_states(x)
    errs, aer_vals, model_vals = [], [], []
    for _ in range(n_pairs):
        i, j = rng.choice(len(x), 2, replace=False)
        k = float(np.abs(np.vdot(states[i], states[j])) ** 2)
        model = f * k + (1.0 - f) / 2.0**n_qubits
        aer = aer_overlap_probability(x[i], x[j], p2)
        errs.append(aer - model)
        aer_vals.append(aer)
        model_vals.append(model)
    errs = np.asarray(errs)
    return dict(
        n_qubits=n_qubits, p2=p2, n_pairs=n_pairs,
        cx_per_circuit=two_qubit_gate_count(n_qubits),
        cx_transpiled_one_map=transpiled_cx_count(n_qubits),
        model_fidelity=f,
        mean_signed_error=float(errs.mean()),
        rms_error=float(np.sqrt(np.mean(errs**2))),
        max_abs_error=float(np.abs(errs).max()),
        mean_aer_probability=float(np.mean(aer_vals)),
        mean_model_probability=float(np.mean(model_vals)),
    )
