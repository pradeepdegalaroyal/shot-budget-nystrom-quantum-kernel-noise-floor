"""Fast unit tests for the simulator, shot model, Nystrom features and analysis helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from nystrom_qkernel import analysis
from nystrom_qkernel.experiment import Grid, full_entries, nystrom_entries, run_cell
from nystrom_qkernel.feature_map import exact_kernel, zz_states
from nystrom_qkernel.hardware import transpiled_cx_count, two_qubit_gate_count
from nystrom_qkernel.nystrom import landmark_eigensystem, noise_floor_threshold, nystrom_features
from nystrom_qkernel.shots import depolarising_fidelity, sample_kernel_block


def test_numpy_simulator_matches_qiskit_statevector():
    from nystrom_qkernel.validation import statevector_error

    assert statevector_error(3, n_points=3) < 1e-10


def test_kernel_is_symmetric_psd_unit_diagonal():
    x = np.random.default_rng(1).uniform(0, 2, (12, 4))
    k = exact_kernel(zz_states(x))
    assert np.allclose(k, k.T)
    assert np.allclose(np.diag(k), 1.0)
    assert np.linalg.eigvalsh(k).min() > -1e-10


def test_gate_count_formula_matches_transpiler():
    for n in (3, 4, 6):
        assert transpiled_cx_count(n) == two_qubit_gate_count(n, compute_uncompute=False)


def test_shot_estimator_is_unbiased_and_noise_shrinks_with_shots():
    rng = np.random.default_rng(0)
    exact = np.full(40_000, 0.3)
    for p2 in (0.0, 0.01):
        est = sample_kernel_block(exact, 200, 4, p2, rng)
        assert abs(est.mean() - 0.3) < 0.005
    lo = sample_kernel_block(exact, 50, 4, 0.0, rng).std()
    hi = sample_kernel_block(exact, 800, 4, 0.0, rng).std()
    assert hi < lo / 3.0
    assert depolarising_fidelity(4, 0.0) == 1.0 and depolarising_fidelity(4, 0.01) < 1.0


def test_nystrom_with_all_landmarks_reproduces_kernel():
    x = np.random.default_rng(2).uniform(0, 1, (20, 3))
    k = exact_kernel(zz_states(0.2 * x))
    vals, vecs = landmark_eigensystem(k)
    phi, rank = nystrom_features(k, vals, vecs, threshold=0.0, floor=1e-12)
    assert rank >= 1
    assert np.allclose(phi @ phi.T, k, atol=1e-6)


def test_noise_floor_cut_reduces_rank_monotonically():
    x = np.random.default_rng(3).uniform(0, 1, (30, 4))
    k = exact_kernel(zz_states(0.1 * x))
    vals, vecs = landmark_eigensystem(k)
    ranks = [nystrom_features(k, vals, vecs, noise_floor_threshold(0.01, 30, c))[1] for c in (0.0, 1.0, 4.0)]
    assert ranks[0] >= ranks[1] >= ranks[2] >= 1


def test_entry_counts():
    assert full_entries(10) == 45
    assert nystrom_entries(10, 3) == 3 + 7 * 3


def test_run_cell_smoke_and_budget_feasibility():
    grid = Grid(("breast_cancer",), ((3, 0.0),), (0,), (30_000, 400_000), (5, 20), cuts=(0.0, 1.0),
                n_train=60, n_test=60)
    rows = pd.DataFrame(run_cell(grid, "breast_cancer", 3, 0.0, 0))
    assert {"quantum_exact_kernel", "classical_rbf_svm", "quantum_nystrom"} <= set(rows.method)
    # 60 points need 1770 distinct entries: both budgets are feasible; shots per entry is an integer >= 1.
    full = rows[rows.method == "quantum_full_kernel"]
    assert (full.shots >= 1).all()
    ny = rows[rows.method == "quantum_nystrom"]
    assert (ny.shots * ny.entries <= ny.budget).all()
    assert rows.test_bacc.between(0, 1).all()


def test_landmark_selection_uses_cv_not_test_score():
    raw = pd.DataFrame({
        "dataset": "d", "n_qubits": 4, "p2": 0.0, "seed": 0, "budget": 10, "c": 1.0,
        "method": "quantum_nystrom", "m": [5, 10, 20], "cv_bacc": [0.7, 0.9, 0.9],
        "test_bacc": [0.99, 0.6, 0.8],
    })
    pick = analysis.select_landmark_count(raw)
    assert int(pick.m.iloc[0]) == 10  # best CV score, ties to the smaller m, ignores test score
