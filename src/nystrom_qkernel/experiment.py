"""Experiment grid: one cell = (dataset, qubits, gate error, seed)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import make_split
from .feature_map import exact_kernel, zz_states
from .models import classical_baselines, fit_linear_svm, fit_precomputed_svm, psd_clip
from .nystrom import landmark_eigensystem, noise_floor_threshold, nystrom_features
from .shots import entry_std, sample_kernel_block

GAMMAS = (0.025, 0.05, 0.1, 0.2)


@dataclass
class Grid:
    """Sweep definition shared by the full run and the quick smoke run."""

    datasets: tuple[str, ...]
    cells: tuple[tuple[int, float], ...]  # (n_qubits, two-qubit error)
    seeds: tuple[int, ...]
    budgets: tuple[int, ...]
    landmarks: tuple[int, ...]
    cuts: tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 4.0)
    n_train: int = 160
    n_test: int = 160
    extra: dict = field(default_factory=dict)


def nystrom_entries(n_train: int, m: int) -> int:
    """Distinct kernel entries needed: landmark block plus every non-landmark row."""
    return m * (m - 1) // 2 + (n_train - m) * m


def full_entries(n_train: int) -> int:
    """Distinct off-diagonal entries of the full training kernel."""
    return n_train * (n_train - 1) // 2


def _sym_sample(exact_mm: np.ndarray, shots: int, n: int, p2: float, rng) -> np.ndarray:
    """Sample the upper triangle once and mirror it; diagonal is exactly 1."""
    est = sample_kernel_block(exact_mm, shots, n, p2, rng)
    upper = np.triu(est, 1)
    return upper + upper.T + np.eye(len(exact_mm))


def _select_scale(x_tr: np.ndarray, y_tr: np.ndarray, seed: int) -> float:
    """Choose the angle scale by exact-kernel cross-validation (simulation-side)."""
    best = (-1.0, GAMMAS[0])
    for g in GAMMAS:
        k = exact_kernel(zz_states(g * x_tr))
        _, cv, _ = fit_precomputed_svm(k, y_tr, k[:8], y_tr[:8], seed)
        if cv > best[0]:
            best = (cv, g)
    return best[1]


def run_cell(grid: Grid, dataset: str, n: int, p2: float, seed: int) -> list[dict]:
    """Run every method at every budget for one (dataset, qubits, p2, seed) cell."""
    sp = make_split(dataset, n, grid.n_train, grid.n_test, seed)
    gamma = _select_scale(sp.x_train, sp.y_train, seed)
    s_tr, s_te = zz_states(gamma * sp.x_train), zz_states(gamma * sp.x_test)
    k_tt, k_et = exact_kernel(s_tr), exact_kernel(s_te, s_tr)
    base = dict(dataset=dataset, n_qubits=n, p2=p2, seed=seed, gamma=gamma)
    rows: list[dict] = []

    def add(**kw):
        rows.append({**base, **kw})

    ref, _, _ = fit_precomputed_svm(k_tt, sp.y_train, k_et, sp.y_test, seed)
    add(method="quantum_exact_kernel", budget=0, m=0, c=np.nan, test_bacc=ref, cv_bacc=np.nan)
    for name, val in classical_baselines(sp.x_train, sp.y_train, sp.x_test, sp.y_test, seed).items():
        add(method=name, budget=0, m=0, c=np.nan, test_bacc=val, cv_bacc=np.nan)

    n_tr = len(sp.y_train)
    for b_idx, budget in enumerate(grid.budgets):
        s_full = budget // full_entries(n_tr)
        if s_full >= 1:
            rng = np.random.default_rng([seed, n, b_idx, 0])
            k_hat = psd_clip(_sym_sample(k_tt, s_full, n, p2, rng))
            k_hat_te = sample_kernel_block(k_et, s_full, n, p2, rng)
            acc, cv, _ = fit_precomputed_svm(k_hat, sp.y_train, k_hat_te, sp.y_test, seed)
            add(method="quantum_full_kernel", budget=budget, m=0, c=np.nan, shots=s_full,
                entries=full_entries(n_tr), test_bacc=acc, cv_bacc=cv)
        for m in grid.landmarks:
            s = budget // nystrom_entries(n_tr, m)
            if s < 1:
                continue
            rng = np.random.default_rng([seed, n, b_idx, m])
            land = np.random.default_rng([seed, n, m]).permutation(n_tr)
            lm, rest = land[:m], land[m:]
            k_mm = _sym_sample(k_tt[np.ix_(lm, lm)], s, n, p2, rng)
            k_rm = sample_kernel_block(k_tt[np.ix_(rest, lm)], s, n, p2, rng)
            k_em = sample_kernel_block(k_et[:, lm], s, n, p2, rng)
            k_nm = np.empty((n_tr, m))
            k_nm[lm], k_nm[rest] = k_mm, k_rm
            vals, vecs = landmark_eigensystem(k_mm)
            sigma = entry_std(k_rm, s, n, p2)
            for c in grid.cuts:
                tau = noise_floor_threshold(sigma, m, c)
                f_tr, rank = nystrom_features(k_nm, vals, vecs, tau)
                f_te, _ = nystrom_features(k_em, vals, vecs, tau)
                acc, cv, _ = fit_linear_svm(f_tr, sp.y_train, f_te, sp.y_test, seed)
                add(method="quantum_nystrom", budget=budget, m=m, c=c, shots=s,
                    entries=nystrom_entries(n_tr, m), rank=rank, sigma=sigma,
                    test_bacc=acc, cv_bacc=cv)
    return rows
