"""Aggregation of raw cell results into the tables quoted in the README."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

KEYS = ["dataset", "n_qubits", "p2", "seed"]
PRIMARY_CUT = 1.0
TARGET_TOLERANCE = 0.02
LOOSE_TOLERANCE = 0.05


def select_landmark_count(raw: pd.DataFrame) -> pd.DataFrame:
    """Pick ``m`` per (cell, budget, cut) by training cross-validation (never by test score).

    Ties go to the smaller ``m``. Returns rows labelled ``nystrom_cv_m``.
    """
    ny = raw[raw.method == "quantum_nystrom"].copy()
    ny = ny.sort_values(["cv_bacc", "m"], ascending=[False, True])
    pick = ny.groupby(KEYS + ["budget", "c"], as_index=False).first()
    pick["method"] = "nystrom_cv_m"
    return pick


def method_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-seed rows for every method curve used in the figures (budget > 0)."""
    sel = select_landmark_count(raw)
    full = raw[raw.method == "quantum_full_kernel"].copy()
    full["c"] = -1.0
    return pd.concat([sel, full], ignore_index=True)


def references(raw: pd.DataFrame) -> pd.DataFrame:
    """Shot-free reference accuracies (exact quantum kernel, classical baselines)."""
    return raw[raw.budget == 0][KEYS + ["method", "test_bacc"]]


def aggregate(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Mean and standard deviation of test balanced accuracy over seeds."""
    g = df.groupby(by)["test_bacc"]
    out = g.agg(["mean", "std", "count"]).reset_index()
    return out.rename(columns={"mean": "bacc_mean", "std": "bacc_std", "count": "n"})


def paired_contrast(
    table: pd.DataFrame, cut: float = PRIMARY_CUT, cut_ref: float | None = None
) -> pd.DataFrame:
    """Paired accuracy difference at matched budget, per (n, p2, budget).

    Compares the CV-selected-m Nystrom model with cut ``cut`` against the full
    kernel SVM (``cut_ref=None``) or against Nystrom with cut ``cut_ref``.
    Units of pairing: (dataset, seed). 95% interval uses a t distribution.
    """
    a = table[(table.method == "nystrom_cv_m") & np.isclose(table.c, cut)]
    if cut_ref is None:
        b = table[table.method == "quantum_full_kernel"]
    else:
        b = table[(table.method == "nystrom_cv_m") & np.isclose(table.c, cut_ref)]
    keys = KEYS + ["budget"]
    merged = a[keys + ["test_bacc"]].merge(b[keys + ["test_bacc"]], on=keys, suffixes=("_a", "_b"))
    merged["diff"] = merged.test_bacc_a - merged.test_bacc_b
    rows = []
    for (n, p2, budget), grp in merged.groupby(["n_qubits", "p2", "budget"]):
        d = grp["diff"].to_numpy()
        half = (
            stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else np.nan
        )
        rows.append(dict(n_qubits=n, p2=p2, budget=budget, n_pairs=len(d),
                         diff_mean=d.mean(), ci95_half=half, wins=int((d > 0).sum())))
    return pd.DataFrame(rows)


def shots_to_target(
    table: pd.DataFrame, refs: pd.DataFrame, tols: tuple[float, ...] = (TARGET_TOLERANCE, LOOSE_TOLERANCE)
) -> pd.DataFrame:
    """Smallest grid budget whose seed-mean accuracy is within ``tol`` of the exact kernel.

    ``tol = 0.02`` is the pre-specified target; ``0.05`` is a looser, exploratory one.
    NaN means the target was not reached on the budget grid (right-censored).
    """
    exact = (
        refs[refs.method == "quantum_exact_kernel"]
        .groupby(["dataset", "n_qubits", "p2"])["test_bacc"].mean().rename("exact")
    )
    curves = aggregate(table, ["dataset", "n_qubits", "p2", "method", "c", "budget"])
    rows = []
    for (ds, n, p2, method, c), grp in curves.groupby(["dataset", "n_qubits", "p2", "method", "c"]):
        for tol in tols:
            target = exact.loc[(ds, n, p2)] - tol
            ok = grp[grp.bacc_mean >= target].sort_values("budget")
            rows.append(dict(dataset=ds, n_qubits=n, p2=p2, method=method, c=c, tol=tol, target=target,
                             budget_to_target=float(ok.budget.iloc[0]) if len(ok) else np.nan))
    return pd.DataFrame(rows)


def all_contrasts(table: pd.DataFrame, cuts: list[float]) -> pd.DataFrame:
    """Paired contrasts of every cut factor against the full kernel and against plain Nystrom."""
    parts = []
    for c in cuts:
        full = paired_contrast(table, cut=c)
        full.insert(0, "reference", "full_kernel")
        full.insert(0, "cut", c)
        parts.append(full)
        if c != 0.0:
            plain = paired_contrast(table, cut=c, cut_ref=0.0)
            plain.insert(0, "reference", "plain_nystrom")
            plain.insert(0, "cut", c)
            parts.append(plain)
    return pd.concat(parts, ignore_index=True)


def headline_table(table: pd.DataFrame, refs: pd.DataFrame, n: int, p2: float) -> pd.DataFrame:
    """Mean and std of balanced accuracy over (dataset, seed) units at one (n, p2)."""
    t = table[(table.n_qubits == n) & np.isclose(table.p2, p2)].copy()
    t["label"] = np.where(t.method == "quantum_full_kernel", "full kernel SVM",
                          "Nystrom CV-m, cut " + t.c.astype(str))
    out = aggregate(t, ["label", "budget"])
    r = refs[(refs.n_qubits == n) & np.isclose(refs.p2, p2)]
    ref = aggregate(r, ["method"]).rename(columns={"method": "label"})
    ref["budget"] = 0
    return pd.concat([out, ref], ignore_index=True)


def _t_half(x: np.ndarray) -> float:
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan


def plain_argmax_landmarks(raw: pd.DataFrame, n: int, p2: float, cut: float = 0.0) -> pd.DataFrame:
    """H1: seed- and dataset-mean accuracy per (budget, m) and the arg-max m per budget."""
    f = raw[(raw.method == "quantum_nystrom") & (raw.n_qubits == n) & np.isclose(raw.p2, p2)
            & np.isclose(raw.c, cut)]
    piv = f.pivot_table(index="budget", columns="m", values="test_bacc")
    out = piv.reset_index()
    out.insert(1, "argmax_m", piv.idxmax(axis=1).to_numpy())
    out.insert(0, "cut", cut)
    return out


def cut_gain_large_m(raw: pd.DataFrame, n: int, p2: float, ms: tuple[int, ...] = (40, 80)) -> pd.DataFrame:
    """H2/ablation: gain (points) of each cut factor over plain Nystrom at large m.

    Averaged over budgets and m within each (dataset, seed) unit, then summarised over units.
    """
    f = raw[(raw.method == "quantum_nystrom") & (raw.n_qubits == n) & np.isclose(raw.p2, p2)
            & raw.m.isin(ms)]
    keys = ["dataset", "seed", "budget", "m"]
    base = f[np.isclose(f.c, 0.0)].set_index(keys).test_bacc
    rows = []
    for c in sorted(f.c.unique()):
        if c == 0.0:
            continue
        d = (f[np.isclose(f.c, c)].set_index(keys).test_bacc - base).groupby(["dataset", "seed"]).mean() * 100
        rows.append(dict(cut=c, gain_points=d.mean(), ci95_half_points=_t_half(d.to_numpy()), n_units=len(d)))
    return pd.DataFrame(rows)


def contrast_sign_counts(contrasts: pd.DataFrame) -> pd.DataFrame:
    """Number of (n, p2, budget) cells where the paired 95% interval is above / below zero."""
    c = contrasts.copy()
    c["sig_pos"] = c.diff_mean - c.ci95_half > 0
    c["sig_neg"] = c.diff_mean + c.ci95_half < 0
    return c.groupby(["reference", "cut"]).agg(
        cells=("diff_mean", "size"), significantly_better=("sig_pos", "sum"),
        significantly_worse=("sig_neg", "sum"), mean_diff_points=("diff_mean", lambda s: 100 * s.mean()),
    ).reset_index()


def target_reach_summary(stt: pd.DataFrame) -> pd.DataFrame:
    """How many (dataset, n, p2) groups reach each accuracy target, and the median budget when reached."""
    g = stt.groupby(["method", "c", "tol"])["budget_to_target"]
    return g.agg(groups="size", reached=lambda s: int(s.notna().sum()), median_budget_reached="median").reset_index()
