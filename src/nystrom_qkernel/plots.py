"""Figures (Okabe-Ito palette, 200 dpi): heatmap, small multiples, Pareto scatter."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .analysis import PRIMARY_CUT, aggregate  # noqa: E402

BLACK, ORANGE, SKY, GREEN, YELLOW, BLUE, VERM, PURPLE = (
    "#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
DIVERGING = LinearSegmentedColormap.from_list("oi_div", [VERM, "#ffffff", BLUE])
DPI = 200
LABELS = {"digits_3v5": "digits 3 vs 5", "breast_cancer": "breast cancer"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def _fmt_budget(b: float) -> str:
    mant, exp = f"{b:.1e}".split("e")
    mant = mant.rstrip("0").rstrip(".")
    return f"{mant}e{int(exp)}"


def heatmap_truncation_gain(raw: pd.DataFrame, focus: tuple[int, float], path: Path) -> None:
    """Gain of the noise-floor cut over plain Nystrom, per (m, budget) and dataset."""
    n, p2 = focus
    ny = raw[(raw.method == "quantum_nystrom") & (raw.n_qubits == n) & np.isclose(raw.p2, p2)]
    datasets = sorted(ny.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 3.8), squeeze=False)
    lim = 0.0
    mats = []
    for ds in datasets:
        sub = ny[ny.dataset == ds]
        piv = sub.pivot_table(index="m", columns="budget", values="test_bacc",
                              aggfunc="mean", observed=True, margins=False)
        a = sub[np.isclose(sub.c, PRIMARY_CUT)].pivot_table(index="m", columns="budget", values="test_bacc")
        b = sub[np.isclose(sub.c, 0.0)].pivot_table(index="m", columns="budget", values="test_bacc")
        diff = (a - b) * 100.0
        mats.append(diff.reindex(index=piv.index, columns=piv.columns))
        lim = max(lim, float(np.nanmax(np.abs(diff.to_numpy()))) if diff.size else 0.0)
    lim = max(lim, 1.0)
    for ax, ds, diff in zip(axes[0], datasets, mats):
        im = ax.imshow(diff.to_numpy(), cmap=DIVERGING, vmin=-lim, vmax=lim, aspect="auto",
                       origin="lower")
        ax.set_xticks(range(diff.shape[1]), [_fmt_budget(b) for b in diff.columns])
        ax.set_yticks(range(diff.shape[0]), [str(int(m)) for m in diff.index])
        for (i, j), v in np.ndenumerate(diff.to_numpy()):
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7.5, color=BLACK)
        ax.set_xlabel("training shot budget (shots)")
        ax.set_ylabel("landmarks m")
        ax.set_title(LABELS.get(ds, ds))
    fig.colorbar(im, ax=axes.ravel().tolist(), label="gain in balanced accuracy (percentage points)")
    fig.suptitle(f"Noise-floor cut minus plain Nystrom, {n} qubits, two-qubit error {p2:g}"
                 " (mean of seeds)", y=1.02)
    _save(fig, path)


def small_multiples_noise(table: pd.DataFrame, refs: pd.DataFrame, n: int, path: Path) -> None:
    """Accuracy versus shot budget; one panel per dataset and two-qubit error."""
    sub = table[table.n_qubits == n]
    datasets = sorted(sub.dataset.unique())
    p2s = sorted(sub.p2.unique())
    fig, axes = plt.subplots(len(datasets), len(p2s), figsize=(3.1 * len(p2s), 2.9 * len(datasets)),
                             sharex=True, sharey=True, squeeze=False)
    styles = [("quantum_full_kernel", -1.0, "full kernel SVM (standard)", VERM, "s"),
              ("nystrom_cv_m", 0.0, "Nystrom, plain", ORANGE, "o"),
              ("nystrom_cv_m", PRIMARY_CUT, "Nystrom, noise-floor cut", BLUE, "^")]
    for r, ds in enumerate(datasets):
        for k, p2 in enumerate(p2s):
            ax = axes[r, k]
            cell = sub[(sub.dataset == ds) & np.isclose(sub.p2, p2)]
            for method, c, label, col, mk in styles:
                s = cell[(cell.method == method) & np.isclose(cell.c, c)]
                agg = aggregate(s, ["budget"]).sort_values("budget")
                if agg.empty:
                    continue
                ax.plot(agg.budget, agg.bacc_mean, marker=mk, ms=4, color=col, label=label)
                ax.fill_between(agg.budget, agg.bacc_mean - agg.bacc_std.fillna(0),
                                agg.bacc_mean + agg.bacc_std.fillna(0), color=col, alpha=0.15, lw=0)
            rf = refs[(refs.dataset == ds) & (refs.n_qubits == n) & np.isclose(refs.p2, p2)]
            for method, col, ls, label in [("quantum_exact_kernel", BLACK, "--", "exact kernel, infinite shots"),
                                           ("classical_rbf_svm", GREEN, ":", "classical RBF SVM")]:
                ax.axhline(rf[rf.method == method].test_bacc.mean(), color=col, ls=ls, lw=1.2, label=label)
            ax.set_xscale("log")
            ax.set_title(f"{LABELS.get(ds, ds)}, p2 = {p2:g}")
            if r == len(datasets) - 1:
                ax.set_xlabel("training shot budget (shots)")
            if k == 0:
                ax.set_ylabel("test balanced accuracy")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Accuracy at a matched shot budget, {n} qubits (mean and standard deviation of seeds;"
                 " Nystrom m chosen by cross-validation)", y=1.0, fontsize=9)
    _save(fig, path)


def pareto_cost_quality(
    raw: pd.DataFrame, refs: pd.DataFrame, focus: tuple[int, float], path: Path
) -> pd.DataFrame:
    """Shot cost versus error for every configuration; returns the Pareto-front rows."""
    n, p2 = focus
    sel = raw[(raw.n_qubits == n) & np.isclose(raw.p2, p2) & (raw.budget > 0)]
    rows = []
    for (method, m, c, budget), g in sel.groupby(["method", "m", "c", "budget"], dropna=False):
        if method == "quantum_nystrom" and not (np.isclose(c, 0.0) or np.isclose(c, PRIMARY_CUT)):
            continue
        rows.append(dict(method=method, m=m, c=c, budget=budget, err=1.0 - g.test_bacc.mean()))
    pts = pd.DataFrame(rows)
    pts["cost"] = pts.budget
    pts = pts.sort_values(["cost", "err"]).reset_index(drop=True)
    best, front = np.inf, []
    for i, r in pts.iterrows():
        if r.err < best - 1e-12:
            best = r.err
            front.append(i)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    groups = [("quantum_full_kernel", None, "full kernel SVM", VERM, "s"),
              ("quantum_nystrom", 0.0, "Nystrom, plain", ORANGE, "o"),
              ("quantum_nystrom", PRIMARY_CUT, "Nystrom, noise-floor cut", BLUE, "^")]
    for method, c, label, col, mk in groups:
        s = pts[(pts.method == method) & (np.isclose(pts.c, c) if c is not None else True)]
        ax.scatter(s.cost, s.err, s=18, color=col, marker=mk, alpha=0.55, label=label, linewidths=0)
    fr = pts.loc[front]
    ax.step(fr.cost, fr.err, where="post", color=BLACK, lw=1.4, label="Pareto front")
    ax.scatter(fr.cost, fr.err, s=46, facecolors="none", edgecolors=BLACK, linewidths=1.3)
    rf = refs[(refs.n_qubits == n) & np.isclose(refs.p2, p2)]
    ax.axhline(1 - rf[rf.method == "quantum_exact_kernel"].test_bacc.mean(), color=BLACK, ls="--", lw=1,
               label="exact kernel, infinite shots")
    ax.axhline(1 - rf[rf.method == "classical_rbf_svm"].test_bacc.mean(), color=GREEN, ls=":", lw=1.4,
               label="classical RBF SVM (no shots)")
    ax.set_xscale("log")
    ax.set_xlabel("training shot budget (shots)")
    ax.set_ylabel("test error = 1 - balanced accuracy")
    ax.set_title(f"Cost versus quality, {n} qubits, two-qubit error {p2:g}\n"
                 "(each marker: one landmark count m or the full kernel; mean of seeds and datasets)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    _save(fig, path)
    return fr


def accuracy_vs_landmarks(raw: pd.DataFrame, focus: tuple[int, float], path: Path) -> None:
    """Small multiples over budgets: accuracy as a function of landmark count m."""
    n, p2 = focus
    sel = raw[(raw.n_qubits == n) & np.isclose(raw.p2, p2)]
    budgets = sorted(sel[sel.budget > 0].budget.unique())
    ncol = min(3, len(budgets))
    nrow = int(np.ceil(len(budgets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.7 * nrow), sharey=True, squeeze=False)
    for ax, b in zip(axes.ravel(), budgets):
        for c, col, mk, label in [(0.0, ORANGE, "o", "plain (no cut)"),
                                  (0.5, SKY, "d", "cut factor 0.5"),
                                  (PRIMARY_CUT, BLUE, "^", "cut factor 1 (primary)")]:
            s = sel[(sel.method == "quantum_nystrom") & np.isclose(sel.c, c) & (sel.budget == b)]
            agg = aggregate(s, ["m"]).sort_values("m")
            if agg.empty:
                continue
            ax.errorbar(agg.m, agg.bacc_mean, yerr=agg.bacc_std.fillna(0), marker=mk, ms=4, capsize=2,
                        color=col, label=label, lw=1.2)
        full = sel[(sel.method == "quantum_full_kernel") & (sel.budget == b)]
        if not full.empty:
            ax.axhline(full.test_bacc.mean(), color=VERM, ls="--", lw=1.2, label="full kernel SVM")
        ax.set_xscale("log")
        ticks = sorted(sel[sel.method == "quantum_nystrom"].m.unique())
        ax.set_xticks(ticks, [str(int(t)) for t in ticks])
        ax.minorticks_off()
        ax.set_title(f"budget {_fmt_budget(b)} shots")
        ax.set_xlabel("landmarks m")
    for ax in axes[:, 0]:
        ax.set_ylabel("test balanced accuracy")
    for ax in axes.ravel()[len(budgets):]:
        ax.axis("off")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(f"Accuracy versus landmark count, {n} qubits, two-qubit error {p2:g}"
                 " (mean and standard deviation over seeds and datasets)", y=1.01, fontsize=9)
    _save(fig, path)


def qubit_scaling(stt: pd.DataFrame, p2: float, tol: float, path: Path) -> None:
    """Shots needed to reach the accuracy target as the register grows."""
    sub = stt[np.isclose(stt.p2, p2) & np.isclose(stt.tol, tol)]
    datasets = sorted(sub.dataset.unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.4 * len(datasets), 3.6), sharey=True, squeeze=False)
    cap = sub.budget_to_target.max() if sub.budget_to_target.notna().any() else 1e8
    specs = [("quantum_full_kernel", -1.0, "full kernel SVM", VERM, "s"),
             ("nystrom_cv_m", 0.0, "Nystrom, plain", ORANGE, "o"),
             ("nystrom_cv_m", PRIMARY_CUT, "Nystrom, noise-floor cut", BLUE, "^")]
    for ax, ds in zip(axes[0], datasets):
        for method, c, label, col, mk in specs:
            s = sub[(sub.dataset == ds) & (sub.method == method) & np.isclose(sub.c, c)].sort_values("n_qubits")
            y = s.budget_to_target.fillna(cap * 4).to_numpy()
            ax.plot(s.n_qubits, y, color=col, marker=mk, label=label, lw=1.2)
            cens = s.budget_to_target.isna().to_numpy()
            ax.scatter(s.n_qubits[cens], y[cens], marker="x", s=60, color=BLACK, zorder=5)
        ax.set_yscale("log")
        ax.set_xticks(sorted(sub.n_qubits.unique()))
        ax.set_xlabel("qubits")
        ax.set_title(LABELS.get(ds, ds))
    axes[0, 0].set_ylabel(f"shots to reach exact-kernel accuracy - {tol * 100:.0f} points")
    h, l = axes[0, 0].get_legend_handles_labels()
    axes[0, -1].scatter([], [], marker="x", color=BLACK, label="target not reached on grid")
    h, l = axes[0, -1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"Shot cost of reaching the target versus register size, two-qubit error {p2:g}", y=1.02)
    _save(fig, path)
