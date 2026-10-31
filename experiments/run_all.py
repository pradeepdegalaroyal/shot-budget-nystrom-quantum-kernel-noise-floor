"""Reproduce every result, table and figure: ``python experiments/run_all.py [--quick]``."""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402

from nystrom_qkernel import analysis, plots, validation  # noqa: E402
from nystrom_qkernel.experiment import Grid, run_cell  # noqa: E402

FULL = Grid(
    datasets=("breast_cancer", "digits_3v5"),
    cells=tuple((6, p) for p in (0.0, 0.003, 0.01, 0.03)) + ((4, 0.003), (8, 0.003)),
    seeds=(0, 1, 2, 3, 4),
    budgets=(50_000, 200_000, 800_000, 3_200_000, 12_800_000, 51_200_000),
    landmarks=(5, 10, 20, 40, 80),
    cuts=(0.0, 0.25, 0.5, 1.0, 2.0),
)
QUICK = Grid(
    datasets=("breast_cancer",),
    cells=((4, 0.0), (4, 0.01)),
    seeds=(0, 1),
    budgets=(200_000, 3_200_000),
    landmarks=(10, 40),
    cuts=(0.0, 0.5, 1.0),
)


def _hardware_summary() -> dict:
    return {"python": platform.python_version(), "platform": platform.platform(),
            "cpu_threads": os.cpu_count(), "gpu_used": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run (< 1 minute), temp output dir")
    ap.add_argument("--jobs", type=int, default=4, help="worker processes")
    ap.add_argument("--reanalyse", action="store_true", help="skip simulation; reuse results/raw_cells.csv")
    ap.add_argument("--out", type=Path, default=None, help="output root (default: repo, or temp if --quick)")
    args = ap.parse_args()
    grid = QUICK if args.quick else FULL
    out = args.out or (Path(tempfile.mkdtemp(prefix="nqk_quick_")) if args.quick else ROOT)
    res, fig = out / "results", out / "figures"
    res.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    jobs = [(ds, n, p2, seed) for ds in grid.datasets for (n, p2) in grid.cells for seed in grid.seeds]
    if args.reanalyse:
        raw = pd.read_csv(res / "raw_cells.csv")
        t_sim = json.loads((res / "run_meta.json").read_text())["simulation_seconds"]
    else:
        print(f"[run_all] {len(jobs)} cells, {args.jobs} workers -> {out}")
        chunks = Parallel(n_jobs=args.jobs)(delayed(run_cell)(grid, *j) for j in jobs)
        raw = pd.DataFrame([r for chunk in chunks for r in chunk])
        raw.to_csv(res / "raw_cells.csv", index=False)
        t_sim = time.time() - t0

    # Simulator and noise-model validation against Qiskit / Aer.
    val_cells = [(4, 0.01)] if args.quick else [(4, 0.003), (6, 0.003), (6, 0.01), (6, 0.03)]
    n_pairs = 3 if args.quick else 12
    validation_report = {
        "statevector_max_abs_kernel_error": {str(n): validation.statevector_error(n) for n in (4, 6)},
        "noise_model_vs_aer": [validation.noise_model_error(n, p, n_pairs) for n, p in val_cells],
    }
    (res / "aer_validation.json").write_text(json.dumps(validation_report, indent=2))

    # Analysis tables.
    table = analysis.method_table(raw)
    refs = analysis.references(raw)
    curves = analysis.aggregate(table, ["dataset", "n_qubits", "p2", "method", "c", "budget"])
    curves.to_csv(res / "summary_curves.csv", index=False)
    refs_s = analysis.aggregate(refs, ["dataset", "n_qubits", "p2", "method"])
    refs_s.to_csv(res / "summary_references.csv", index=False)
    analysis.all_contrasts(table, list(grid.cuts)).to_csv(res / "contrasts.csv", index=False)
    stt = analysis.shots_to_target(table, refs)
    stt.to_csv(res / "shots_to_target.csv", index=False)
    focus = grid.cells[0] if args.quick else (6, 0.003)
    analysis.headline_table(table, refs, *focus).to_csv(res / "headline_table.csv", index=False)
    contrasts = pd.read_csv(res / "contrasts.csv")
    analysis.contrast_sign_counts(contrasts).to_csv(res / "contrast_sign_counts.csv", index=False)
    analysis.target_reach_summary(stt).to_csv(res / "target_reach_summary.csv", index=False)
    pd.concat([analysis.plain_argmax_landmarks(raw, *focus, cut=c) for c in (0.0, analysis.PRIMARY_CUT)]).to_csv(
        res / "h1_accuracy_by_landmark_count.csv", index=False)
    analysis.cut_gain_large_m(raw, *focus).to_csv(res / "h2_cut_gain_large_m.csv", index=False)
    ny = raw[raw.method == "quantum_nystrom"]
    ny.groupby(["n_qubits", "p2", "budget", "m", "c"])[["rank", "shots", "sigma"]].mean().reset_index().to_csv(
        res / "rank_and_noise_floor.csv", index=False)

    # Figures.
    plots.heatmap_truncation_gain(raw, focus, fig / "fig1_cut_gain_heatmap.png")
    plots.small_multiples_noise(table, refs, focus[0], fig / "fig2_noise_small_multiples.png")
    front = plots.pareto_cost_quality(raw, refs, focus, fig / "fig3_pareto_cost_quality.png")
    front.to_csv(res / "pareto_front.csv", index=False)
    plots.accuracy_vs_landmarks(raw, focus, fig / "fig4_accuracy_vs_landmarks.png")
    if len({n for n, _ in grid.cells}) > 1:
        plots.qubit_scaling(stt, 0.003, analysis.LOOSE_TOLERANCE, fig / "fig5_qubit_scaling.png")

    if args.reanalyse:
        return
    meta = {"quick": args.quick, "cells": len(jobs), "simulation_seconds": round(t_sim, 1),
            "total_seconds": round(time.time() - t0, 1), "workers": args.jobs, **_hardware_summary(),
            "numpy": np.__version__, "pandas": pd.__version__}
    (res / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[run_all] done in {meta['total_seconds']} s (simulation {meta['simulation_seconds']} s)")


if __name__ == "__main__":
    main()
