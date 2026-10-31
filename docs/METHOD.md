# Method

Abbreviations: support vector machine (SVM), radial basis function (RBF), principal component analysis (PCA), cross-validation (CV), positive semidefinite (PSD), controlled-NOT gate (CX), quantum processing unit (QPU), noisy intermediate-scale quantum (NISQ).

## Quantum model

* **Feature map.** Qiskit `zz_feature_map`, 2 repetitions, linear entanglement, one qubit per PCA component (n = 4, 6, 8). Inputs are z-scored PCA components multiplied by an angle scale chosen from {0.025, 0.05, 0.1, 0.2} by 3-fold CV of the exact-kernel SVM on the training set (a simulation-side choice, identical for all quantum methods).
* **Kernel.** `k(x, y) = |<psi(x)|psi(y)>|^2`, the probability of reading all zeros after the compute-uncompute circuit. The numpy statevector simulator is checked against Qiskit `Statevector` in the tests and in `results/aer_validation.json`.
* **Cost unit.** One shot is one execution of one compute-uncompute circuit. The **training shot budget** B is split evenly over all *distinct* kernel entries the method needs; shots per entry are `floor(B / entries)` and a method is infeasible if that is below 1. Test rows are estimated with the same shots per entry; the per-prediction entry count (N for the full kernel, m for Nystrom) is reported separately.

## Methods compared at a matched budget B

| Method | Role | Entries estimated |
|---|---|---|
| Full kernel SVM, PSD-projected | standard quantum baseline (Schuld and Killoran, 2019) | N(N-1)/2 |
| Nystrom, plain | landmark baseline (as used by Saberikamarposhti, 2027) | m(m-1)/2 + (N-m) m |
| Nystrom, noise-floor cut (proposed) | spectral truncation tied to the shot-noise level | same as plain |
| Exact-kernel SVM | infinite-shot reference, not a feasible method | none |
| Tuned RBF SVM, random forest | classical baselines on the same n-dimensional inputs | none (no shots) |

**Nystrom features.** Landmarks are the first m points of a seeded permutation of the training set. With the estimated landmark block `K_mm = U L U^T`, features are `Phi = K_xm U_r L_r^(-1/2)`, where r counts eigenvalues above a threshold. *Plain* keeps eigenvalues above 1e-10. The *noise-floor cut* keeps eigenvalues above `tau = c * 2 * sigma * sqrt(m)`, where `sigma` is the shot-noise standard deviation of a single entry computed from the estimated entries and the shot count (a random symmetric m x m matrix with entry deviation sigma has spectral norm about `2 sigma sqrt(m)`). The primary setting is c = 1; the ablation sweeps c in {0, 0.25, 0.5, 1, 2}. A linear SVM is fitted on `Phi`.

**Model selection (no test peeking).** The SVM parameter C in {0.1, 1, 10, 100} is chosen by 3-fold stratified CV on the training set for every model. The landmark count m in {5, 10, 20, 40, 80} is chosen by the same CV score (ties go to the smaller m) for the headline "CV-selected m" curves; per-m curves are also reported.

## Metrics

Test **balanced accuracy** on 160 held-out points (mean and standard deviation over 5 seeds; seeds change the data split, landmarks and shot noise). Primary contrast: paired difference (Nystrom with cut minus full kernel SVM) at equal B, paired over (dataset, seed) units, t-based 95% interval. Secondary: **shots to target**, the smallest grid budget at which the seed-mean accuracy is within 2 percentage points of the exact-kernel SVM (NaN if not reached).

## Data

Breast cancer (30 features, 569 samples, scikit-learn) and digits 3 versus 5 (64 features, 365 samples). Stratified 160/160 train/test subsample per seed; PCA is fitted on the training part only.

## Ablation, noise sweep, scaling

* **Ablation:** noise-floor cut factor c in {0, 0.25, 0.5, 1, 2} at each (m, B); heatmap of c = 1 minus c = 0; accuracy versus m per budget.
* **Noise sweep:** two-qubit depolarising probability p2 in {0, 0.003, 0.01, 0.03} at n = 6. A compute-uncompute circuit has `G = 8 (n - 1)` CX gates (formula verified against the transpiler). Gate noise is modelled as a global depolarising channel of fidelity `f = (1 - p2)^G`, so `p_obs = f k + (1 - f) / 2^n`; the estimator inverts this affine map using the known `f` (calibrated rescaling), which inflates the entry variance by `1 / f^2`. Single-qubit gate and readout errors are not modelled. The model is checked against the Aer density-matrix simulator with local two-qubit depolarising errors.
* **Scaling:** n in {4, 6, 8} at p2 = 0.003.
* **Budget grid:** 5.0e4, 2.0e5, 8.0e5, 3.2e6, 1.28e7, 5.12e7 shots.

## Reproduction and compute

`python experiments/run_all.py` runs everything on CPU (statevector and Aer on at most 8 qubits); `--quick` is a smoke run. The hypotheses and thresholds in `docs/RESEARCH_GAP.md` were fixed before the full run.
