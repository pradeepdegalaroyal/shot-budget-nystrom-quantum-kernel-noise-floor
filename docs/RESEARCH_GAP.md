# Research gap: what a shot costs when a quantum kernel is approximated by landmarks

Abbreviations: digital object identifier (DOI), quantum machine learning (QML), support vector machine (SVM), quantum support vector machine (QSVM), noisy intermediate-scale quantum (NISQ), quantum processing unit (QPU), parameterised quantum circuit (PQC), radial basis function (RBF), software-defined networking (SDN).

## State of the art (from the supplied literature; citations give the DOI)

Two routes dominate QML on near-term devices (Schuld and Killoran, 2019, doi:10.1103/PhysRevLett.122.040504): a **fidelity kernel** whose entries are overlap estimates fed to a classical SVM, and an **explicit variational model** trained by gradients (Mitarai et al., 2018, doi:10.1103/PhysRevA.98.032309; Benedetti et al., 2019, doi:10.1088/2058-9565/ab4eb5). The kernel route avoids the variational optimisation problem, whose trainability is an acknowledged obstacle (Cerezo et al., 2022, doi:10.1038/s43588-022-00311-3), but pays a quadratic price: a training set of N points needs N(N-1)/2 kernel entries, each estimated from a finite number of shots.

Recent applied work makes this price visible. Dagli et al. (2027, doi:10.1007/978-3-032-39763-8_20) measure a QSVM at roughly 12,970 times the energy of a random forest with a lower F1 score, and ask for a reproducible way to decide where a task should run. Saberikamarposhti (2027, doi:10.1016/j.future.2026.108884) names the quadratic kernel-matrix cost as a bottleneck shared by classical and quantum kernel SVMs, applies a **Nystrom approximation** to cut circuit evaluations, reports a Frobenius reconstruction error of 0.15%, compares models at a matched sample budget, and finds the quantum kernel competitive with a linear classical kernel but below tree ensembles. The same paper includes a depolarising-plus-readout noise study.

## Related work

| Paper | Year | Role | What it does | What it does NOT cover |
|---|---|---|---|---|
| Saberikamarposhti, doi:10.1016/j.future.2026.108884 | 2027 | recent | Multi-class QSVM for SDN attacks; Nystrom approximation of the quantum kernel; budget-matched baselines; noise study | Per its abstract, accuracy is reported against a sample budget, not a shot budget; no analysis of how landmark count trades against per-entry shot noise |
| Dagli et al., doi:10.1007/978-3-032-39763-8_20 | 2027 | recent | HERO cost-energy-latency model for hybrid workloads; QSVM versus classical on KDD Cup 99 | Takes the QSVM as given; does not reduce its quantum cost |
| Zia et al., doi:10.1016/j.enbuild.2026.118407 | 2027 | recent | Fixed quantum feature embeddings improve tree models for facade pressure | No shot, noise or kernel-estimation cost accounting |
| Schuld and Killoran, doi:10.1103/PhysRevLett.122.040504 | 2019 | landmark | Feature-map view; quantum kernel and variational classifier | Assumes kernel entries are available; no finite-budget scaling |
| Mitarai et al., doi:10.1103/PhysRevA.98.032309 | 2018 | landmark | Quantum circuit learning with gradient-trained circuits | Gradient shot cost is not compared with kernel shot cost |
| Benedetti et al., doi:10.1088/2058-9565/ab4eb5 | 2019 | landmark | PQCs as machine-learning models | Review-style; no budgeted comparison |
| Abbas et al., doi:10.1038/s43588-021-00084-1 | 2021 | landmark | Effective dimension as a capacity and trainability measure | Not about kernel estimation cost |
| Schuld et al., doi:10.1103/PhysRevA.103.032430 | 2021 | landmark | Data encoding fixes the Fourier spectrum of a PQC model | Does not link spectrum to the rank a shot-noisy kernel can resolve |
| Cerezo et al., doi:10.1038/s43588-022-00311-3 | 2022 | landmark / review | Challenges: trainability, advantage | Names scalability as open; no measured shot-cost remedy |
| Sajjan et al., doi:10.1039/d2cs00203e | 2022 | review | QML for chemistry and physics | Survey; no budgeted kernel study |
| Houssein et al., doi:10.1016/j.eswa.2022.116512 | 2022 | review | QML state of the art and challenges | Survey; no measurements |
| Maheshwari et al., doi:10.1109/ACCESS.2022.3195044 | 2022 | review | Systematic review of biomedical QML | Notes limitations in general terms; no budgeted study |
| Ayman et al., doi:10.1016/j.future.2026.108856 | 2027 | adjacent | Four-qubit dressed quantum neural network on thermograms; bounded, no-advantage conclusion | Variational model with frozen classical features; no kernel-cost analysis |

## Gap

Landmark (Nystrom) kernels are already used to reduce the number of quantum-kernel evaluations, but the evidence in the supplied papers counts *entries* or *samples*, not *shots*. On a QPU the quantity that is billed is the shot. With a fixed total shot budget, adding landmarks gives a better rank-m approximation of the kernel but leaves fewer shots per entry, and the Nystrom feature map divides by the square roots of the eigenvalues of the estimated landmark block, so the smallest eigenvalues, which shot noise corrupts first, are amplified most. The consequence is a trade-off in the landmark count m that a budget-matched accuracy table cannot show, and a stability question (which eigen-directions are above the noise floor) that is independent of the classifier.

## Testable hypotheses and success metrics

Simulation scale is explicit: 4, 6 and 8 qubits, 160 training points, exact statevectors with shot noise and a global-depolarising gate-noise model validated against the Aer density-matrix simulator.

* **H1 (interior optimum).** At a fixed shot budget, plain Nystrom accuracy peaks at an interior landmark count, not at the largest m, for at least 4 of the 6 budgets (6 qubits, two-qubit error 0.003, seed and dataset mean).
* **H2 (noise-floor cut).** Discarding landmark eigenvalues below `c * 2 * sigma * sqrt(m)` (sigma: per-entry shot-noise standard deviation, c = 1) raises the accuracy of plain Nystrom at m = 40 and 80 by at least 1.0 percentage point on average, with a paired 95% interval that excludes zero.
* **H3 (matched-budget benefit, primary).** With m chosen by training cross-validation only, Nystrom with the cut beats the full kernel SVM (the standard baseline, with positive-semidefinite projection) at the same total training shots on the low-budget end of the grid; success means a paired 95% interval above zero at the three smallest feasible budgets. If the full kernel wins or ties there, the shot-saving claim is rejected.
* **H4 (noise scaling).** Gate noise raises the shot budget needed to get within 2 percentage points of the exact-kernel accuracy; the sweep reports by how much for the full kernel and for Nystrom.

## Novelty in one sentence

The study charges every shot used to build a Nystrom quantum kernel, shows how landmark count and per-entry shot noise interact, and tests a parameter-light spectral cut set by the shot-noise level, against the standard full-kernel SVM and a tuned classical RBF SVM, under a gate-noise sweep.
