"""Classifiers fitted on precomputed kernels or on Nystrom features."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

C_GRID = (0.1, 1.0, 10.0, 100.0)
CV_FOLDS = 3


def psd_clip(k: np.ndarray) -> np.ndarray:
    """Project a symmetric kernel estimate onto the PSD cone (clip eigenvalues at 0)."""
    sym = 0.5 * (k + k.T)
    vals, vecs = np.linalg.eigh(sym)
    return (vecs * np.clip(vals, 0.0, None)) @ vecs.T


def _folds(y: np.ndarray, seed: int):
    return list(StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed).split(y, y))


def fit_precomputed_svm(
    k_train: np.ndarray, y_train: np.ndarray, k_test: np.ndarray, y_test: np.ndarray, seed: int
) -> tuple[float, float, float]:
    """Cross-validate ``C`` on a precomputed kernel; return (test, cv balanced acc, C)."""
    folds = _folds(y_train, seed)
    best = (-1.0, C_GRID[0])
    for c in C_GRID:
        scores = []
        for tr, va in folds:
            clf = SVC(kernel="precomputed", C=c).fit(k_train[np.ix_(tr, tr)], y_train[tr])
            scores.append(balanced_accuracy_score(y_train[va], clf.predict(k_train[np.ix_(va, tr)])))
        if np.mean(scores) > best[0]:
            best = (float(np.mean(scores)), c)
    clf = SVC(kernel="precomputed", C=best[1]).fit(k_train, y_train)
    return float(balanced_accuracy_score(y_test, clf.predict(k_test))), best[0], best[1]


def fit_linear_svm(
    f_train: np.ndarray, y_train: np.ndarray, f_test: np.ndarray, y_test: np.ndarray, seed: int
) -> tuple[float, float, float]:
    """Cross-validate ``C`` for a linear SVM on explicit features."""
    folds = _folds(y_train, seed)
    best = (-1.0, C_GRID[0])
    for c in C_GRID:
        scores = []
        for tr, va in folds:
            clf = SVC(kernel="linear", C=c).fit(f_train[tr], y_train[tr])
            scores.append(balanced_accuracy_score(y_train[va], clf.predict(f_train[va])))
        if np.mean(scores) > best[0]:
            best = (float(np.mean(scores)), c)
    clf = SVC(kernel="linear", C=best[1]).fit(f_train, y_train)
    return float(balanced_accuracy_score(y_test, clf.predict(f_test))), best[0], best[1]


def classical_baselines(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, seed: int
) -> dict[str, float]:
    """Tuned RBF-kernel SVM and random forest on the same ``n``-dimensional inputs."""
    folds = _folds(y_train, seed)
    best = (-1.0, None)
    for gamma in (0.03, 0.1, 0.3, 1.0):
        for c in C_GRID:
            scores = []
            for tr, va in folds:
                clf = SVC(kernel="rbf", gamma=gamma, C=c).fit(x_train[tr], y_train[tr])
                scores.append(balanced_accuracy_score(y_train[va], clf.predict(x_train[va])))
            if np.mean(scores) > best[0]:
                best = (float(np.mean(scores)), (gamma, c))
    gamma, c = best[1]
    rbf = SVC(kernel="rbf", gamma=gamma, C=c).fit(x_train, y_train)
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=1).fit(x_train, y_train)
    return {
        "classical_rbf_svm": float(balanced_accuracy_score(y_test, rbf.predict(x_test))),
        "classical_random_forest": float(balanced_accuracy_score(y_test, rf.predict(x_test))),
    }
