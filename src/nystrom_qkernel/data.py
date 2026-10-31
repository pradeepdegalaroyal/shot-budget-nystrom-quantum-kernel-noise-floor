"""Datasets: stratified subsampling, PCA to one component per qubit, z-scoring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATASETS = ("breast_cancer", "digits_3v5")


@dataclass
class Split:
    """Train/test split already reduced to ``n_qubits`` standardised features."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _load(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name == "breast_cancer":
        d = load_breast_cancer()
        return d.data, d.target
    if name == "digits_3v5":
        d = load_digits()
        mask = np.isin(d.target, (3, 5))
        return d.data[mask], (d.target[mask] == 5).astype(int)
    raise ValueError(f"unknown dataset {name!r}")


def make_split(name: str, n_qubits: int, n_train: int, n_test: int, seed: int) -> Split:
    """Stratified split with PCA fitted on the training part only."""
    x, y = _load(name)
    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, train_size=n_train, test_size=n_test, stratify=y, random_state=seed
    )
    scaler = StandardScaler().fit(x_tr)
    pca = PCA(n_components=n_qubits, random_state=seed).fit(scaler.transform(x_tr))
    z_tr = pca.transform(scaler.transform(x_tr))
    z_te = pca.transform(scaler.transform(x_te))
    post = StandardScaler().fit(z_tr)
    return Split(post.transform(z_tr), y_tr, post.transform(z_te), y_te)
