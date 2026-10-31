"""Nystrom features from a noisy landmark block, with a noise-floor spectral cut."""

from __future__ import annotations

import numpy as np


def noise_floor_threshold(sigma: float, m: int, c: float) -> float:
    """Eigenvalue cut ``c * 2 * sigma * sqrt(m)``.

    A symmetric ``m x m`` matrix with independent zero-mean entries of standard
    deviation ``sigma`` has spectral norm about ``2 * sigma * sqrt(m)``; any
    eigenvalue of the estimated landmark kernel below that level cannot be told
    apart from shot noise.
    """
    return float(c * 2.0 * sigma * np.sqrt(m))


def landmark_eigensystem(k_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigen-decomposition of the symmetrised landmark block (descending order)."""
    sym = 0.5 * (k_mm + k_mm.T)
    vals, vecs = np.linalg.eigh(sym)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]


def nystrom_features(
    k_xm: np.ndarray,
    vals: np.ndarray,
    vecs: np.ndarray,
    threshold: float,
    floor: float = 1e-10,
) -> tuple[np.ndarray, int]:
    """Project kernel rows onto the retained landmark eigenspace.

    ``Phi = K_xm U_r Lambda_r^(-1/2)`` for eigenvalues above
    ``max(threshold, floor)``; ``threshold = 0`` gives the plain Nystrom
    pseudo-inverse square root.

    Returns:
        (features of shape (N, r), retained rank r). If nothing survives the cut
        the single leading eigenpair is kept so a model can still be fitted.
    """
    keep = vals > max(threshold, floor)
    if not keep.any():
        keep = np.zeros_like(vals, dtype=bool)
        keep[0] = True
        vals = np.maximum(vals, floor)
    scale = 1.0 / np.sqrt(vals[keep])
    return k_xm @ (vecs[:, keep] * scale[None, :]), int(keep.sum())
