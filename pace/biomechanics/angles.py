"""2-D joint angle calculations."""

from __future__ import annotations

import numpy as np


def angle_series(a: np.ndarray, vertex: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Return the smaller angle A-vertex-C in degrees for each row."""
    u = a[:, :2] - vertex[:, :2]
    v = c[:, :2] - vertex[:, :2]
    denom = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
    dot = np.einsum("ij,ij->i", u, v)
    cosine = np.divide(dot, denom, out=np.full_like(dot, np.nan), where=denom > 1e-9)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def safe_range(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return float("nan")
    return float(np.percentile(finite, 95) - np.percentile(finite, 5))

