"""Foot position and normalized velocity helpers."""

from __future__ import annotations

import numpy as np


def leg_length(points: np.ndarray, side: str, aspect: float,
               min_visibility: float = 0.45) -> float:
    """Median thigh plus median shank length in image-height units."""
    from pace.pose.landmarks import LANDMARK_NAMES
    lookup = {name: i for i, name in enumerate(LANDMARK_NAMES)}
    lengths = []
    for first, second in (("hip", "knee"), ("knee", "ankle")):
        a = points[:, lookup[f"{side}_{first}"]]
        b = points[:, lookup[f"{side}_{second}"]]
        valid = (np.isfinite(a[:, :2]).all(axis=1) & np.isfinite(b[:, :2]).all(axis=1)
                 & (a[:, 3] >= min_visibility) & (b[:, 3] >= min_visibility))
        delta = (a[valid, :2] - b[valid, :2]) * [aspect, 1.0]
        lengths.append(float(np.median(np.linalg.norm(delta, axis=1))) if len(delta) else np.nan)
    return float(sum(lengths)) if np.isfinite(lengths).all() else float("nan")


def body_scale(shoulder: np.ndarray, hip: np.ndarray, knee: np.ndarray,
               ankle: np.ndarray) -> float:
    segments = np.concatenate([
        np.linalg.norm(shoulder[:, :2] - hip[:, :2], axis=1),
        np.linalg.norm(hip[:, :2] - knee[:, :2], axis=1),
        np.linalg.norm(knee[:, :2] - ankle[:, :2], axis=1),
    ])
    finite = segments[np.isfinite(segments) & (segments > 1e-5)]
    return float(np.median(finite)) if len(finite) else 1.0


def velocity(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(values, 1.0 / fps)
