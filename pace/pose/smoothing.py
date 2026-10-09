"""Missing value interpolation and time-series smoothing."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def interpolate_and_smooth(values: np.ndarray, fps: float,
                           window_seconds: float = 0.18) -> np.ndarray:
    """Interpolate NaNs per coordinate and apply a Savitzky-Golay filter."""
    result = np.asarray(values, dtype=np.float64).copy()
    if result.ndim == 1:
        result = result[:, None]
        squeeze = True
    else:
        squeeze = False

    n = len(result)
    x = np.arange(n)
    for col in range(result.shape[1]):
        series = result[:, col]
        valid = np.isfinite(series)
        if not valid.any():
            continue
        series[~valid] = np.interp(x[~valid], x[valid], series[valid])
        desired = max(5, int(round(window_seconds * fps)))
        if desired % 2 == 0:
            desired += 1
        window = min(desired, n if n % 2 else n - 1)
        if window >= 5:
            series[:] = savgol_filter(series, window, 2, mode="interp")
        result[:, col] = series
    return result[:, 0] if squeeze else result


def preprocess_landmarks(raw: np.ndarray, fps: float, min_visibility: float,
                         window_seconds: float) -> np.ndarray:
    """Mask low-confidence coordinates, interpolate, then smooth x/y/z."""
    clean = raw.copy()
    for landmark in range(clean.shape[1]):
        visible = clean[:, landmark, 3] >= min_visibility
        clean[~visible, landmark, :3] = np.nan
        clean[:, landmark, :3] = interpolate_and_smooth(
            clean[:, landmark, :3], fps, window_seconds
        )
    return clean

