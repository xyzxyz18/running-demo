"""Rule-based gait event detection for fixed-camera side-view video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import median_filter, uniform_filter1d, percentile_filter


@dataclass
class FootEvents:
    strikes: List[int]
    toe_offs: List[int]


def detect_ankle_events(ankle: np.ndarray, leg_length: float, fps: float,
                        min_visibility: float = 0.45) -> tuple[FootEvents, np.ndarray]:
    """Detect descending clearance crossings after a real swing peak."""
    n = len(ankle)
    clearance = np.full(n, np.nan)
    observed = np.isfinite(ankle[:, 1]) & (ankle[:, 3] >= min_visibility)
    if observed.sum() < max(8, int(fps)) or not np.isfinite(leg_length) or leg_length <= 0:
        return FootEvents([], []), clearance
    ids = np.arange(n)
    y = np.interp(ids, ids[observed], ankle[observed, 1])
    # Keep long occlusions out of the event signal.
    nearest = np.minimum(np.abs(ids - np.maximum.accumulate(np.where(observed, ids, -n))),
                         np.abs(np.minimum.accumulate(np.where(observed, ids, 2*n)[::-1])[::-1] - ids))
    usable = nearest <= max(2, round(0.12 * fps))
    window = max(5, int(round(0.8 * fps)) | 1)
    ground = percentile_filter(y, 90, size=window, mode="nearest")
    signal = np.maximum(0, (ground - y) / leg_length)
    signal = uniform_filter1d(median_filter(signal, size=5), size=5)
    clearance[usable] = signal[usable]
    peak_distance = max(2, int(round(0.3 * fps)))
    peaks, _ = find_peaks(signal, distance=peak_distance, prominence=0.025)
    peaks = [int(p) for p in peaks if usable[p] and signal[p] >= 0.04]
    if not peaks:
        return FootEvents([], []), clearance
    typical = float(np.median(signal[peaks]))
    threshold = max(0.015, typical * 0.25)
    strikes = []
    toe_offs = []
    for peak in peaks:
        if signal[peak] < typical * 0.55:
            continue
        end = min(n - 1, peak + max(2, int(round(0.5 * fps))))
        crossing = next((i for i in range(peak + 1, end + 1)
                         if usable[i-1] and usable[i] and signal[i-1] > threshold >= signal[i]), None)
        if crossing is None or (strikes and (crossing - strikes[-1]) / fps < 0.35):
            continue
        strikes.append(crossing)
    for first, second in zip(strikes, strikes[1:]):
        between = [p for p in peaks if first < p < second]
        if between:
            toe_offs.append(int(first + np.argmin(signal[first:between[0]+1])))
    return FootEvents(strikes, toe_offs), clearance


def _enforce_spacing(indices: np.ndarray, scores: np.ndarray, min_frames: int) -> List[int]:
    order = indices[np.argsort(scores[indices])[::-1]]
    selected: List[int] = []
    for idx in order:
        if all(abs(int(idx) - old) >= min_frames for old in selected):
            selected.append(int(idx))
    return sorted(selected)


def detect_foot_events(foot_x: np.ndarray, foot_y: np.ndarray, fps: float,
                       min_interval_seconds: float = 0.22,
                       ground_percentile: float = 72.0,
                       velocity_tolerance: float = 0.42) -> FootEvents:
    """Detect approximate strike/toe-off events from normalized image coordinates.

    Image y increases downward. Strikes are local low points near the runner's
    estimated ground band. Toe-off is the strongest upward transition following
    a strike and before the next strike.
    """
    valid = np.isfinite(foot_y)
    if valid.sum() < max(8, int(fps)):
        return FootEvents([], [])
    y = foot_y.copy()
    x = foot_x.copy()
    ids = np.arange(len(y))
    y[~valid] = np.interp(ids[~valid], ids[valid], y[valid])
    valid_x = np.isfinite(x)
    if valid_x.any():
        x[~valid_x] = np.interp(ids[~valid_x], ids[valid_x], x[valid_x])
    else:
        x[:] = 0.0
    vy = np.gradient(y) * fps
    min_frames = max(2, int(round(min_interval_seconds * fps)))
    spread = max(float(np.percentile(y, 90) - np.percentile(y, 10)), 1e-4)
    peaks, properties = find_peaks(y, distance=min_frames, prominence=0.035 * spread)
    # Camera/runner drift changes the absolute image height over a long video.
    # Compare a peak with its neighbourhood, not one whole-video ground line.
    window = max(min_frames * 2, int(round(0.6 * fps)))
    slow_limit = max(np.percentile(np.abs(vy), 45) * velocity_tolerance, 0.01)
    scores = np.zeros(len(y))
    candidates = []
    for peak, prominence in zip(peaks, properties["prominences"]):
        neighbourhood = y[max(0, peak - window):min(len(y), peak + window + 1)]
        local_ground = np.percentile(neighbourhood, ground_percentile)
        if y[peak] < local_ground - 0.18 * spread:
            continue
        scores[peak] = (prominence + 0.5 * max(0.0, y[peak] - np.median(neighbourhood))) * (
            1.0 if abs(vy[peak]) <= slow_limit else 0.8)
        candidates.append(peak)
    # A same-foot stride cannot occur at the faster left/right step interval.
    stride_frames = max(min_frames, int(round(0.35 * fps)))
    strikes = _enforce_spacing(np.asarray(candidates, dtype=int), scores, stride_frames)

    toe_offs: List[int] = []
    for pos, strike in enumerate(strikes):
        end = strikes[pos + 1] if pos + 1 < len(strikes) else min(len(y), strike + int(fps))
        start = strike + max(1, int(0.08 * fps))
        if end - start < 3:
            continue
        # Most negative vy is the clearest upward departure in image coordinates.
        local = start + int(np.argmin(vy[start:end]))
        if vy[local] < -max(0.015, 0.12 * np.percentile(np.abs(vy), 80)):
            toe_offs.append(local)
    return FootEvents(strikes, toe_offs)
