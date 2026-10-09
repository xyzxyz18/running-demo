"""Aggregate gait measurements from pace.pose time series."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from pace.biomechanics.angles import safe_range


def _rounded_finite(value: float, digits: int = 1) -> Optional[float]:
    return round(float(value), digits) if np.isfinite(value) else None


def _mean_intervals(events: List[int], fps: float, low: float = 0.0,
                    high: float = 99.0) -> Optional[float]:
    if len(events) < 2:
        return None
    intervals = np.diff(events) / fps
    intervals = intervals[(intervals >= low) & (intervals <= high)]
    return round(float(np.mean(intervals)), 3) if len(intervals) else None


def _symmetry(left: Optional[float], right: Optional[float]) -> Optional[float]:
    if left is None or right is None or (left + right) == 0:
        return None
    return round(abs(left - right) / ((left + right) / 2) * 100, 2)


def compute_metrics(fps: float, frame_count: int, left_strikes: List[int],
                    right_strikes: List[int], left_knee: np.ndarray,
                    right_knee: np.ndarray, left_hip: np.ndarray,
                    right_hip: np.ndarray, left_foot: np.ndarray,
                    right_foot: np.ndarray, left_hip_xy: np.ndarray,
                    right_hip_xy: np.ndarray, body_scale: float,
                    min_stride: float, max_stride: float) -> Dict[str, object]:
    duration = frame_count / fps
    raw_steps = sorted([(i, "left") for i in left_strikes] +
                       [(i, "right") for i in right_strikes])
    # Side-view landmarks can overlap and produce a duplicate left/right event.
    # Merge only near-simultaneous events; true running steps are much farther apart.
    all_steps = []
    duplicate_window = max(1, int(round(0.10 * fps)))
    for step in raw_steps:
        if all_steps and step[0] - all_steps[-1][0] <= duplicate_window:
            continue
        all_steps.append(step)
    alternating = [b[0] - a[0] for a, b in zip(all_steps, all_steps[1:]) if a[1] != b[1]]
    step_time = round(float(np.mean(alternating) / fps), 3) if alternating else None
    left_stride = _mean_intervals(left_strikes, fps, min_stride, max_stride)
    right_stride = _mean_intervals(right_strikes, fps, min_stride, max_stride)
    stride_values = [v for v in (left_stride, right_stride) if v is not None]
    stride_time = round(float(np.mean(stride_values)), 3) if stride_values else None
    cadence = round(120 / stride_time, 1) if stride_time else None
    left_knee_rom = _rounded_finite(safe_range(left_knee))
    right_knee_rom = _rounded_finite(safe_range(right_knee))
    hip_combined = np.concatenate([left_hip, right_hip])
    finite_hip = hip_combined[np.isfinite(hip_combined)]

    strike_offsets = []
    for events, foot, hip in ((left_strikes, left_foot, left_hip_xy),
                              (right_strikes, right_foot, right_hip_xy)):
        for i in events:
            if np.isfinite(foot[i, 0]) and np.isfinite(hip[i, 0]):
                strike_offsets.append((foot[i, 0] - hip[i, 0]) / max(body_scale, 1e-6))

    return {
        "duration_seconds": round(duration, 2),
        "detected_steps": len(all_steps),
        "cadence_steps_per_min": cadence,
        "step_time_seconds": step_time,
        "stride_time_seconds": stride_time,
        "left_stride_time_seconds": left_stride,
        "right_stride_time_seconds": right_stride,
        "left_knee_rom_degrees": left_knee_rom,
        "right_knee_rom_degrees": right_knee_rom,
        "knee_rom_degrees": _rounded_finite(np.mean(
            [v for v in (left_knee_rom, right_knee_rom) if v is not None]
        )) if left_knee_rom is not None or right_knee_rom is not None else None,
        "hip_rom_degrees": _rounded_finite(safe_range(hip_combined)),
        "max_hip_flexion_degrees": _rounded_finite(np.min(finite_hip)) if len(finite_hip) else None,
        "max_hip_extension_degrees": _rounded_finite(np.max(finite_hip)) if len(finite_hip) else None,
        "stride_time_asymmetry_percent": _symmetry(left_stride, right_stride),
        "knee_rom_asymmetry_percent": _symmetry(left_knee_rom, right_knee_rom),
        "mean_foot_strike_offset_body_ratio": (
            round(float(np.mean(strike_offsets)), 3) if strike_offsets else None
        ),
    }
