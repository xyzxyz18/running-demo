"""Compare every detected foot-strike cycle against its side's mean path."""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from pace.pose.landmarks import LANDMARK_NAMES


SAMPLES_PER_CYCLE = 64
INDEX = {name: index for index, name in enumerate(LANDMARK_NAMES)}


def _rounded(value: Optional[float]) -> Optional[float]:
    return round(float(value), 3) if value is not None and np.isfinite(value) else None


def _select_reference_cycles(cycles: List[dict], paths: Dict[int, np.ndarray]) -> List[int]:
    """Choose complete interior cycles and reject shape outliers around a medoid."""
    candidates = []
    for cycle in cycles:
        number = cycle["number"]
        if number in (1, len(cycles)):
            cycle["status"] = "首尾排除"
        elif cycle["path"] is None:
            cycle["status"] = "无法绘制"
        elif (cycle["visibility_percent"] < 70 or cycle["phase_start"] > .1
              or cycle["phase_end"] < .9 or cycle["max_gap_fraction"] > .2):
            cycle["status"] = "可见度不足"
        else:
            candidates.append(number)
    if len(candidates) >= 3:
        matrix = np.stack([paths[number] for number in candidates])
        distances = np.sqrt(np.mean(np.sum((matrix[:, None] - matrix[None, :]) ** 2, axis=3), axis=2))
        medoid = int(np.argmin(distances.sum(axis=1)))
        distance_from_medoid = distances[medoid]
        center = float(np.median(distance_from_medoid))
        mad = float(np.median(np.abs(distance_from_medoid - center)))
        limit = max(.12, center + 3 * 1.4826 * mad)
        candidates = [number for number, distance in zip(candidates, distance_from_medoid)
                      if distance <= limit]
        for cycle in cycles:
            if (cycle["number"] not in candidates and cycle["status"] == "可比较"):
                cycle["status"] = "轨迹异常"
    for cycle in cycles:
        cycle["included_in_mean"] = cycle["number"] in candidates
        if cycle["included_in_mean"]:
            cycle["status"] = "纳入平均"
    return candidates


def foot_cycle_analysis(points: np.ndarray, strikes: Dict[str, List[int]],
                        scale: float, min_visibility: float = 0.45,
                        timestamps: Optional[np.ndarray] = None,
                        aspect: float = 1.0,
                        trajectories: Optional[dict] = None,
                        validity: Optional[dict] = None) -> Dict[str, object]:
    """Return all same-side strike intervals, mean paths and per-cycle RMS errors.

    All pose backends track the left/right ankle only.
    Both ankles use the midpoint of the visible hips as the origin and the same
    body-segment scale. Image Y is inverted so positive Y means upward motion.
    Intervals with too few visible points remain listed but have no path/error.
    """
    if points.ndim != 3 or points.shape[1:] != (33, 4):
        raise ValueError("关键点数组必须为 (帧数, 33, 4)")
    body_scale = float(scale) if np.isfinite(scale) and scale > 1e-6 else 1.0
    phase = np.linspace(0.0, 1.0, SAMPLES_PER_CYCLE)
    hips = np.stack((points[:, INDEX["left_hip"], :],
                     points[:, INDEX["right_hip"], :]), axis=1)
    hip_visible = (np.isfinite(hips[:, :, :2]).all(axis=2) &
                   np.isfinite(hips[:, :, 3]) &
                   (hips[:, :, 3] >= min_visibility))
    pelvis = np.full((len(points), 2), np.nan)
    for frame_id in range(len(points)):
        if hip_visible[frame_id].any():
            pelvis[frame_id] = hips[frame_id, hip_visible[frame_id], :2].mean(axis=0)

    output = {}
    all_deviations = []
    for side in ("left", "right"):
        landmark = f"{side}_ankle"
        ankle = points[:, INDEX[landmark], :]
        if trajectories is None:
            relative = (ankle[:, :2] - pelvis) * [aspect, 1.0] / body_scale
            relative[:, 1] *= -1  # Up is positive in the chart.
            visible = (np.isfinite(relative).all(axis=1) &
                       np.isfinite(ankle[:, 3]) & (ankle[:, 3] >= min_visibility))
        else:
            relative = np.asarray(trajectories[side], dtype=float)
            if relative.shape != (len(points), 2):
                raise ValueError('校正轨迹必须为 (帧数, 2)')
            visible = np.isfinite(relative).all(axis=1)
            if validity is not None:
                visible &= np.asarray(validity[side], dtype=bool)
        cycles = []
        paths = {}
        raw_paths = {}
        side_strikes = strikes.get(side, [])
        for number, (start, end) in enumerate(zip(side_strikes, side_strikes[1:]), 1):
            record = {
                "number": number, "start_frame": int(start), "end_frame": int(end),
                "duration_seconds": None, "visibility_percent": 0,
                "path": None, "phase_start": None, "phase_end": None,
                "max_gap_fraction": None,
                "deviation_body_ratio": None, "status": "无法绘制",
                "included_in_mean": False,
            }
            if 0 <= start < end < len(points):
                if timestamps is not None and len(timestamps) == len(points):
                    duration = float(timestamps[end] - timestamps[start])
                    record["duration_seconds"] = round(duration, 3) if duration >= 0 else None
                indices = np.arange(start, end + 1)
                valid = indices[visible[indices]]
                coverage = len(valid) / len(indices)
                record["visibility_percent"] = round(coverage * 100)
                if len(valid) >= 3:
                    source_phase = (valid - start) / (end - start)
                    if (timestamps is not None and len(timestamps) == len(points) and
                            np.isfinite(timestamps[start:end + 1]).all() and
                            timestamps[end] > timestamps[start] and
                            np.all(np.diff(timestamps[valid]) > 0)):
                        source_phase = ((timestamps[valid] - timestamps[start]) /
                                        (timestamps[end] - timestamps[start]))
                    path = np.column_stack([
                        np.interp(phase, source_phase, relative[valid, axis])
                        for axis in (0, 1)
                    ])
                    record["path"] = np.round(path, 4).tolist()
                    record["phase_start"] = float(source_phase[0])
                    record["phase_end"] = float(source_phase[-1])
                    record["max_gap_fraction"] = round(float(np.max(np.diff(source_phase))), 3)
                    record["status"] = "低置信度" if coverage < 0.6 else "可比较"
                    paths[number] = (path, (phase >= source_phase[0]) & (phase <= source_phase[-1]))
                    raw_paths[number] = path
            cycles.append(record)

        selected = _select_reference_cycles(cycles, raw_paths)
        if selected:
            # Average corresponding cycle phases only where a landmark was
            # actually observed. np.interp's constant endpoint extrapolation
            # must not pull the thick mean line towards an occluded foot.
            sum_path = np.zeros((SAMPLES_PER_CYCLE, 2))
            sample_count = np.zeros(SAMPLES_PER_CYCLE, dtype=int)
            for number in selected:
                path, observed = paths[number]
                sum_path[observed] += path[observed]
                sample_count[observed] += 1
            mean = np.zeros_like(sum_path)
            covered = sample_count > 0
            mean[covered] = sum_path[covered] / sample_count[covered, None]
            for axis in (0, 1):
                mean[:, axis] = np.interp(phase, phase[covered], mean[covered, axis])
        else:
            mean = np.empty((0, 2))
        deviations = []
        for cycle in cycles:
            if cycle["path"] is None or not selected:
                continue
            path = raw_paths[cycle["number"]]
            observed = (phase >= cycle["phase_start"]) & (phase <= cycle["phase_end"])
            deviation = float(np.sqrt(np.mean(np.sum((path[observed] - mean[observed]) ** 2, axis=1))))
            cycle["deviation_body_ratio"] = _rounded(deviation)
            if cycle["included_in_mean"] and len(selected) >= 2:
                deviations.append(deviation)
                all_deviations.append(deviation)
        side_dispersion = float(np.sqrt(np.mean(np.square(deviations)))) if deviations else None
        output[side] = {
            "landmark": landmark,
            "landmark_index": INDEX[landmark],
            "cycle_count": len(cycles),
            "drawable_count": len(paths),
            "included_count": len(selected),
            "dispersion_body_ratio": _rounded(side_dispersion),
            "mean_path": np.round(mean, 4).tolist(),
            "cycles": cycles,
        }

    overall = (float(np.sqrt(np.mean(np.square(all_deviations))))
               if all_deviations else None)
    left_mean = np.asarray(output["left"]["mean_path"])
    right_mean = np.asarray(output["right"]["mean_path"])
    side_gap = (float(np.sqrt(np.mean(np.sum((left_mean - right_mean) ** 2, axis=1))))
                if len(left_mean) and len(right_mean) else None)
    if overall is None:
        assessment = "数据不足"
        explanation = "同侧至少需要两个纳入平均的周期，才能计算周期与平均曲线的差异。"
    elif min(output["left"]["included_count"], output["right"]["included_count"]) < 3:
        assessment = "样本较少"
        explanation = "已展示全部检测到的周期，但至少一侧不足三个纳入平均的周期，稳定性结论需谨慎。"
    elif overall < 0.15:
        assessment = "周期较一致"
        explanation = "各周期相对本侧平均曲线的差异较小。"
    elif overall < 0.30:
        assessment = "有一定波动"
        explanation = "各周期相对本侧平均曲线有一定差异。"
    else:
        assessment = "周期波动较大"
        explanation = "部分周期与本侧平均曲线差异较大，建议结合视频检查关键点与着地检测。"
    return {
        "version": 6,
        "left": output["left"], "right": output["right"],
        "overall_dispersion_body_ratio": _rounded(overall),
        "side_mean_gap_body_ratio": _rounded(side_gap),
        "assessment": assessment,
        "explanation": explanation,
        "method": "同侧相邻脚踝着地事件形成周期；脚踝相对双髋中点，横坐标按视频宽高比修正，再按腿长归一化。每侧首尾各排除一个周期，中间周期需达到 70% 可见度、覆盖 10%-90% 相位且无超过 20% 周期的观测空缺；对至少三个候选周期，以距离总和最小的代表周期为基准，用中位数和 MAD 筛除明显离群轨迹。平均值仅使用入选周期的实际可见区间。排除周期仍显示但不参与均值和稳定性评分。",
    }
