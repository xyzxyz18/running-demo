"""Local running-pose analysis CLI using RTMPose.

Usage:
    python main.py input/test.mp4 --output output
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from analysis.feedback import build_feedback
from analysis.metrics import compute_metrics
from analysis.stability import foot_cycle_analysis
from analysis.view_correction import correct_trajectory, validate_calibration, LIMITATIONS
from pose.lifting import lift_pose, H36M_NAMES
from pose.skeleton3d import estimate_skeleton, LIMITATION as SKELETON3D_LIMITATION
from pose.quality import temporal_support, smooth_supported
from biomechanics.angles import angle_series
from biomechanics.foot_tracking import body_scale, leg_length
from biomechanics.gait_events import FootEvents, detect_ankle_events
from config import AnalysisConfig
from pose.landmarks import LANDMARK_NAMES
from pose.backends import MODEL_NAMES, create_estimator, validate_model
from pose.smoothing import preprocess_landmarks
from visualization.plots import create_report
from visualization.video_overlay import draw_panel, draw_pose
from visualization.pdf_report import create_pdf_report


IDX = {name: i for i, name in enumerate(LANDMARK_NAMES)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="侧面跑步视频姿态与步态分析")
    parser.add_argument("video", type=Path, help="输入 .mp4/.mov/.avi 视频")
    parser.add_argument("--output", type=Path, default=Path("output"), help="输出目录")
    parser.add_argument("--calibration", type=Path, help="站立标定 JSON")
    parser.add_argument("--reference-video", type=Path, help="同机位站立参考视频")
    return parser.parse_args()


def validate_video(path: Path) -> None:
    if not path.is_file():
        raise ValueError(f"找不到输入视频: {path}")
    if path.suffix.lower() not in {".mp4", ".mov", ".avi"}:
        raise ValueError("仅支持 .mp4、.mov、.avi 视频")


def extract_pose(video: Path, model: str = "rtmpose") -> Tuple[np.ndarray, float, int, int, np.ndarray]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV 无法打开视频: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: List[np.ndarray] = []
    timestamps: List[float] = []
    try:
        with create_estimator(model) as estimator:
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
                result = estimator.process(frame)
                frames.append(result.landmarks if result.landmarks is not None
                              else np.full((33, 4), np.nan))
                timestamps.append(timestamp)
                index += 1
                if index % max(1, int(fps * 2)) == 0:
                    print(f"\r姿态检测: {index}/{total or '?'} 帧", end="", flush=True)
    finally:
        capture.release()
    print()
    if not frames:
        raise RuntimeError("视频不包含可读取的帧")
    frame_times = np.asarray(timestamps, dtype=np.float64)
    if (not np.all(np.isfinite(frame_times)) or len(frame_times) < 2
            or np.count_nonzero(np.diff(frame_times) > 0) < len(frame_times) * 0.8):
        frame_times = np.arange(len(frames), dtype=np.float64) / fps
    else:
        frame_times -= frame_times[0]
    return np.stack(frames), fps, width, height, frame_times


def side_angles(points: np.ndarray, side: str) -> Dict[str, np.ndarray]:
    p = lambda name: points[:, IDX[f"{side}_{name}"], :]
    return {
        "knee": 180 - angle_series(p("hip"), p("knee"), p("ankle")),
        "hip": angle_series(p("shoulder"), p("hip"), p("knee")),
        "ankle": angle_series(p("knee"), p("ankle"), p("foot_index")),
    }


def save_landmarks(path: Path, raw: np.ndarray, smooth: np.ndarray, fps: float) -> None:
    fields = ["frame_id", "timestamp", "landmark", "x", "y", "z", "visibility",
              "x_smooth", "y_smooth", "z_smooth"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame_id in range(len(raw)):
            for i, name in enumerate(LANDMARK_NAMES):
                writer.writerow({
                    "frame_id": frame_id,
                    "timestamp": round(frame_id / fps, 6),
                    "landmark": name,
                    "x": raw[frame_id, i, 0], "y": raw[frame_id, i, 1],
                    "z": raw[frame_id, i, 2], "visibility": raw[frame_id, i, 3],
                    "x_smooth": smooth[frame_id, i, 0],
                    "y_smooth": smooth[frame_id, i, 1],
                    "z_smooth": smooth[frame_id, i, 2],
                })


def save_timeline(path: Path, points: np.ndarray, fps: float, timestamps: np.ndarray,
                  angles: Dict[str, Dict[str, np.ndarray]],
                  event_map: Dict[int, str], foot_motion: Dict[str, object],
                  correction: dict = None, corrected_motion: dict = None,
                  plane_motion: dict = None) -> None:
    """Save compact frame data used by the synchronized browser player."""
    def series(values: np.ndarray) -> List[object]:
        return [round(float(value), 2) if np.isfinite(value) else None for value in values]

    frames = []
    for frame in points:
        frames.append([
            [round(float(value), 5) if np.isfinite(value) else None for value in point]
            for point in frame
        ])
    payload = {
        "fps": round(float(fps), 4),
        "frame_count": len(points),
        "timestamps": [round(float(value), 6) for value in timestamps],
        "landmarks": frames,
        "angles": {
            "left_knee": series(angles["left"]["knee"]),
            "right_knee": series(angles["right"]["knee"]),
            "left_hip": series(angles["left"]["hip"]),
            "right_hip": series(angles["right"]["hip"]),
        },
        "events": {str(frame): label for frame, label in event_map.items()},
        "foot_motion": foot_motion,
    }
    if correction is not None:
        payload["view_correction"] = correction
    if corrected_motion is not None:
        payload["foot_motion_corrected"] = corrected_motion
    if plane_motion is not None:
        payload['foot_motion_plane'] = plane_motion
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))


def save_report_html(path: Path, result: Dict[str, object]) -> None:
    """Write a portable, expandable one-page summary next to the static chart."""
    report = result["report"]
    observations = "\n".join(
        f"<li>{html.escape(str(item))}</li>" for item in report["observations"]
    )
    content = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(str(report['title']))}</title><style>
body{{margin:0;background:#0b100e;color:#edf3ee;font:16px/1.7 system-ui,sans-serif}}
main{{max-width:840px;margin:5vh auto;padding:28px}}small{{color:#c9ff42;letter-spacing:.15em}}
h1{{font-size:clamp(36px,7vw,70px);line-height:1.1}}p,li{{color:#b7c3bb}}
details{{border:1px solid #334039;border-radius:12px;margin:25px 0;padding:20px}}
summary{{cursor:pointer;color:#c9ff42;font-weight:700}}li{{margin:12px 0}}
img{{display:block;max-width:100%;margin:20px auto;border-radius:8px}}
</style></head><body><main><small>PACE LAB · REPORT</small>
<h1>{html.escape(str(report['title']))}</h1><p>{html.escape(str(report['summary']))}</p>
<details open><summary>展开分析结论</summary><ul>{observations}</ul>
<p>{html.escape(str(report['method']))}</p></details>
<details><summary>查看图表报告</summary><img src="report.png" alt="分析图表"></details>
<p>{html.escape(str(result['disclaimer']))}</p></main></body></html>"""
    path.write_text(content, encoding="utf-8")


def summarize_foot_motion(motion: Dict[str, object], metrics: Dict[str, object],
                          feedback: List[str]) -> Tuple[Dict[str, object], Dict[str, object]]:
    summary = {
        **{side: {"landmark": f"{side}_ankle",
                  "landmark_index": IDX[f"{side}_ankle"],
                  "cycle_count": motion[side]["cycle_count"],
                  "drawable_count": motion[side]["drawable_count"],
                  "included_count": motion[side]["included_count"],
                  "dispersion_body_ratio": motion[side]["dispersion_body_ratio"]}
           for side in ("left", "right")},
        "version": motion["version"],
        "overall_dispersion_body_ratio": motion["overall_dispersion_body_ratio"],
        "side_mean_gap_body_ratio": motion["side_mean_gap_body_ratio"],
        "assessment": motion["assessment"],
        "explanation": motion["explanation"],
        "method": motion["method"],
    }
    report = {
        "title": "跑姿分析简报",
        "summary": motion["explanation"],
        "observations": [
            f"步频：{metrics['cadence_steps_per_min'] if metrics.get('cadence_steps_per_min') is not None else '数据不足'} 步/分钟。",
            f"脚踝轨迹重复性：{motion['assessment']}；左右检测到的周期分别为 {motion['left']['cycle_count']} 和 {motion['right']['cycle_count']} 个。",
            f"纳入平均周期：左 {motion['left']['included_count']} 个，右 {motion['right']['included_count']} 个；总体周期偏差为 {motion['overall_dispersion_body_ratio'] if motion['overall_dispersion_body_ratio'] is not None else '数据不足'} 个腿长。",
            f"左右平均轨迹差异：{motion['side_mean_gap_body_ratio'] if motion['side_mean_gap_body_ratio'] is not None else '数据不足'} 个腿长。",
            *feedback,
        ],
        "method": motion["method"],
    }
    return summary, report


def save_annotated_video(source: Path, destination: Path, points: np.ndarray, fps: float,
                         width: int, height: int, angles: Dict[str, Dict[str, np.ndarray]],
                         event_map: Dict[int, str], metrics: Dict[str, object],
                         config: AnalysisConfig) -> None:
    capture = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建输出视频: {destination}")
    recent_event = ""
    recent_until = -1
    for i in range(len(points)):
        ok, frame = capture.read()
        if not ok:
            break
        if i in event_map:
            recent_event, recent_until = event_map[i], i + int(config.event_display_seconds * fps)
        event_text = recent_event if i <= recent_until else ""
        draw_pose(frame, points[i], config.min_visibility)
        values = [angles[s]["knee"][i] for s in ("left", "right")]
        finite = [v for v in values if np.isfinite(v)]
        draw_panel(frame, float(np.mean(finite)) if finite else None,
                   metrics.get("cadence_steps_per_min"), event_text, (i + 1) / len(points))
        writer.write(frame)
    capture.release()
    writer.release()


def analyze(video: Path, output: Path, config: AnalysisConfig, model: str = "rtmpose",
            calibration: dict = None, reference_video: Path = None) -> Dict[str, object]:
    calibration = validate_calibration(calibration)
    validate_model(model)
    validate_video(video)
    output.mkdir(parents=True, exist_ok=True)
    (output / "pose3d.npz").unlink(missing_ok=True)
    for name in ('skeleton3d.json', 'skeleton3d.npz'):
        (output / name).unlink(missing_ok=True)
    raw, fps, width, height, timestamps = extract_pose(video, model)
    detected_ratio = float(np.isfinite(raw[:, :, 0]).any(axis=1).mean())
    if detected_ratio < 0.2:
        raise RuntimeError(f"人体检测有效帧仅 {detected_ratio:.1%}，请使用无遮挡的固定侧面全身视频")
    smooth = preprocess_landmarks(raw, fps, config.min_visibility,
                                  config.smoothing_window_seconds)
    aspect = width / height if height else 1.0
    corrected = smooth.copy()
    corrected[:, :, 0] *= aspect
    angles = {side: side_angles(corrected, side) for side in ("left", "right")}
    events: Dict[str, FootEvents] = {}
    clearances = {}
    leg_lengths = {}
    for side in ("left", "right"):
        leg_lengths[side] = leg_length(smooth, side, aspect, config.min_visibility)
        events[side], clearances[side] = detect_ankle_events(
            smooth[:, IDX[f"{side}_ankle"]], leg_lengths[side], fps, config.min_visibility)
    lengths = [value for value in leg_lengths.values() if np.isfinite(value)]
    scale = float(np.mean(lengths)) if lengths else 1.0
    metrics = compute_metrics(
        fps, len(smooth), events["left"].strikes, events["right"].strikes,
        angles["left"]["knee"], angles["right"]["knee"],
        angles["left"]["hip"], angles["right"]["hip"],
        corrected[:, IDX["left_ankle"]], corrected[:, IDX["right_ankle"]],
        corrected[:, IDX["left_hip"]], corrected[:, IDX["right_hip"]], scale,
        config.min_stride_seconds, config.max_stride_seconds,
    )
    metrics.update({
        "source_video": str(video.resolve()), "fps": round(fps, 3),
        "pose_model": model, "pose_model_name": MODEL_NAMES[model],
        "pose_keypoint_count": 17,
        "video_aspect_ratio": round(aspect, 6),
        "frame_count": len(smooth), "pose_detection_rate": round(detected_ratio, 3),
        "left_foot_strikes": events["left"].strikes,
        "right_foot_strikes": events["right"].strikes,
        "left_toe_offs": events["left"].toe_offs,
        "right_toe_offs": events["right"].toe_offs,
        "left_leg_length_image_heights": round(leg_lengths["left"], 3) if np.isfinite(leg_lengths["left"]) else None,
        "right_leg_length_image_heights": round(leg_lengths["right"], 3) if np.isfinite(leg_lengths["right"]) else None,
    })
    feedback = build_feedback(metrics, config)
    foot_motion = foot_cycle_analysis(
        smooth, {side: events[side].strikes for side in ("left", "right")},
        scale, config.min_visibility, timestamps, aspect,
    )
    correction = {"status": "not_requested", "limitations": LIMITATIONS}
    corrected_motion = None
    pose3d = None
    original_pose3d = None
    skeleton_correction = None
    if calibration is not None:
        try:
            print("正在估计三维骨架和侧面轨迹…", flush=True)
            pose3d = lift_pose(raw, timestamps, aspect, config.min_visibility)
            original_pose3d = pose3d.copy()
            support, _ = temporal_support(raw, timestamps, config.min_visibility)
            pose3d = smooth_supported(pose3d, timestamps, support)
            ref_points = ref_pose = ref_times = None
            ref_support = support
            if calibration['source'] == 'reference':
                if reference_video is None:
                    raise ValueError('缺少站立参考视频')
                ref_raw, ref_fps, ref_width, ref_height, ref_times = extract_pose(reference_video, model)
                if abs(ref_width/ref_height - aspect) > .001:
                    raise ValueError('参考视频与跑步视频的画面比例必须相同，且相机不能变焦或移动')
                ref_points = preprocess_landmarks(ref_raw, ref_fps, config.min_visibility,
                                                 config.smoothing_window_seconds)
                ref_pose = lift_pose(ref_raw, ref_times, aspect, config.min_visibility)
                ref_support, _ = temporal_support(ref_raw, ref_times, config.min_visibility)
                ref_pose = smooth_supported(ref_pose, ref_times, ref_support)
            corrected = correct_trajectory(smooth, pose3d, timestamps, aspect, calibration,
                                           ref_points, ref_pose, ref_times, config.min_visibility,
                                           temporal_valid=support, reference_valid=ref_support)
            skeleton_correction = corrected
            corrected_motion = foot_cycle_analysis(smooth,
                {side: events[side].strikes for side in ('left','right')}, 1,
                config.min_visibility, timestamps, aspect,
                trajectories=corrected['trajectories'], validity=corrected['validity'])
            corrected_motion['method'] = ('三维脚踝投影至固定前后—上下坐标系；'
                '原点为双髋中点沿估计竖直方向在地面的投影；以三维腿长归一化。' + LIMITATIONS)
            corrected_motion['coordinate_system'] = 'estimated_sagittal_ground'
            correction = corrected['metadata']
            np.savez_compressed(output/'pose3d.npz',
                joint_names=np.asarray(H36M_NAMES), timestamps=timestamps,
                camera_pose_pelvis_relative_m=corrected['pose3d_m'],
                ground_origins_pelvis_relative_m=corrected['ground_origins_pelvis_relative_m'],
                hip_height_m=corrected['hip_height_m'],
                left_trajectory=corrected['trajectories']['left'],
                right_trajectory=corrected['trajectories']['right'],
                left_valid=corrected['validity']['left'], right_valid=corrected['validity']['right'])
            metrics['corrected_path_dispersion_leg_ratio'] = corrected_motion['overall_dispersion_body_ratio']
            metrics['corrected_side_mean_gap_leg_ratio'] = corrected_motion['side_mean_gap_body_ratio']
        except (RuntimeError, ValueError, OSError, ImportError) as exc:
            corrected_motion = None
            skeleton_correction = None
            (output / 'pose3d.npz').unlink(missing_ok=True)
            correction = {"status": "unavailable", "reason": str(exc),
                          "calibration": calibration, "limitations": LIMITATIONS}
    (output/'view_correction.json').write_text(json.dumps(correction, ensure_ascii=False,
                                                         indent=2, allow_nan=False), 'utf-8')
    metrics["foot_path_dispersion_body_ratio"] = foot_motion["overall_dispersion_body_ratio"]
    metrics["left_right_mean_path_gap_body_ratio"] = foot_motion["side_mean_gap_body_ratio"]
    foot_summary, report = summarize_foot_motion(foot_motion, metrics, feedback)
    result = {"metrics": metrics, "feedback": feedback,
              "foot_motion": foot_summary,
              "report": report,
              "disclaimer": "二维视频估算结果，仅供运动观察，不用于医疗诊断。"}
    result['view_correction'] = correction
    plane_motion = None
    try:
        print('正在使用 RTMPose 二维序列估计 VideoPose3D 三维骨架…', flush=True)
        result['skeleton3d'] = estimate_skeleton(video, raw, timestamps, output, pose3d=pose3d,
                                               correction=skeleton_correction,
                                               correction_metadata=correction,
                                               original_pose=original_pose3d)
        if result['skeleton3d'].get('leg_plane_constraint',{}).get('status')=='available':
            with np.load(output/'skeleton3d.npz') as data:
                projected=data['plane_side_keypoints']
            paths={side:projected[:,joint] for side,joint in [('left',6),('right',3)]}
            masks={side:np.isfinite(path).all(axis=1) for side,path in paths.items()}
            plane_motion=foot_cycle_analysis(smooth,
                {side:events[side].strikes for side in ('left','right')},1,
                config.min_visibility,timestamps,aspect,trajectories=paths,validity=masks)
            plane_motion['coordinate_system']='hip_plane_pelvis_relative'
            plane_motion['method']='双髋连线法向活动平面侧面投影；髋中心原点；腿长尺度；周期边界来自二维步态事件；非地面标定'
            result['foot_motion_plane'],_=summarize_foot_motion(plane_motion,metrics,feedback)
    except Exception as exc:  # Optional 3D playback must not discard valid 2D results.
        for name in ('skeleton3d.json', 'skeleton3d.npz', 'runningpose_raw.npz'):
            (output / name).unlink(missing_ok=True)
        result['skeleton3d'] = dict(status='unavailable', reason=str(exc),
                                    limitations=SKELETON3D_LIMITATION)
        print(f'三维骨架不可用，继续保存二维分析：{exc}', flush=True)
    if corrected_motion is not None:
        result['foot_motion_corrected'], _ = summarize_foot_motion(corrected_motion, metrics, feedback)
        report['observations'].append(f"估计偏离正侧面 {correction['deviation_from_side_degrees']}°；侧面轨迹以髋中点的地面投影为原点。")
    elif correction['status'] == 'unavailable':
        report['observations'].append('侧面校正不可用：' + correction['reason'])
    report['observations'].append('膝/髋角度为二维估计。' + (LIMITATIONS if calibration else ''))
    save_landmarks(output / "landmarks.csv", raw, smooth, fps)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    save_report_html(output / "report.html", result)
    event_frames: Dict[int, str] = {}
    for side in ("left", "right"):
        for i in events[side].strikes:
            event_frames[i] = f"{side.title()} foot strike"
        for i in events[side].toe_offs:
            event_frames[i] = f"{side.title()} toe-off"
    save_timeline(output / "timeline.json", smooth, fps, timestamps, angles, event_frames, foot_motion, correction, corrected_motion, plane_motion)
    save_annotated_video(video, output / "annotated.mp4", smooth, fps, width, height,
                         angles, event_frames, metrics, config)
    times = timestamps
    create_report(
        output / "report.png", times,
        {"left_knee": angles["left"]["knee"], "right_knee": angles["right"]["knee"]},
        {"left": clearances["left"], "right": clearances["right"]},
        {"left_strikes": events["left"].strikes,
         "right_strikes": events["right"].strikes}, metrics, feedback,
    )
    create_pdf_report(output / "report.pdf", result, times, angles, clearances, foot_motion, corrected_motion)
    create_browser_video(video, output / "player.mp4")
    return result


def create_browser_video(source: Path, destination: Path) -> None:
    """Produce H.264 with a seekable MP4 index for long MOV/AVI and MP4 uploads."""
    command = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(source),
               "-map", "0:v:0", "-an", "-vf",
               "scale=1280:720:force_original_aspect_ratio=decrease:force_divisible_by=2",
               "-c:v", "libx264", "-preset", "veryfast",
               "-crf", "25", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(destination)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, timeout=7200)


def main() -> int:
    args = parse_args()
    try:
        calibration = json.loads(args.calibration.read_text('utf-8')) if args.calibration else None
        result = analyze(args.video, args.output, AnalysisConfig(), calibration=calibration,
                         reference_video=args.reference_video)
    except (ValueError, RuntimeError) as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n完成，结果位于: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
