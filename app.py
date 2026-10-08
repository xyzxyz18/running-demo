"""Local browser UI for running-pose analysis."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import threading
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from flask import Flask, abort, jsonify, render_template, request, send_from_directory, url_for

from config import AnalysisConfig
from analysis.view_correction import validate_calibration
from main import analyze
from pose.backends import create_estimator, validate_model


BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = Path(os.environ.get("PACE_DATA_DIR", str(BASE_DIR / "output"))).resolve() / "jobs"
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi"}
ARTIFACTS = {"annotated.mp4", "player.mp4", "landmarks.csv", "metrics.json", "report.png", "report.pdf", "report.html", "timeline.json", "view_correction.json", "pose3d.npz", "skeleton3d.json", "skeleton3d.npz", "runningpose_raw.npz"}

app = Flask(__name__)


def _max_upload_bytes() -> int:
    """Read the upload limit once, while keeping a safe default for bad values."""
    raw = os.environ.get("PACE_MAX_UPLOAD_GB", "2")
    try:
        gigabytes = float(raw)
        if not 0.1 <= gigabytes <= 20:
            raise ValueError
    except ValueError:
        gigabytes = 2.0
    return int(gigabytes * 1024 * 1024 * 1024)


app.config["MAX_CONTENT_LENGTH"] = _max_upload_bytes()

jobs: Dict[str, Dict[str, object]] = {}
jobs_lock = threading.Lock()
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="running-analysis")


def artifact_urls(job_id: str, output_dir: Path, source_name: str = "") -> Dict[str, str]:
    artifacts = {
        name: f"/results/{job_id}/{name}"
        for name in ARTIFACTS
        if (output_dir / name).is_file()
    }
    if source_name and (output_dir / source_name).is_file():
        artifacts["source"] = f"/results/{job_id}/{source_name}"
    for reference in output_dir.glob("reference.*"):
        if reference.suffix.lower() in ALLOWED_EXTENSIONS:
            artifacts["reference"] = f"/results/{job_id}/{reference.name}"
            break
    return artifacts


def persist_job_metadata(job_id: str, job: Dict[str, object]) -> None:
    metadata = {
        "id": job_id,
        "filename": job.get("filename", "历史视频"),
        "source_filename": job.get("source_filename", ""),
        "created_at": job.get("created_at"),
        "model": job.get("model", "rtmpose"),
        "calibration": job.get("calibration"),
        "reference_filename": job.get("reference_filename", ""),
    }
    path = JOBS_DIR / job_id / "job.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)


def source_for_job(job: Dict[str, object], output_dir: Path) -> Optional[Path]:
    source_name = str(job.get("source_filename") or "")
    if source_name and (output_dir / source_name).is_file():
        return output_dir / source_name
    result = job.get("result")
    if isinstance(result, dict):
        metrics = result.get("metrics", {})
        candidate = Path(str(metrics.get("source_video", ""))).resolve()
        if candidate.is_file() and BASE_DIR in candidate.parents:
            return candidate
    return None


def load_existing_jobs() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    for output_dir in JOBS_DIR.iterdir():
        if not output_dir.is_dir():
            continue
        metadata_path = output_dir / "job.json"
        try:
            metadata = json.loads(metadata_path.read_text("utf-8")) if metadata_path.is_file() else {}
            result_path = output_dir / "metrics.json"
            result = json.loads(result_path.read_text("utf-8")) if result_path.is_file() else None
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        source_files = [path for path in output_dir.glob("source.*") if path.suffix.lower() in ALLOWED_EXTENSIONS]
        source_name = str(metadata.get("source_filename") or (source_files[0].name if source_files else ""))
        fallback_name = "历史视频"
        if isinstance(result, dict):
            fallback_name = Path(str(result.get("metrics", {}).get("source_video", fallback_name))).name
            if fallback_name.startswith("source."):
                fallback_name = f"历史视频 {output_dir.name[:6]}"
        created_at = metadata.get("created_at") or datetime.fromtimestamp(output_dir.stat().st_mtime).astimezone().isoformat()
        state = "completed" if result else "failed"
        job = {
            "id": output_dir.name,
            "model": metadata.get("model") or (result or {}).get("metrics", {}).get("pose_model", "mediapipe"),
            "calibration": metadata.get("calibration"),
            "reference_filename": metadata.get("reference_filename", ""),
            "filename": metadata.get("filename") or fallback_name,
            "source_filename": source_name,
            "created_at": created_at,
            "state": state,
            "message": "分析完成" if result else "上次分析未完成",
            "artifacts": artifact_urls(output_dir.name, output_dir, source_name),
        }
        if result:
            job["result"] = result
        jobs[output_dir.name] = job


class RealtimePoseService:
    """Keep pose estimators on one dedicated thread for browser-camera frames."""

    def __init__(self) -> None:
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="realtime-pose")
        self.estimator = None
        self.model = None

    def process(self, encoded: bytes, model: str = "rtmpose") -> Dict[str, object]:
        return self.executor.submit(self._process, encoded, model).result()

    def _process(self, encoded: bytes, model: str = "rtmpose") -> Dict[str, object]:
        validate_model(model)
        frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("无法解码摄像头画面")
        if self.estimator is None or self.model != model:
            if self.estimator is not None:
                self.estimator.close()
            self.estimator = None
            self.model = None
            self.estimator = create_estimator(model)
            self.model = model
        pose = self.estimator.process(frame).landmarks
        if pose is None:
            return {"detected": False}

        def joint_angle(a: int, vertex: int, c: int):
            if not np.all(np.isfinite(pose[[a, vertex, c], :2])) or np.any(pose[[a, vertex, c], 3] < 0.5):
                return None
            aspect = frame.shape[1] / frame.shape[0]
            u = (pose[a, :2] - pose[vertex, :2]) * [aspect, 1]
            v = (pose[c, :2] - pose[vertex, :2]) * [aspect, 1]
            denominator = np.linalg.norm(u) * np.linalg.norm(v)
            if denominator < 1e-9:
                return None
            cosine = np.clip(float(np.dot(u, v) / denominator), -1.0, 1.0)
            return round(float(np.degrees(np.arccos(cosine))), 1)

        return {
            "detected": True,
            "model": model,
            "landmarks": [[round(float(v), 5) if np.isfinite(v) else None for v in p] for p in pose],
            "angles": {
            "left_knee": round(180 - joint_angle(23, 25, 27), 1) if joint_angle(23, 25, 27) is not None else None,
            "right_knee": round(180 - joint_angle(24, 26, 28), 1) if joint_angle(24, 26, 28) is not None else None,
                "left_hip": joint_angle(11, 23, 25),
                "right_hip": joint_angle(12, 24, 26),
            },
        }


realtime_pose = RealtimePoseService()


def update_job(job_id: str, **values: object) -> None:
    with jobs_lock:
        jobs[job_id].update(values)


def run_analysis(job_id: str, source: Path, output_dir: Path) -> None:
    update_job(job_id, state="running", message="正在识别人体姿态、估计三维骨架并计算跑姿指标；首次三维分析需要下载模型…")
    try:
        with jobs_lock:
            model = str(jobs[job_id].get("model", "rtmpose"))
            calibration = jobs[job_id].get('calibration')
            reference_name = jobs[job_id].get('reference_filename')
        reference = output_dir / reference_name if reference_name else None
        result = analyze(source, output_dir, AnalysisConfig(), model, calibration, reference)
        artifacts = artifact_urls(job_id, output_dir, source.name)
        update_job(job_id, state="completed", message="分析完成", result=result, artifacts=artifacts)
    except Exception as exc:  # Turn pipeline errors into an actionable UI message.
        update_job(job_id, state="failed", message=str(exc))


load_existing_jobs()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/healthz")
def healthz():
    return jsonify(status="ok")


@app.get("/api/jobs")
def list_jobs():
    with jobs_lock:
        values = []
        for job in jobs.values():
            output_dir = JOBS_DIR / str(job["id"])
            summary = dict(job)
            summary["can_reanalyze"] = source_for_job(job, output_dir) is not None
            values.append(summary)
    values.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return jsonify(jobs=values)


@app.post("/api/jobs")
def create_job():
    upload = request.files.get("video")
    if upload is None or not upload.filename:
        return jsonify(error="请选择一个视频文件"), 400
    suffix = Path(upload.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify(error="仅支持 MP4、MOV 或 AVI 视频"), 400

    try:
        model = validate_model(request.form.get("model", "rtmpose"))
        calibration = validate_calibration(json.loads(request.form.get('calibration', 'null')))
        reference = request.files.get('reference_video')
        if reference and Path(reference.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError('站立参考视频仅支持 MP4、MOV 或 AVI')
        if calibration and calibration['source'] == 'reference' and not reference:
            raise ValueError('请选择站立参考视频')
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    job_id = uuid.uuid4().hex
    output_dir = JOBS_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=False)
    source = output_dir / f"source{suffix}"
    upload.save(source)
    reference_name = ''
    if calibration and calibration['source'] == 'reference':
        reference_name = 'reference' + Path(reference.filename).suffix.lower()
        reference.save(output_dir / reference_name)
    with jobs_lock:
        jobs[job_id] = {
            "id": job_id,
            "model": model,
            "calibration": calibration,
            "reference_filename": reference_name,
            "state": "queued",
            "message": "视频已上传，等待开始分析…",
            "filename": upload.filename,
            "source_filename": source.name,
            "created_at": datetime.now().astimezone().isoformat(),
        }
        job_snapshot = dict(jobs[job_id])
    persist_job_metadata(job_id, job_snapshot)
    executor.submit(run_analysis, job_id, source, output_dir)
    return jsonify(id=job_id, status_url=url_for("job_status", job_id=job_id)), 202


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        payload = dict(job)
    return jsonify(payload)


@app.post("/api/jobs/<job_id>/reanalyze")
def reanalyze_job(job_id: str):
    with jobs_lock:
        previous = jobs.get(job_id)
        if previous is None:
            abort(404)
        previous = dict(previous)
    source = source_for_job(previous, JOBS_DIR / job_id)
    if source is None:
        return jsonify(error="原视频已不存在，无法重新分析"), 409

    payload = dict(request.form) if request.form else request.get_json(silent=True) or {}
    if not isinstance(payload, dict):
        return jsonify(error="请求必须是 JSON 对象"), 400
    try:
        model = validate_model(payload.get("model", "rtmpose"))
        setting = payload.get('calibration', previous.get('calibration'))
        if isinstance(setting, str):
            setting = json.loads(setting)
        calibration = validate_calibration(setting)
        reference = request.files.get('reference_video')
        if reference and Path(reference.filename).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise ValueError('站立参考视频仅支持 MP4、MOV 或 AVI')
        previous_reference = JOBS_DIR / job_id / str(previous.get('reference_filename') or 'reference.mp4')
        if calibration and calibration['source'] == 'reference' and not reference and not previous_reference.is_file():
            raise ValueError('站立参考视频已不存在，请重新上传')
    except (ValueError, TypeError) as exc:
        return jsonify(error=str(exc)), 400

    new_id = uuid.uuid4().hex
    output_dir = JOBS_DIR / new_id
    output_dir.mkdir(parents=True, exist_ok=False)
    copied_source = output_dir / f"source{source.suffix.lower()}"
    shutil.copy2(source, copied_source)
    reference_name = ''
    if calibration and calibration['source'] == 'reference':
        suffix = Path(reference.filename).suffix.lower() if reference else previous_reference.suffix.lower()
        reference_name = 'reference' + suffix
        if reference:
            reference.save(output_dir / reference_name)
        else:
            shutil.copy2(previous_reference, output_dir / reference_name)
    new_job = {
        "id": new_id,
        "model": model,
        "calibration": calibration,
        "reference_filename": reference_name,
        "state": "queued",
        "message": "已加入重新分析队列…",
        "filename": previous.get("filename", source.name),
        "source_filename": copied_source.name,
        "created_at": datetime.now().astimezone().isoformat(),
        "reanalyzed_from": job_id,
    }
    with jobs_lock:
        jobs[new_id] = new_job
    persist_job_metadata(new_id, new_job)
    executor.submit(run_analysis, new_id, copied_source, output_dir)
    return jsonify(id=new_id, status_url=url_for("job_status", job_id=new_id)), 202


@app.delete("/api/jobs/<job_id>")
def delete_job(job_id: str):
    """Delete one history entry and every file belonging to that analysis."""
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        if job.get("state") in {"queued", "running"}:
            return jsonify(error="分析进行中，完成后才能删除"), 409
        output_dir = JOBS_DIR / job_id
        jobs.pop(job_id, None)
        try:
            shutil.rmtree(output_dir)
        except FileNotFoundError:
            pass
        except OSError as exc:
            jobs[job_id] = job
            return jsonify(error=f"删除文件失败：{exc}"), 500
    return jsonify(id=job_id, deleted=True)


@app.get("/results/<job_id>/<filename>")
def artifact(job_id: str, filename: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            abort(404)
        allowed = filename in ARTIFACTS or filename in {job.get("source_filename"), job.get("reference_filename")}
    if not allowed:
        abort(404)
    return send_from_directory(JOBS_DIR / job_id, filename, as_attachment=False)


@app.post("/api/realtime/pose")
def realtime_frame():
    if request.content_type != "image/jpeg":
        return jsonify(error="请发送 JPEG 画面"), 415
    encoded = request.get_data(cache=False)
    if not encoded or len(encoded) > 3 * 1024 * 1024:
        return jsonify(error="摄像头画面无效或过大"), 400
    try:
        return jsonify(realtime_pose.process(encoded, request.args.get("model", "rtmpose")))
    except (ValueError, RuntimeError) as exc:
        return jsonify(error=str(exc)), 400


@app.errorhandler(413)
def too_large(_error):
    return jsonify(error="视频超过 2 GB，请压缩后重试"), 413


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="跑步机跑姿分析 Web 应用")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--no-browser", action="store_true", help="启动时不自动打开浏览器")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
