"""Offline COCO-to-Human3.6M lifting with the official VideoPose3D model."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import threading
import urllib.request

import numpy as np

from pace.paths import data_dir
from pace.pose.backends import COCO_TO_LANDMARKS
from pace.pose.quality import fill_for_model

MODEL_URL = 'https://dl.fbaipublicfiles.com/video-pose-3d/pretrained_h36m_detectron_coco.bin'
MODEL_SHA256 = 'd3219e005b50591f694da5cbaf6849f060d6b2cf895864a779f8a992ac63a232'
H36M_NAMES = ['pelvis', 'right_hip', 'right_knee', 'right_ankle', 'left_hip',
             'left_knee', 'left_ankle', 'spine', 'thorax', 'neck', 'head',
             'left_shoulder', 'left_elbow', 'left_wrist', 'right_shoulder',
             'right_elbow', 'right_wrist']
_lock = threading.Lock()
_model = None


def model_path() -> Path:
    root = data_dir()
    path = root / 'models' / 'pretrained_h36m_detectron_coco.bin'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temporary = None
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        handle.write(block)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256:
                raise RuntimeError('三维模型校验失败')
            temporary.replace(path)
        except Exception as exc:
            if temporary:
                temporary.unlink(missing_ok=True)
            raise RuntimeError('VideoPose3D 权重下载失败，请检查网络后重新分析') from exc
    if hashlib.sha256(path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError('VideoPose3D 权重校验失败，请删除损坏的缓存后重试')
    return path


def get_model():
    global _model
    with _lock:
        if _model is None:
            try:
                import torch
                from pace.pose.vendor.videopose3d import TemporalModel
            except ImportError as exc:
                raise RuntimeError('三维估计需要 PyTorch，请安装 requirements.txt 中的依赖') from exc
            torch.set_num_threads(2)
            model = TemporalModel(17, 2, 17, [3, 3, 3, 3, 3], channels=1024)
            # Only the SHA256-pinned, official checkpoint is loaded using pickle.
            checkpoint = torch.load(str(model_path()), map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['model_pos'], strict=True)
            _model = model.eval()
        return _model


def lift_pose(points: np.ndarray, timestamps: np.ndarray, aspect: float,
              min_visibility: float = .45) -> np.ndarray:
    """Return root-relative camera-space 3D, in the model's learned units.

    Model input is COCO order, normalized by image width, as in upstream.
    Resample to 50 Hz and return to original timestamps; do not invent a global root.
    """
    output = temporal_lift(points, timestamps, aspect, min_visibility, get_model(),
                           50, 17, [4, 5, 6, 11, 12, 13], [1, 2, 3, 14, 15, 16])
    output -= (output[:, 1:2] + output[:, 4:5]) / 2
    return output


def temporal_lift(points, timestamps, aspect, min_visibility, model, fps,
                  joint_count, left_out, right_out):
    """Shared official screen normalization and temporal/flip inference."""
    import torch
    times = np.asarray(timestamps, dtype=float)
    if len(times) < 3 or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError('三维估计至少需要 3 帧及严格递增的时间戳')
    xy = points[:, COCO_TO_LANDMARKS, :2].copy()
    confidence = points[:, COCO_TO_LANDMARKS, 3]
    # Face confidence is often lower in profile; body quality is gated separately.
    thresholds = np.full(17, min_visibility); thresholds[:5] = .1
    xy[(confidence < thresholds) | ~np.isfinite(confidence)] = np.nan
    if not np.isfinite(aspect) or aspect <= 0:
        raise ValueError('视频宽高比无效')
    grid = np.arange(times[0], times[-1] + .0001, 1 / fps)
    inputs = np.empty((len(grid), 17, 2), dtype=np.float32)
    for joint in range(17):
        for axis in range(2):
            filled = fill_for_model(xy[:, joint, axis],times)
            inputs[:, joint, axis] = np.interp(grid,times,filled)
    inputs[:, :, 0] = inputs[:, :, 0] * 2 - 1
    inputs[:, :, 1] = inputs[:, :, 1] * 2 / aspect - 1 / aspect
    pad = (model.receptive_field() - 1) // 2
    padded = np.pad(inputs, ((pad, pad), (0, 0), (0, 0)), mode='edge')
    results = []
    left_in, right_in = [1, 3, 5, 7, 9, 11, 13, 15], [2, 4, 6, 8, 10, 12, 14, 16]
    with torch.inference_mode():
        for start in range(0, len(grid), 128):
            end = min(start + 128, len(grid))
            chunk = padded[start:end + 2 * pad].copy()
            flipped = chunk.copy()
            flipped[:, :, 0] *= -1
            flipped[:, left_in + right_in] = flipped[:, right_in + left_in]
            batch = torch.from_numpy(np.stack([chunk, flipped]))
            pred = model(batch).numpy()
            pred[1, :, :, 0] *= -1
            pred[1, :, left_out + right_out] = pred[1, :, right_out + left_out].copy()
            results.append(pred.mean(axis=0))
    lifted = np.concatenate(results)
    output = np.empty((len(times), joint_count, 3))
    for joint in range(joint_count):
        for axis in range(3):
            output[:, joint, axis] = np.interp(times, grid, lifted[:, joint, axis])
    if not np.isfinite(output).all():
        raise ValueError('三维模型输出无效')
    return output
