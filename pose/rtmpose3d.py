"""RTMPose regions -> RTMW3D image inference -> pelvis-relative 3D playback.

Decodes metric relative depth, restores image xy, then backprojects with an
explicit assumed camera. Raw SimCC bins are not equal physical axis units.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import urllib.request

import cv2
import numpy as np

from pose.backends import COCO_TO_LANDMARKS

MODEL_URL = ('https://huggingface.co/Soykaf/RTMW3D-x/resolve/'
             'a4f0d54c52ee44ee4e1dda1936d714490c6234d2/onnx/'
             'rtmw3d-x_8xb64_cocktail14-384x288-b0a0eab7_20240626.onnx')
MODEL_SHA256 = '4a289c0e99d47eb595e99679d9d4a2d1def1b4241f9adcbafba44b9ff585ebcd'
JOINT_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
               'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
               'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
               'left_knee', 'right_knee', 'left_ankle', 'right_ankle',
               'left_big_toe', 'left_small_toe', 'left_heel',
               'right_big_toe', 'right_small_toe', 'right_heel', 'pelvis', 'neck']
EDGES = [[23, 24], [24, 0], [0, 1], [0, 2], [1, 3], [2, 4],
         [24, 5], [24, 6], [5, 7], [7, 9], [6, 8], [8, 10],
         [23, 11], [23, 12], [11, 13], [13, 15], [12, 14], [14, 16],
         [15, 17], [17, 18], [18, 19], [19, 15],
         [16, 20], [20, 21], [21, 22], [22, 16]]
LIMITATION = ('RTMPose 定位人体区域，RTMW3D 直接估计三维骨架；无需站立标定。'
              '解码相对深度并按假设相机反投影，髋中心固定，使用全片固定腿长尺度。'
              '相机焦距与根深度为先验，不是标定后的米制测量，不能恢复真实地面或全局位移。')
ROOT_DEPTH = 5.14388
Z_RANGE = 2.1744869
_LOCK = threading.Lock()
_MODEL = None


def _digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def model_path():
    root = Path(os.environ.get('PACE_DATA_DIR', Path(__file__).resolve().parents[1] / 'output'))
    path = root / 'models' / 'rtmw3d-x.onnx'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and _digest(path) == MODEL_SHA256:
        return path
    partial = path.with_suffix('.onnx.download')
    print('正在下载 RTMW3D 模型（约 352 MiB，首次使用）…', flush=True)
    try:
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response, partial.open('wb') as handle:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
        if _digest(partial) != MODEL_SHA256:
            raise RuntimeError('RTMW3D 模型校验失败，请重试')
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)
    return path


class RTMPose3DEstimator:
    def __init__(self):
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path()), sess_options=options,
                                           providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name

    def process(self, image, bbox):
        from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs, top_down_affine
        center, scale = bbox_xyxy2cs(np.asarray(bbox, dtype=np.float32), padding=1.25)
        crop, scale = top_down_affine((288, 384), scale, center, image)
        crop = (crop.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
        batch = np.ascontiguousarray(crop.transpose(2, 0, 1)[None], dtype=np.float32)
        outputs = self.session.run(None, {self.input_name: batch})
        return decode_simcc(outputs, center, scale, (image.shape[1], image.shape[0]))


def decode_simcc(outputs, center=(144,192), scale=(288,384), image_size=(288,384)):
    if len(outputs) != 3 or any(x.ndim != 3 or x.shape[0] != 1 or x.shape[1] < 23 for x in outputs):
        raise ValueError('RTMW3D 模型输出格式不匹配')
    locs = np.stack([np.argmax(x[0, :23], axis=-1) for x in outputs], axis=-1) / 2.0
    # Codec input_size=(288,384,288): depth bins encode metres, not pixels.
    relative_depth = (locs[:,2] / (outputs[2].shape[-1]/4) - 1) * Z_RANGE
    image_xy = locs[:,:2] / [288,384] * scale + np.asarray(center) - .5*np.asarray(scale)
    # Official camera-space backprojection, with resolution-scaled default f.
    # These are explicit priors, not the runner's calibrated camera parameters.
    focal = np.asarray([1145.04940459,1143.78109572]) * max(image_size) / 1000
    depth = ROOT_DEPTH + relative_depth
    xyz = np.column_stack([(image_xy-np.asarray(image_size)/2)/focal*depth[:,None],depth])
    scores = np.minimum(outputs[0][0, :23].max(axis=-1), outputs[1][0, :23].max(axis=-1))
    xyz = np.vstack([xyz, xyz[[11, 12]].mean(axis=0), xyz[[5, 6]].mean(axis=0)])
    scores = np.r_[scores, scores[[11, 12]].min(), scores[[5, 6]].min()]
    xyz = (xyz - xyz[23]) * [1, -1, -1]  # x right, y up, z towards camera
    legs = [np.linalg.norm(xyz[h] - xyz[k]) + np.linalg.norm(xyz[k] - xyz[a])
            for h, k, a in [(11, 13, 15), (12, 14, 16)]
            if np.min(scores[[h, k, a]]) >= .3]
    if not legs:
        return None
    leg = float(np.mean(legs))
    if leg < .05 or not np.isfinite(leg) or scores[23] < .3:
        return None
    valid = np.isfinite(xyz).all(axis=1) & (scores >= .3)
    valid[:23] &= (outputs[2][0,:23].max(axis=-1) > 0)
    valid[23] = valid[[11,12]].all()
    valid[24] = valid[[5,6]].all()
    if not valid[23]:
        return None
    xyz[~valid] = np.nan
    return xyz, np.clip(scores, 0, 1)


def runner_bbox(points, width, height):
    coco = points[COCO_TO_LANDMARKS]
    valid = np.isfinite(coco[:, :2]).all(axis=1) & (coco[:, 3] >= .3)
    if valid.sum() < 8 or not valid[[11, 12]].all():
        return None
    xy = coco[valid, :2] * [width, height]
    lower, upper = xy.min(axis=0), xy.max(axis=0)
    extent = upper - lower
    if np.min(extent) < 5:
        return None
    lower = np.maximum(lower - extent * .12, 0)
    upper = np.minimum(upper + extent * .12, [width, height])
    return np.r_[lower, upper]


def estimate_skeleton(video, points, timestamps, output):
    """Sample the original video, preserving sample timestamps and missing poses."""
    global _MODEL
    if len(points) == 0 or len(points) != len(timestamps) or not np.isfinite(timestamps).all():
        raise ValueError('三维分析的骨架和时间戳不匹配')
    target_fps = float(os.environ.get('PACE_POSE3D_FPS', '30'))
    if not np.isfinite(target_fps) or not 1 <= target_fps <= 60:
        raise ValueError('PACE_POSE3D_FPS 必须在 1–60 范围内')
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError('无法读取三维分析视频')
    xyz, confidence, times, indices = [], [], [], []
    next_time = 0.0
    try:
        with _LOCK:
            if _MODEL is None:
                _MODEL = RTMPose3DEstimator()
            index = 0
            while index < len(points):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError('三维视频解码提前结束')
                time = float(timestamps[index])
                if time + 1e-6 >= next_time or index == len(points) - 1:
                    bbox = runner_bbox(points[index], frame.shape[1], frame.shape[0])
                    result = _MODEL.process(frame, bbox) if bbox is not None else None
                    coordinates, scores = result if result is not None else (np.full((25, 3), np.nan), np.zeros(25))
                    xyz.append(coordinates); confidence.append(scores); times.append(time); indices.append(index)
                    next_time = time + 1 / target_fps - 1e-6
                    if len(times) % 50 == 0:
                        print(f'RTMW3D 已处理 {len(times)} 个三维骨架采样…', flush=True)
                index += 1
    finally:
        capture.release()
    xyz, confidence = np.asarray(xyz), np.asarray(confidence)
    valid = np.isfinite(xyz[:, 23]).all(axis=1)
    if not valid.any():
        raise RuntimeError('RTMW3D 未得到有效三维骨架，请使用全身可见的单人视频')
    lengths = []
    for h,k,a in [(11,13,15),(12,14,16)]:
        length = np.linalg.norm(xyz[:,h]-xyz[:,k],axis=1)+np.linalg.norm(xyz[:,k]-xyz[:,a],axis=1)
        lengths.extend(length[np.isfinite(length)].tolist())
    if not lengths:
        raise RuntimeError('三维腿长不可用')
    fixed_leg = float(np.median(lengths))
    xyz /= fixed_leg
    metadata = dict(status='available', model='RTMW3D-X', method='RTMPose region + RTMPose3D/RTMW3D',
                    version=2, units='estimated_leg_length', coordinate_system='camera_pelvis_relative',
                    scale_method='clip_median_leg_length', assumed_leg_length_m=fixed_leg,
                    camera_assumption=dict(root_depth_m=ROOT_DEPTH,focal_at_1000px=[1145.04940459,1143.78109572]),
                    sample_count=len(times), target_fps=target_fps,
                    valid_sample_ratio=round(float(valid.mean()), 3), limitations=LIMITATION)
    payload = dict(metadata, joint_names=JOINT_NAMES, edges=EDGES, timestamps=times,
                   frame_indices=indices, confidence=np.round(confidence, 4).tolist(),
                   keypoints=[[[round(float(v), 5) for v in joint] if np.isfinite(joint).all() else None
                               for joint in frame] for frame in xyz])
    output = Path(output)
    np.savez_compressed(output / 'skeleton3d.npz', keypoints=xyz, confidence=confidence,
                        timestamps=np.asarray(times), frame_indices=np.asarray(indices),
                        joint_names=np.asarray(JOINT_NAMES), edges=np.asarray(EDGES))
    (output / 'skeleton3d.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False,
                                                      separators=(',', ':')), 'utf-8')
    return metadata
