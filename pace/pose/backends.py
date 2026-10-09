"""RTMPose inference with a shared normalized landmark layout."""
from __future__ import annotations

import threading

import numpy as np

from pace.pose.landmarks import PoseFrame

MODEL_NAMES = {"rtmpose": "RTMPose"}
# COCO 17 -> shared 33-slot layout. Missing toes, heels, hands and face detail stay missing.
COCO_TO_LANDMARKS = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
_RTMPOSE_LOAD_LOCK = threading.Lock()


def validate_model(model: str) -> str:
    if not isinstance(model, str) or model not in MODEL_NAMES:
        raise ValueError("当前仅支持 RTMPose")
    return model


def coco_landmarks(xy: np.ndarray, scores: np.ndarray) -> PoseFrame:
    points = np.full((33, 4), np.nan, dtype=np.float64)
    points[:, 3] = 0
    points[COCO_TO_LANDMARKS, :2] = xy
    # COCO models are 2D; z=0 is only a placeholder, never a depth measurement.
    points[COCO_TO_LANDMARKS, 2] = 0
    points[COCO_TO_LANDMARKS, 3] = np.clip(scores, 0, 1)
    if np.count_nonzero(scores[5:] >= 0.3) < 4:
        return PoseFrame(None)
    return PoseFrame(points)


class RTMPoseEstimator:
    def __init__(self) -> None:
        try:
            from rtmlib import Body
        except ImportError as exc:
            raise RuntimeError("RTMPose 依赖缺失，请运行 pip install -r requirements.txt") from exc
        try:
            with _RTMPOSE_LOAD_LOCK:
                self._body = Body(mode="lightweight", to_openpose=False,
                                  backend="onnxruntime", device="cpu")
        except Exception as exc:
            raise RuntimeError(f"RTMPose 加载失败（首次使用需要联网下载模型）：{exc}") from exc

    def process(self, frame: np.ndarray) -> PoseFrame:
        keypoints, scores = self._body(frame)
        if len(keypoints) == 0:
            return PoseFrame(None)
        # This demo analyzes one runner: select the most confident body.
        index = int(np.argmax(np.mean(scores[:, 5:], axis=1)))
        height, width = frame.shape[:2]
        return coco_landmarks(keypoints[index] / [width, height], scores[index])

    def close(self) -> None:
        self._body = None

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def create_estimator(model: str = "rtmpose"):
    validate_model(model)
    return RTMPoseEstimator()
