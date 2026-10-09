"""Shared landmark layout for analysis, reports and pose overlays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer", "left_ear",
    "right_ear", "mouth_left", "mouth_right", "left_shoulder",
    "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
    "right_wrist", "left_pinky", "right_pinky", "left_index",
    "right_index", "left_thumb", "right_thumb", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel",
    "right_heel", "left_foot_index", "right_foot_index",
]


@dataclass
class PoseFrame:
    landmarks: Optional[np.ndarray]
    # Array shape: (33, 4), columns x, y, z, visibility.


def landmarks_to_dict(points: np.ndarray) -> Dict[str, np.ndarray]:
    return {name: points[i] for i, name in enumerate(LANDMARK_NAMES)}

