"""Annotated-video drawing helpers."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np

from pace.pose.landmarks import LANDMARK_NAMES


CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
]


def draw_pose(frame: np.ndarray, points: np.ndarray, min_visibility: float) -> None:
    h, w = frame.shape[:2]
    def pixel(idx: int) -> Optional[Tuple[int, int]]:
        p = points[idx]
        if not np.all(np.isfinite(p[:2])) or p[3] < min_visibility:
            return None
        return int(p[0] * w), int(p[1] * h)
    for a, b in CONNECTIONS:
        pa, pb = pixel(a), pixel(b)
        if pa and pb:
            cv2.line(frame, pa, pb, (76, 220, 116), 3, cv2.LINE_AA)
    for idx in sorted({i for pair in CONNECTIONS for i in pair}):
        p = pixel(idx)
        if p:
            cv2.circle(frame, p, 4, (40, 200, 255), -1, cv2.LINE_AA)


def draw_panel(frame: np.ndarray, knee_angle: Optional[float], cadence: object,
               event_text: str, progress: float) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (14, 14), (350, 132), (15, 22, 32), -1)
    cv2.addWeighted(overlay, 0.74, frame, 0.26, 0, frame)
    texts = [
        f"Knee angle: {knee_angle:.1f} deg" if knee_angle is not None else "Knee angle: --",
        f"Cadence: {cadence} steps/min" if cadence is not None else "Cadence: --",
        f"Event: {event_text or '--'}",
    ]
    for i, value in enumerate(texts):
        cv2.putText(frame, value, (30, 48 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (245, 248, 250), 2, cv2.LINE_AA)
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, h - 7), (int(w * progress), h), (72, 190, 115), -1)

