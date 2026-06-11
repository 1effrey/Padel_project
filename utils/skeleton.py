"""utils/skeleton.py
Draw the COCO-17 skeleton on a frame.

COCO-17 keypoint indices (the order YOLO-pose outputs):
   0 nose        1 l_eye       2 r_eye       3 l_ear       4 r_ear
   5 l_shoulder  6 r_shoulder  7 l_elbow     8 r_elbow     9 l_wrist
  10 r_wrist    11 l_hip      12 r_hip      13 l_knee     14 r_knee
  15 l_ankle    16 r_ankle

We only draw a joint / limb when its confidence beats a threshold, so the
weak far-side detections (architecture decision #2: far players are small and
unreliable, and that's EXPECTED) don't render as jittery noise.
"""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# Limb connections -- pairs of keypoint indices that get a line between them.
COCO_SKELETON: list[Tuple[int, int]] = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),   # shoulders + arms
    (5, 11), (6, 12), (11, 12),                # torso
    (11, 13), (13, 15), (12, 14), (14, 16),    # legs
    (0, 1), (0, 2), (1, 3), (2, 4),            # head
]


def draw_skeleton(
    frame: np.ndarray,
    keypoints: np.ndarray,
    color: Tuple[int, int, int],
    kp_conf_threshold: float = 0.5,
) -> None:
    """Draw joints + limbs IN-PLACE. keypoints: (17, 3) -> (x, y, conf)."""
    # Joints first.
    for x, y, c in keypoints:
        if c >= kp_conf_threshold:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)

    # Then limbs -- only if BOTH endpoints are confident.
    for a, b in COCO_SKELETON:
        if keypoints[a, 2] >= kp_conf_threshold and keypoints[b, 2] >= kp_conf_threshold:
            pa = (int(keypoints[a, 0]), int(keypoints[a, 1]))
            pb = (int(keypoints[b, 0]), int(keypoints[b, 1]))
            cv2.line(frame, pa, pb, color, 2)
