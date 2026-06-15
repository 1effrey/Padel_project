"""utils/court_position.py
Turn a tracked player into a position ON THE COURT, in meters.

WHY FEET (and not the box center)
  The homography is only valid for points that lie on the floor plane. A player's
  FEET are on the floor, so projecting the feet with pixel_to_meters() gives a real
  court coordinate. The box center is up around the hips/chest -- above the plane --
  and would map to a point further away than the player actually stands. So we
  deliberately pick a foot point.

FOOT POINT CHOICE (COCO-17 keypoints; see utils/skeleton.py for the index list)
  1. If BOTH ankles (15, 16) are confident -> midpoint of the two ankles.
  2. Else if ONE ankle is confident          -> that ankle.
  3. Else (ankles hidden/occluded)           -> bottom-center of the bbox.
  The fallback keeps far/occluded players usable, just less precise -- which is
  expected on the far side (architecture decision #2).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from utils.homography import Homography, is_inside_court

L_ANKLE, R_ANKLE = 15, 16
Point = Tuple[float, float]


def foot_pixel(
    keypoints: np.ndarray,
    bbox: np.ndarray,
    kp_conf_threshold: float = 0.5,
) -> Tuple[Point, str]:
    """Return (foot_point_px, source) where source is 'ankles' | 'ankle' | 'bbox'.

    `source` is handed back so callers/logs can see how reliable the point is."""
    la = keypoints[L_ANKLE]
    ra = keypoints[R_ANKLE]
    la_ok = float(la[2]) >= kp_conf_threshold
    ra_ok = float(ra[2]) >= kp_conf_threshold

    if la_ok and ra_ok:
        return (float((la[0] + ra[0]) / 2.0), float((la[1] + ra[1]) / 2.0)), "ankles"
    if la_ok:
        return (float(la[0]), float(la[1])), "ankle"
    if ra_ok:
        return (float(ra[0]), float(ra[1])), "ankle"
    # fallback: bottom-center of the box (feet are at the bottom of the person)
    x1, y1, x2, y2 = (float(v) for v in bbox)
    return ((x1 + x2) / 2.0, y2), "bbox"


def player_court_position(
    det: Dict[str, Any],
    homog: Homography,
    kp_conf_threshold: float = 0.5,
    inside_margin_m: float = 0.0,
) -> Dict[str, Any]:
    """Project one tracked detection's feet to court meters.

    Returns a plain dict (JSON-friendly):
        track_id, foot_px (x,y), foot_m (x,y), foot_source, inside (bool)
    """
    foot_px, source = foot_pixel(det["keypoints"], det["bbox"], kp_conf_threshold)
    foot_m = homog.pixel_to_meters(foot_px)
    return {
        "track_id": det.get("track_id"),
        "foot_px": [round(foot_px[0], 1), round(foot_px[1], 1)],
        "foot_m": [round(foot_m[0], 3), round(foot_m[1], 3)],
        "foot_source": source,
        "inside": bool(is_inside_court(foot_m, inside_margin_m)),
    }
