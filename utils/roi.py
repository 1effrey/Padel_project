"""utils/roi.py
Court Region-Of-Interest filter -- PURE GEOMETRY, no AI.

WHY (architecture decision #4)
  The camera also sees spectators behind the glass, parked cars and chairs.
  Any detection whose BOX-CENTER lands outside the court polygon is discarded
  BEFORE tracking. We use cv2.pointPolygonTest -- plain geometry, no model.

The polygon is NEVER hard-coded: it lives in config.json["court"]["polygon"]
and is defined once per camera with `python main.py --calibrate-roi`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


def to_polygon(points: Optional[list]) -> Optional[np.ndarray]:
    """Convert a [[x, y], ...] list from config into an int32 array for OpenCV.

    Returns None when fewer than 3 points are configured -> the filter then
    becomes a harmless pass-through (so the pipeline still runs before the
    court is calibrated).
    """
    if not points or len(points) < 3:
        return None
    return np.array(points, dtype=np.int32)


def box_center(bbox: np.ndarray) -> Tuple[float, float]:
    """Geometric center of [x1, y1, x2, y2] -- the point we test against the
    polygon (the brief specifies box-center)."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def is_inside(polygon: np.ndarray, point: Tuple[float, float]) -> bool:
    """True if point is inside (or on) the polygon.

    cv2.pointPolygonTest with measureDist=False returns:
        +1 inside, 0 on the edge, -1 outside.
    """
    return cv2.pointPolygonTest(polygon, (float(point[0]), float(point[1])), False) >= 0


def filter_detections(
    detections: List[Dict[str, Any]],
    polygon: Optional[np.ndarray],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split detections into (kept_inside_court, removed_outside_court).

    If no polygon is configured we keep everything and `removed` is empty, so
    the rest of the pipeline behaves identically before/after calibration.
    """
    if polygon is None:
        return list(detections), []

    kept: List[Dict[str, Any]] = []
    removed: List[Dict[str, Any]] = []
    for det in detections:
        if is_inside(polygon, box_center(det["bbox"])):
            kept.append(det)
        else:
            removed.append(det)
    return kept, removed
