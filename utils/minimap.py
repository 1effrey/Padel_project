"""utils/minimap.py
Top-down (bird's-eye) 2D court view.

Draws a clean schematic of the 10x20 m padel court ONCE, then on each frame stamps
the players' court positions (in meters) onto a copy of it. This is the first
visible payoff of the homography: you can watch the dots move around a real court
map while the players move in the video.

ORIENTATION
  The far baseline (y = 20 m, the end the camera faces) is drawn at the TOP, the
  near baseline (y = 0) at the BOTTOM -- so the minimap is oriented the same way
  the camera looks down the court.

NOTE (single camera)
  This camera only maps its visible ~3/4 well, so dots in the near quarter come
  from extrapolated homography and drift. That near end is the OTHER camera's job;
  Phase-4 fusion will fill it in. We draw the whole court anyway for context.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from utils.homography import COURT_LENGTH_M, COURT_WIDTH_M, LINE_COLORS, court_lines_m

Point = Tuple[float, float]
_FONT = cv2.FONT_HERSHEY_SIMPLEX


class Minimap:
    """Renders the court-from-above and plots player dots on it."""

    def __init__(self, scale_px_per_m: int = 30, margin_px: int = 28) -> None:
        self.scale = scale_px_per_m
        self.margin = margin_px
        self.w = int(COURT_WIDTH_M * self.scale + 2 * self.margin)
        self.h = int(COURT_LENGTH_M * self.scale + 2 * self.margin)
        self._base = self._draw_court()

    # -- meters -> minimap pixels -------------------------------------------
    def m2px(self, point: Point) -> Tuple[int, int]:
        x, y = point
        px = self.margin + x * self.scale
        py = self.margin + (COURT_LENGTH_M - y) * self.scale   # far end at top
        return int(round(px)), int(round(py))

    def _on_canvas(self, point: Point) -> bool:
        """Is this metric point anywhere near the court? Keeps wildly extrapolated
        far/near points (which can map to huge pixels) from being drawn."""
        x, y = point
        return -2.0 <= x <= COURT_WIDTH_M + 2.0 and -2.0 <= y <= COURT_LENGTH_M + 2.0

    def _draw_court(self) -> np.ndarray:
        img = np.full((self.h, self.w, 3), (40, 70, 40), np.uint8)  # dark green
        for _name, p0, p1, kind in court_lines_m():
            t = 3 if kind in ("net", "center") else 2
            cv2.line(img, self.m2px(p0), self.m2px(p1), LINE_COLORS[kind], t, cv2.LINE_AA)
        cv2.putText(img, "TOP-DOWN (m)", (8, 18), _FONT, 0.5, (255, 255, 255), 1)
        return img

    # -- per-frame rendering ------------------------------------------------
    def render(
        self,
        positions: Sequence[Dict[str, Any]],
        color_fn: Optional[Callable[[int], Tuple[int, int, int]]] = None,
    ) -> np.ndarray:
        """Return a fresh minimap image with one dot per player position dict
        (as produced by court_position.player_court_position)."""
        img = self._base.copy()
        for p in positions:
            foot_m = p["foot_m"]
            if not self._on_canvas(foot_m):
                continue
            cx, cy = self.m2px(foot_m)
            tid = p.get("track_id")
            col = color_fn(tid) if (color_fn and tid is not None) else (0, 255, 255)
            cv2.circle(img, (cx, cy), 6, col, -1)
            # white ring normally, red ring if the player is OUTSIDE the court
            ring = (0, 0, 255) if not p.get("inside", True) else (255, 255, 255)
            cv2.circle(img, (cx, cy), 6, ring, 1)
            if tid is not None:
                cv2.putText(img, str(tid), (cx + 8, cy - 8), _FONT, 0.5, col, 1)
        return img

    # -- paste into the main frame -----------------------------------------
    def composite(self, frame: np.ndarray, minimap: np.ndarray,
                  corner: str = "tr", pad: int = 12) -> np.ndarray:
        """Paste the minimap into a corner of `frame` (in place). corner is one of
        'tr','tl','br','bl'. If the frame is somehow smaller than the minimap we
        skip (nothing to paste into)."""
        fh, fw = frame.shape[:2]
        mh, mw = minimap.shape[:2]
        if mh + 2 * pad > fh or mw + 2 * pad > fw:
            return frame
        x0 = pad if "l" in corner else fw - mw - pad
        y0 = pad if "t" in corner else fh - mh - pad
        frame[y0:y0 + mh, x0:x0 + mw] = minimap
        return frame
