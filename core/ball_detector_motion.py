"""core/ball_detector_motion.py
A CLASSICAL, training-free ball detector -- the "works today, no labels" baseline.

WHY THIS EXISTS
  TrackNetV2 (core/ball_detector.py) is the accurate detector, but it needs trained
  weights we do not have yet. This module finds the ball using plain computer vision
  -- background subtraction + blob shape filtering -- so the team can SEE a ball
  being tracked immediately and sanity-check the whole pipeline while the real
  TrackNet weights are being labeled and trained. It is also the project's own
  Phase-3 plan ("ball (MOG2)").

HOW IT WORKS
  The cameras are static, so a background model (MOG2) learns the court/glass/seats
  and flags everything that MOVES as foreground. The ball is a SMALL, roughly ROUND,
  fast blob. We keep foreground blobs whose area is in a ball-sized range and whose
  shape is round enough, optionally inside the court polygon, and report the most
  ball-like one.

HONEST LIMITS (this is a baseline, not the final detector)
  * It picks up ANY motion: a player's hand, a racket, a shadow, the swaying net.
  * Heavy motion blur smears the ball into a streak (low circularity) -> may miss it.
  * A momentarily still ball blends into the background -> missed.
  * It reports at most ONE ball and has no temporal memory (that is Phase 2's job).
  Treat its output as noisy candidates, good for visualization and for bootstrapping
  labels -- not as ground truth.

INTERFACE
  Identical to BallDetector: .detect(frame) -> BallDetection, .reset(), plus
  .operational (always True here) and .mode_label, so it is a drop-in alternative
  selected by config["ball"]["method"] == "motion".
"""
from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from core.ball_detector import BallDetection


class MotionBallDetector:
    """Background-subtraction ball detector. Stateful (MOG2 learns over time)."""

    def __init__(
        self,
        history: int = 200,
        var_threshold: float = 25.0,
        min_area: float = 8.0,
        max_area: float = 1500.0,
        min_circularity: float = 0.35,
        morph_kernel: int = 3,
        warmup_frames: int = 15,
        court_polygon: Optional[np.ndarray] = None,
        roi_margin_px: float = 0.0,
    ) -> None:
        self._history = int(history)
        self._var_threshold = float(var_threshold)
        self.min_area = float(min_area)
        self.max_area = float(max_area)
        self.min_circularity = float(min_circularity)
        self.warmup_frames = int(warmup_frames)
        self.court_polygon = court_polygon
        self.roi_margin_px = float(roi_margin_px)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (max(1, int(morph_kernel)), max(1, int(morph_kernel))))

        # interface parity with BallDetector (used by the eval harness)
        self.operational = True            # works without any weights
        self.mode_label = "motion"
        self.device = "cpu"                # OpenCV/CPU
        self.last_candidates: list = []    # for the tracker's motion-based selection

        self._bg = self._new_bg()
        self._n = 0

    # ------------------------------------------------------------------ public
    def detect(self, frame: np.ndarray) -> BallDetection:
        """Find the most ball-like moving blob in this BGR frame."""
        self._n += 1
        fg = self._bg.apply(frame)         # 0 / 255 (shadows disabled)

        # MOG2 needs a few frames to learn the background; before that the whole
        # frame reads as foreground, so don't pretend we can localize a ball yet.
        if self._n <= self.warmup_frames:
            self.last_candidates = []
            return BallDetection(found=False, reason="warmup")

        _, mask = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)   # drop specks
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best: Optional[tuple] = None       # (cx, cy, circularity)
        best_score = 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue                    # too small (noise) or too big (player)
            perim = cv2.arcLength(c, True)
            if perim <= 0:
                continue
            circularity = 4.0 * np.pi * area / (perim * perim)   # 1.0 == perfect circle
            if circularity < self.min_circularity:
                continue                    # too elongated to be a ball
            m = cv2.moments(c)
            if m["m00"] == 0:
                continue
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
            if self.court_polygon is not None:
                dist = cv2.pointPolygonTest(self.court_polygon, (float(cx), float(cy)), True)
                if dist < -self.roi_margin_px:
                    continue                # outside the court (+margin) -> reject
            if circularity > best_score:    # prefer the roundest candidate
                best_score = circularity
                best = (cx, cy, circularity)

        if best is None:
            self.last_candidates = []
            return BallDetection(found=False, reason="no-ball")
        cx, cy, circularity = best
        # circularity (capped at 1.0) doubles as a crude confidence
        det = BallDetection(found=True, u=float(cx), v=float(cy),
                            confidence=float(min(1.0, circularity)), reason="ok")
        self.last_candidates = [det]
        return det

    def reset(self) -> None:
        """Forget the learned background (call between independent clips)."""
        self._bg = self._new_bg()
        self._n = 0

    # ----------------------------------------------------------------- private
    def _new_bg(self) -> Any:
        return cv2.createBackgroundSubtractorMOG2(
            history=self._history, varThreshold=self._var_threshold,
            detectShadows=False)
