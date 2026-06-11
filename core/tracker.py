"""core/tracker.py
ByteTrack wrapper (via the `supervision` library).

WHAT THIS DOES
  Takes the ROI-filtered list of detection dicts and gives each one a STABLE
  integer `track_id` that persists across frames. Returns the same dicts with
  an added "track_id" key.

HOW (and the one subtlety to understand)
  ByteTrack matches BOXES across frames -- it knows nothing about keypoints.
  So we:
    1. convert our dict list -> sv.Detections (boxes + confidence),
    2. stash each detection's original index in detections.data,
    3. call update_with_detections(), which returns boxes WITH tracker_ids,
    4. use the stashed index to re-attach the correct keypoints to each id.
  Step 2-4 matter because ByteTrack may DROP or REORDER detections, so we can't
  assume the output lines up 1:1 with the input.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import supervision as sv


class PlayerTracker:
    """Stateful ByteTrack wrapper. Create one per video stream."""

    def __init__(
        self,
        track_activation_threshold: float = 0.25,
        lost_track_buffer: int = 30,
        minimum_matching_threshold: float = 0.8,
        frame_rate: int = 30,
        minimum_consecutive_frames: int = 1,
    ) -> None:
        # lost_track_buffer = how many frames a track is REMEMBERED after it
        # disappears, so a player who is briefly occluded keeps the same id.
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_activation_threshold,
            lost_track_buffer=lost_track_buffer,
            minimum_matching_threshold=minimum_matching_threshold,
            frame_rate=frame_rate,
            minimum_consecutive_frames=minimum_consecutive_frames,
        )

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Advance the tracker by one frame and return dicts with track_id."""
        if len(detections) == 0:
            # Still tick the tracker so lost-track buffers age correctly even on
            # empty frames (otherwise occluded players never time out).
            self.tracker.update_with_detections(sv.Detections.empty())
            return []

        xyxy = np.array([d["bbox"] for d in detections], dtype=float)
        conf = np.array([d["conf"] for d in detections], dtype=float)
        class_id = np.zeros(len(detections), dtype=int)
        orig_idx = np.arange(len(detections))  # carried through to re-link kpts

        sv_dets = sv.Detections(
            xyxy=xyxy,
            confidence=conf,
            class_id=class_id,
            data={"orig_idx": orig_idx},
        )

        tracked = self.tracker.update_with_detections(sv_dets)

        out: List[Dict[str, Any]] = []
        for i in range(len(tracked)):
            tid = tracked.tracker_id[i]
            if tid is None:
                continue  # not yet a confirmed track
            orig = int(tracked.data["orig_idx"][i])
            det = dict(detections[orig])  # shallow copy keeps bbox/conf/keypoints
            det["track_id"] = int(tid)
            out.append(det)
        return out
