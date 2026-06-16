"""utils/player_keypoint_log.py
Write ONE CSV per player (P1..P4) with the full per-frame detection: bounding box,
confidence, foot point (image + court), and all 17 COCO keypoints.

Each run creates four NEW timestamped files (never overwrites):
    output/player1_<run>_<YYYYMMDD_HHMMSS>.csv
    output/player2_<run>_<YYYYMMDD_HHMMSS>.csv
    ... (3, 4)

Columns (one row per detection of that player):
    player_id, source_video, frame, track_id,
    bbox_x1, bbox_y1, bbox_x2, bbox_y2, bbox_conf,
    foot_img_x, foot_img_y,            # chosen foot point, image pixels
    foot_canvas_x, foot_canvas_y,      # foot on the court, METRES
    kp0_x, kp0_y, kp0_c, ... kp16_x, kp16_y, kp16_c   # COCO-17 keypoints (image px + conf)

Routing is by the STABLE player id (1..4) from the ReID layer; detections without a
1..4 id are skipped (they have no player file to go in). In fusion a player seen by
both cameras produces two rows that frame -- the source_video column says which camera.
"""
from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Dict, Sequence

from utils.movement_log import _slug   # same filename-stem helper

_N_KP = 17  # COCO-17 keypoints (kp0..kp16)
_KP_COLS = [f"kp{i}_{c}" for i in range(_N_KP) for c in ("x", "y", "c")]
HEADER = (["player_id", "source_video", "frame", "track_id",
           "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "bbox_conf",
           "foot_img_x", "foot_img_y", "foot_canvas_x", "foot_canvas_y"] + _KP_COLS)


class PlayerKeypointWriter:
    """Holds four CSV writers (one per player id) for a single run."""

    def __init__(self, out_dir: str, run_tag: Any,
                 player_ids: Sequence[int] = (1, 2, 3, 4)) -> None:
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._files: Dict[int, Any] = {}
        self._writers: Dict[int, Any] = {}
        self._paths: Dict[int, str] = {}
        self.counts: Dict[int, int] = {}
        for pid in player_ids:
            path = os.path.join(out_dir, f"player{pid}_{_slug(run_tag)}_{stamp}.csv")
            f = open(path, "w", newline="")            # newline="" -> no blank rows
            w = csv.writer(f)
            w.writerow(HEADER)
            self._files[pid] = f
            self._writers[pid] = w
            self._paths[pid] = path
            self.counts[pid] = 0

    def add(self, player_id: Any, source_video: Any, frame: int, track_id: Any,
            bbox: Sequence[float], conf: float,
            foot_img: Sequence[float], foot_canvas: Sequence[float],
            keypoints: Any) -> None:
        """Append one detection row to the given player's file. No-op if the player
        id isn't one of the four (e.g. ReID left it unassigned)."""
        w = self._writers.get(player_id)
        if w is None:
            return
        x1, y1, x2, y2 = (round(float(v), 1) for v in bbox)
        row = [player_id, source_video, frame, ("" if track_id is None else track_id),
               x1, y1, x2, y2, round(float(conf), 4),
               round(float(foot_img[0]), 1), round(float(foot_img[1]), 1),
               round(float(foot_canvas[0]), 3), round(float(foot_canvas[1]), 3)]
        for i in range(_N_KP):
            kx, ky, kc = keypoints[i]
            row += [round(float(kx), 1), round(float(ky), 1), round(float(kc), 3)]
        w.writerow(row)
        self.counts[player_id] += 1

    def close(self) -> None:
        """Flush + close all four files and report the row counts."""
        for f in self._files.values():
            f.close()
        per = ", ".join(f"P{pid}={self.counts[pid]}" for pid in sorted(self.counts))
        print(f"[keypoints] wrote per-player CSVs ({per}) -> {os.path.dirname(next(iter(self._paths.values())))}")
        self._files.clear()
