"""utils/movement_log.py
Write every player's court-position track to a timestamped CSV -- one new file per run.

Each run creates a NEW file (never overwrites):
    output/movement_<source>_<YYYYMMDD_HHMMSS>.csv

Tidy / "long" format -- one row per player per frame:
    frame, time_s, player_id, x_m, y_m

x_m / y_m are COURT METRES (via the homography), so this is the players' real movement
on the court -- ready to open in Excel, pivot, or plot. Rows are only written for
detections that have a court position AND a stable id; players off-court or unassigned
that frame are skipped. Requires a homography (no metres -> no movement to log).
"""
from __future__ import annotations

import csv
import os
import re
from datetime import datetime
from typing import Any, Dict, List


def _slug(source: Any) -> str:
    """Turn a video path / camera index into a safe filename stem,
    e.g. 'side-1-full-vid.mp4' -> 'side-1-full-vid'."""
    stem = os.path.splitext(os.path.basename(str(source)))[0]
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "camera"


class MovementWriter:
    """Accumulates per-frame player court positions into one timestamped CSV."""

    def __init__(self, out_dir: str, source: Any, fps: float) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.fps = float(fps) if fps and fps > 0 else 0.0
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(out_dir, f"movement_{_slug(source)}_{stamp}.csv")
        # newline="" is the documented way to avoid blank rows from csv on Windows
        self._f = open(self.path, "w", newline="")
        self._writer = csv.writer(self._f)
        self._writer.writerow(["frame", "time_s", "player_id", "x_m", "y_m"])
        self.rows = 0

    def add(self, frame_idx: int, positions: List[Dict[str, Any]]) -> None:
        """Append one row per player that has a court position this frame.

        Each item in `positions` is expected to carry:
            "foot_m"   -> [x_m, y_m]  (court metres), and
            "track_id" -> the stable player id (1..4 when ReID is on).
        Items missing either are skipped.
        """
        time_s = round(frame_idx / self.fps, 3) if self.fps > 0 else ""
        for p in positions:
            pid = p.get("track_id")
            foot = p.get("foot_m")
            if pid is None or foot is None:
                continue
            self._writer.writerow(
                [frame_idx, time_s, pid, round(float(foot[0]), 3), round(float(foot[1]), 3)])
            self.rows += 1

    def close(self) -> None:
        """Flush + close the file and report where it landed."""
        if self._f is not None:
            self._f.close()
            self._f = None
            print(f"[movement] wrote {self.rows} player-position rows -> {self.path}")
