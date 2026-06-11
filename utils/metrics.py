"""utils/metrics.py
Lightweight measurement for the Phase-1 quality gate.

Architecture decision #5: "Measure failures, don't hide them." After building,
we must be able to put NUMBERS on quality. None of these need ground-truth
labels -- they are proxies we can compute on any clip:

  - detections before / after the ROI filter  (how much spectator noise we cut)
  - per-frame mean detection confidence
  - bbox heights in pixels                     (near players big, far players small)
  - unique track_ids seen
  - an ID-CHURN proxy for swaps: every time a BRAND-NEW id appears after the
    first frame it is counted. On a court that should hold <= 4 players, churn
    far above 4 hints at id swaps / track fragmentation worth investigating.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Set

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Convert numpy scalars/arrays to native Python before json.dump
    (coding convention in the brief)."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class Metrics:
    """Accumulates per-frame numbers and produces a JSON-able summary."""

    def __init__(self) -> None:
        self.frames = 0
        self.total_raw = 0
        self.total_after_roi = 0
        self.removed_by_roi = 0
        self.conf_sum = 0.0
        self.conf_n = 0
        self.bbox_heights: List[float] = []
        self.unique_ids: Set[int] = set()
        self.new_id_churn = 0           # swap / fragmentation proxy
        self.tracks_per_frame: List[int] = []
        self._seen_any = False

    def update(self, raw: int, after_roi: int, tracked: List[Dict[str, Any]]) -> None:
        self.frames += 1
        self.total_raw += raw
        self.total_after_roi += after_roi
        self.removed_by_roi += (raw - after_roi)
        self.tracks_per_frame.append(len(tracked))

        for det in tracked:
            self.conf_sum += det["conf"]
            self.conf_n += 1
            x1, y1, x2, y2 = det["bbox"]
            self.bbox_heights.append(float(y2 - y1))
            tid = det["track_id"]
            if tid not in self.unique_ids:
                if self._seen_any:        # a new id appearing mid-clip = churn
                    self.new_id_churn += 1
                self.unique_ids.add(tid)
        self._seen_any = True

    def summary(self) -> Dict[str, Any]:
        heights = np.array(self.bbox_heights) if self.bbox_heights else np.array([0.0])
        return {
            "frames_processed": self.frames,
            "detections_raw_total": self.total_raw,
            "detections_after_roi_total": self.total_after_roi,
            "detections_removed_by_roi": self.removed_by_roi,
            "roi_removed_pct": round(100.0 * self.removed_by_roi / max(1, self.total_raw), 2),
            "mean_confidence": round(self.conf_sum / max(1, self.conf_n), 4),
            "unique_track_ids": len(self.unique_ids),
            "id_churn_proxy": self.new_id_churn,
            "avg_tracks_per_frame": round(
                float(np.mean(self.tracks_per_frame)) if self.tracks_per_frame else 0.0, 2
            ),
            "bbox_height_px": {
                "min": round(float(heights.min()), 1),
                "median": round(float(np.median(heights)), 1),
                "max": round(float(heights.max()), 1),
            },
        }

    def save(self, path: str, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        data = self.summary()
        if extra:
            data.update(extra)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        return data
