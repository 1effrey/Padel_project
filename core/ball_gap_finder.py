"""core/ball_gap_finder.py
TRIANGULATION-GAP finder -- the highest-yield side-2 labels for 3D height.

WHY
  3D height needs the ball seen by BOTH cameras on the SAME synced frame. So the side-2
  frames most worth labeling are the ones where side-1 ALREADY sees the ball clearly but
  side-2 misses it -- label one and it becomes a NEW triangulation point, a depth anchor
  the Phase-5 height is starved for.

  This runs both cameras in lockstep (side-2 frame = side-1 frame + sync offset) and
  lists the side-2 frames where:
       side-1 FOUND the ball (confident)   AND   side-2 found NOTHING
       AND that side-2 frame is not already labeled.
  Output -> a CSV the labeler consumes via `--label-from`, sorted by side-2 frame.

  It's a TARGETED variant of `--mine-hard`: instead of side-2's misses in general, it
  surfaces the misses that would DIRECTLY add 3D coverage. Run:
       python -m core.ball_gap_finder config-side1.json config-side2.json [max_frames]
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Any, Dict, List, Optional

from core.ball_eval import _build_detector
from core.ball_label import _csv_path, _load_existing
from utils.video_io import ThreadedVideoReader


def run_gap_finder(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any],
                   max_frames: Optional[int] = None,
                   side1_conf_min: float = 0.5) -> str:
    """List side-2 frames where side-1 sees the ball but side-2 misses (and which are not
    already labeled). Writes output/triangulation_gaps_<side2>.csv; returns its path."""
    out_dir = cfg_a.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)

    sync = cfg_a.get("sync")
    if not sync or sync.get("offset_frames") is None:
        raise RuntimeError("config-A needs a 'sync' block (side-B = side-A + offset).")
    offset = int(sync["offset_frames"])

    detA = _build_detector(cfg_a)
    detB = _build_detector(cfg_b)
    if not detA.operational or not detB.operational:
        print("[gaps] a detector is not operational (no weights) -- train a model first.")
        return ""

    # side-2 frames we already have a label for -> skip (relabeling adds no new info)
    labeled_b = set(_load_existing(_csv_path(out_dir, cfg_b["source"])).keys())

    dec = cfg_a.get("decode", {})
    readerA = ThreadedVideoReader(cfg_a["source"], hw_accel=dec.get("hw_accel", True),
                                  start_frame=0)
    readerB = ThreadedVideoReader(cfg_b["source"], hw_accel=dec.get("hw_accel", True),
                                  start_frame=offset)

    rows: List[Dict[str, Any]] = []
    n = 0
    n_s1 = 0                       # frames where side-1 saw the ball confidently
    n_both = 0                     # frames both already see (already triangulating)
    while True:
        okA, frameA = readerA.read()
        okB, frameB = readerB.read()
        if not okA or not okB:
            break
        dA = detA.detect(frameA)
        dB = detB.detect(frameB)
        b_frame = offset + n
        if dA.found and dA.confidence >= side1_conf_min:
            n_s1 += 1
            if dB.found:
                n_both += 1
            elif b_frame not in labeled_b:
                rows.append({"frame": b_frame, "reason": "tri_gap", "priority": 1,
                             "side1_conf": round(float(dA.confidence), 3)})
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    readerA.stop()
    readerB.stop()

    rows.sort(key=lambda r: r["frame"])
    base = os.path.splitext(os.path.basename(str(cfg_b["source"])))[0]
    path = os.path.join(out_dir, f"triangulation_gaps_{base}.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "reason", "priority", "side1_conf"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[gaps] scanned {n} synced frames; side-1 confidently saw the ball on {n_s1}.")
    print(f"[gaps]   of those, both cameras already see it on {n_both} "
          f"(those already triangulate).")
    print(f"[gaps]   -> {len(rows)} TRIANGULATION GAPS (side-1 sees, side-2 misses, "
          f"unlabeled).")
    print(f"[gaps] each labeled gap is a potential NEW 3D depth anchor -- the highest-"
          f"value side-2 labels you can add.")
    print(f"[gaps] wrote {path}")
    print(f"[gaps] label them (click the ball; B=not-visible if you can't find it):")
    print(f"       python main.py --config config-side2.json --label-ball --label-from {path}")
    return path


if __name__ == "__main__":
    cpa = sys.argv[1] if len(sys.argv) > 1 else "config-side1.json"
    cpb = sys.argv[2] if len(sys.argv) > 2 else "config-side2.json"
    mf = int(sys.argv[3]) if len(sys.argv) > 3 else None
    with open(cpa, "r", encoding="utf-8") as fh:
        ca = json.load(fh)
    with open(cpb, "r", encoding="utf-8") as fh:
        cb = json.load(fh)
    print(f"[gaps] finding triangulation gaps: {ca['source']} sees, "
          f"{cb['source']} misses (sync offset {ca.get('sync', {}).get('offset_frames')})")
    run_gap_finder(ca, cb, max_frames=mf)
