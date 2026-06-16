"""core/ball_mine.py
HARD-EXAMPLE MINING for ball labeling.

WHY
  Labeling is the team's scarcest resource, and most frames are EASY -- the model
  already gets them. Labeling those teaches it nothing. This tool runs the current
  model over a clip and lists the frames where it FAILS, so you spend your labeling
  time exactly where it moves the needle:

    * MISS        -- the detector said "no ball" but the tracker was mid-flight
                  (coasting a SHORT gap), i.e. the ball was almost certainly there
                  and the model missed it. Label the ball -> attacks RECALL.
    * FP_ISOLATED -- an ISOLATED one-off detection (found this frame, but not the
                  frames around it). The real ball arrives in RUNS, so a lone
                  detection is usually the model firing on a light/limb. Label it
                  "not visible" (B) -> a HARD NEGATIVE that attacks the 40% FALSE-
                  POSITIVE rate the precision harness measured.
    * FP_STATIC   -- a SUSTAINED detection that barely moves over many frames. The
                  real ball never stays put, so this is the model locking onto a fixed
                  object (a light / reflection). Also a HARD NEGATIVE -> label "not
                  visible". (Catches the sustained FPs that FP_ISOLATED misses.)
    * LOWCONF     -- the detector found a ball but with low confidence: it is unsure,
                  so a human label confirms or corrects it.

  It writes output/hard_frames_<clip>.csv (a 'frame' column the labeler can consume
  via --label-from), so the loop is:  mine -> label those frames -> retrain.

NOTE: this is pure ACTIVE LEARNING -- it only SELECTS frames to hand-label. It does
NOT auto-accept the model's guesses as labels (that would risk training the model on
its own mistakes). A human still labels every mined frame.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Any, Dict, List, Optional

from core.ball_eval import _build_detector, _build_tracker
from core.ball_tracker import BallTracker
from utils.video_io import ThreadedVideoReader


def _hard_frames_path(out_dir: str, source: str) -> str:
    base = os.path.splitext(os.path.basename(str(source)))[0]
    return os.path.join(out_dir, f"hard_frames_{base}.csv")


def run_mine_hard(
    config: Dict[str, Any],
    max_frames: Optional[int] = None,
    lowconf: float = 0.6,
    max_miss_gap: int = 8,
    isolation_window: int = 3,
    min_support: int = 1,
    static_window: int = 5,
    static_radius: float = 25.0,
    min_static_run: int = 5,
) -> str:
    """Scan config['source'] and write the frames worth labeling -- MISSES (ball
    present, model silent), suspected FALSE POSITIVES (isolated one-off detections),
    and low-confidence detections. Returns the path to the hard-frames CSV."""
    out_dir = config.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)

    detector = _build_detector(config)
    if not detector.operational:
        print("[mine] detector is not operational (no weights / stub) -- nothing to "
              "mine. Train a model first, or use ball.method='motion'.")
        return ""

    reader = ThreadedVideoReader(
        config["source"], hw_accel=config.get("decode", {}).get("hw_accel", True),
        queue_size=config.get("decode", {}).get("queue_size", 4))
    fps_in = reader.fps or 20.0
    tracker, _ = _build_tracker(config, fps_in)   # for MISS detection (coasting gaps)
    if tracker is None:
        tracker = BallTracker(dt=1.0 / fps_in if fps_in else 0.05)

    # --- pass 1: collect per-frame detection + track state ---
    per_frame = []                                # (frame, found, u, v, conf)
    track_info: Dict[int, tuple] = {}             # frame -> (status, coast, pred_x, pred_y)
    n = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        det = detector.detect(frame)
        track = tracker.update_multi(getattr(detector, "last_candidates", []))
        per_frame.append((n, det.found,
                          None if det.u is None else round(float(det.u), 1),
                          None if det.v is None else round(float(det.v), 1),
                          round(float(det.confidence), 3)))
        track_info[n] = (track.status, track.coast, track.x, track.y)
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    reader.stop()

    # --- pass 2: classify each frame ---
    found_frames = {fr[0] for fr in per_frame if fr[1]}
    found_pos = {fr[0]: (fr[2], fr[3]) for fr in per_frame if fr[1] and fr[2] is not None}
    rows: List[Dict[str, Any]] = []
    last_static = -10 ** 9
    for (f, found, u, v, conf) in per_frame:
        if found:
            # support = how many surrounding frames also have a detection
            support = sum(1 for d in range(1, isolation_window + 1)
                          if (f - d) in found_frames or (f + d) in found_frames)
            if support < min_support:
                # isolated one-off detection -> likely a flicker false positive
                rows.append({"frame": f, "reason": "fp_isolated", "priority": 1,
                             "hint_u": u, "hint_v": v, "confidence": conf})
                continue
            # sustained STATIC run -> a fixed distractor (a light / reflection); the
            # real ball never stays put. Flagged sparsely (one per static_window).
            win = [found_pos[g] for g in range(f - static_window, f + static_window + 1)
                   if g in found_pos]
            if (u is not None and len(win) >= min_static_run
                    and max(math.hypot(px - u, py - v) for (px, py) in win) <= static_radius
                    and f - last_static >= static_window):
                rows.append({"frame": f, "reason": "fp_static", "priority": 1,
                             "hint_u": u, "hint_v": v, "confidence": conf})
                last_static = f
                continue
            if conf < lowconf:
                rows.append({"frame": f, "reason": "lowconf", "priority": 2,
                             "hint_u": u, "hint_v": v, "confidence": conf})
        else:
            st, co, px, py = track_info.get(f, ("", 0, None, None))
            if st == "coasting" and co <= max_miss_gap:
                rows.append({"frame": f, "reason": "miss", "priority": 1,
                             "hint_u": round(float(px), 1) if px is not None else "",
                             "hint_v": round(float(py), 1) if py is not None else "",
                             "confidence": ""})

    rows.sort(key=lambda r: r["frame"])           # label order = clip order
    path = _hard_frames_path(out_dir, config["source"])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["frame", "reason", "priority",
                                           "hint_u", "hint_v", "confidence"])
        w.writeheader()
        w.writerows(rows)

    n_miss = sum(1 for r in rows if r["reason"] == "miss")
    n_fp = sum(1 for r in rows if r["reason"] in ("fp_isolated", "fp_static"))
    n_low = sum(1 for r in rows if r["reason"] == "lowconf")
    print(f"\n[mine] scanned {n} frames -> {len(rows)} worth labeling "
          f"({n_miss} misses, {n_fp} suspected false-positives "
          f"[{sum(1 for r in rows if r['reason']=='fp_isolated')} flicker + "
          f"{sum(1 for r in rows if r['reason']=='fp_static')} static], "
          f"{n_low} low-confidence).")
    print(f"[mine] wrote {path}")
    print(f"[mine] label them (click the real ball, or B = not-visible for a false positive):")
    print(f"       python main.py --config <your-config> --label-ball --label-from {path}")
    print(f"[mine] then retrain + re-measure:")
    print(f"       python train_ball.py --config config-side1.json --config2 config-side2.json")
    print(f"       python main.py --precision --config config-side1.json")
    return path
