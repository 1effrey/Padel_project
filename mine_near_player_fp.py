"""mine_near_player_fp.py -- propose NEAR-PLAYER false positives for a hand-verified pass.

The residual ball-track error is the detector firing on a player's hand / racket / the fence
behind them (a "ball" glued to a near-stationary player). This miner finds those candidates so a
human can CONFIRM each one before it becomes a training hard-negative -- never auto-labelled,
because that is exactly what poisoned the last retrain (real slow/brief balls were auto-marked
not-a-ball).

SIGNATURE we exploit: a REAL ball only passes THROUGH a player's box for 1-3 frames (it arrives,
is hit, leaves). A false positive LINGERS inside/near a player box for many frames (it's stuck on
the hand/racket/fence) and barely moves. So we flag runs of consecutive drawn-ball frames that
sit inside a player box for >= MIN_LINGER frames with small displacement. This catches the stuck
FPs while leaving real contacts alone -- but it is still only a PROPOSAL; the human decides.

INPUT  (cheap: one POSE pass; reuses the render's ball track so no ball-detector re-run):
  - output/clean_side{N}.json  (the drawn ball per frame, produced by render_selector.py)
  - the pose model from the config (player boxes)
OUTPUT: a proposals CSV in the miner format the labeler already reads
  (frame,reason,priority,hint_u,hint_v,confidence) -> review with:
    python main.py --config config-sideN.json --label-ball \
        --label-from fp_near_player_side-N.csv --label-out output/hardneg_ball_<clip>.csv

Usage (pod, needs the render cache to exist):
  python mine_near_player_fp.py config-side2.json
  python mine_near_player_fp.py config-side2.json output/clean_side2.json fp_near_player_side-2.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

MIN_LINGER = 5        # consecutive frames a drawn ball must sit in a player box to be suspect
STUCK_PX = 120.0      # ... and move less than this (full-res px) across the run -> stuck, not flight
BOX_MARGIN = 0.15     # expand each player box by this fraction (racket/hand reach beyond the body)


def _load_track(path: str) -> Dict[int, Tuple[float, float]]:
    """Read render_selector's clean cache -> {frame: (u,v)} for frames where a ball was drawn."""
    blob = json.load(open(path))
    out: Dict[int, Tuple[float, float]] = {}
    for k, v in blob.get("track", {}).items():
        if v:
            out[int(k)] = (float(v[0]), float(v[1]))
    return out


def _in_box(u: float, v: float, box, margin: float) -> bool:
    x1, y1, x2, y2 = box
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    return (x1 - mx) <= u <= (x2 + mx) and (y1 - my) <= v <= (y2 + my)


def _runs(flagged: Dict[int, bool]) -> List[List[int]]:
    """Maximal runs of consecutive frame indices that are all near a player."""
    runs, cur = [], []
    for f in sorted(flagged):
        if not flagged[f]:
            continue
        if cur and f == cur[-1] + 1:
            cur.append(f)
        else:
            if cur:
                runs.append(cur)
            cur = [f]
    if cur:
        runs.append(cur)
    return runs


def mine(config_path: str, clean_path: Optional[str] = None, out_path: Optional[str] = None) -> None:
    import cv2
    from core.detector import PoseDetector

    cfg = json.load(open(config_path))
    src = cfg["source"]
    clip = os.path.splitext(os.path.basename(src))[0]
    if clean_path is None:
        clean_path = os.path.join(cfg.get("output", {}).get("dir", "output"),
                                  f"clean_side{'2' if 'side2' in config_path else '1'}.json")
    if out_path is None:
        out_path = f"fp_near_player_{clip}.csv"
    if not os.path.isfile(clean_path):
        sys.exit(f"no render cache at {clean_path} -- run render_selector.py first "
                 f"(the miner reuses its ball track, so it doesn't re-run the ball detector).")

    track = _load_track(clean_path)
    pose = PoseDetector(model_path=cfg["model"], device=cfg.get("device", "cuda"),
                        conf_threshold=cfg.get("detection", {}).get("conf_threshold", 0.3))
    cap = cv2.VideoCapture(src)

    near: Dict[int, bool] = {}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n in track:                                   # only frames where a ball was DRAWN
            u, v = track[n]
            boxes = [d["bbox"] for d in pose.detect(frame)]
            near[n] = any(_in_box(u, v, b, BOX_MARGIN) for b in boxes)
        n += 1
    cap.release()

    # keep only lingering + low-displacement runs (stuck on the player, not a ball in flight)
    proposals: List[Tuple[int, float, float, int]] = []   # (frame, u, v, run_len)
    kept_runs = 0
    for run in _runs(near):
        if len(run) < MIN_LINGER:
            continue
        us = [track[f][0] for f in run]
        vs = [track[f][1] for f in run]
        disp = ((max(us) - min(us)) ** 2 + (max(vs) - min(vs)) ** 2) ** 0.5
        if disp > STUCK_PX:                              # it moved like a ball -> not our target
            continue
        kept_runs += 1
        for f in run:
            proposals.append((f, track[f][0], track[f][1], len(run)))

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "reason", "priority", "hint_u", "hint_v", "confidence"])
        for f, u, v, rl in proposals:
            w.writerow([f, "fp_near_player", rl, f"{u:.1f}", f"{v:.1f}", ""])

    print(f"{clip}: scanned {len(track)} drawn-ball frames -> {kept_runs} lingering-near-player "
          f"runs, {len(proposals)} candidate frames -> {out_path}")
    print("NEXT: hand-verify (default = it's a real ball; press B ONLY when it's clearly NOT):")
    print(f"  python main.py --config {config_path} --label-ball --label-from {out_path} "
          f"--label-out output/hardneg_ball_{clip}.csv")


# --------------------------------------------------------------------------- #
# Unit tests (logic only, no GPU/video):  python mine_near_player_fp.py --test
# --------------------------------------------------------------------------- #
def _test() -> None:
    # a real ball passes through a box for 2 frames -> NOT flagged; a stuck FP lingers 6 -> flagged
    flagged = {f: False for f in range(20)}
    for f in (5, 6):            # brief real contact
        flagged[f] = True
    for f in range(10, 16):     # stuck FP (6 frames)
        flagged[f] = True
    runs = _runs(flagged)
    assert [len(r) for r in runs] == [2, 6], runs
    long_runs = [r for r in runs if len(r) >= MIN_LINGER]
    assert long_runs == [[10, 11, 12, 13, 14, 15]], long_runs
    # box/overlap + margin
    assert _in_box(100, 100, (90, 90, 110, 110), 0.0)
    assert not _in_box(200, 200, (90, 90, 110, 110), 0.0)
    assert _in_box(113, 100, (90, 90, 110, 110), 0.15)      # racket reach via margin
    print("mine_near_player_fp tests PASS: brief contact ignored, lingering FP flagged, box ok")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        _test()
    else:
        mine(sys.argv[1],
             sys.argv[2] if len(sys.argv) > 2 else None,
             sys.argv[3] if len(sys.argv) > 3 else None)
