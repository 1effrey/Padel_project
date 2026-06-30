"""render_selector.py -- SEE the fixed-lag selector's clean track on the video.

Pass 1: run the detector + FixedLagBallSelector per camera -> a clean per-frame ball
        position (ghosts/teleports/static removed), keyed by that camera's frame index.
Pass 2: redraw the two camera feeds side by side with the clean ball + a short trail
        (the trail BREAKS on a gap, so a lost ball never draws a guessed line).

No EKF / triangulation / events here -- this is purely to judge TRACK CLEANLINESS vs the
earlier ball_dual videos. Real-time-faithful: the selector is the same fixed-lag component;
we just store its output by frame and draw it on the matching frame (no lag misalignment).

Usage (pod):
  python render_selector.py            # uses config-side1.json / config-side2.json, defaults
  python render_selector.py 600        # only first 600 frames (quick)
Writes output/selector_render.mp4.
"""
from __future__ import annotations

import json
import os
import sys
from collections import deque

import cv2

from core.ball_eval import _build_detector
from core.ball_selector import FixedLagBallSelector

PANEL_H = 540
TRAIL_LEN = 25
# per-camera selector params (from the eval: side-1 likes a bigger tube, side-2 the default)
SEL = {
    "side-1": dict(lag=5, max_step_px=450.0, min_support=2, static_radius_px=20.0),
    "side-2": dict(lag=5, max_step_px=350.0, min_support=2, static_radius_px=20.0),
}


def detect_and_select(cfg, params, max_frames):
    """Pass 1 for one camera -> {frame_idx: (u,v) or None} clean track."""
    det = _build_detector(cfg)
    sel = FixedLagBallSelector(**params)
    cap = cv2.VideoCapture(cfg["source"])
    clean = {}
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det.detect(frame)
        pt = sel.push(n, list(getattr(det, "last_candidates", [])))
        if pt is not None:
            clean[pt.frame] = (pt.u, pt.v) if pt.source == "detected" else None
        n += 1
        if max_frames and n >= max_frames:
            break
    for pt in sel.flush():
        clean[pt.frame] = (pt.u, pt.v) if pt.source == "detected" else None
    cap.release()
    print(f"  {cfg['source']}: {sum(1 for v in clean.values() if v)} clean ball frames / {len(clean)}")
    return clean


def draw_trail(panel, trail, scale):
    """Fading polyline + head dot; None entries BREAK the line (a true gap)."""
    pts = list(trail)
    prev = None
    for i, p in enumerate(pts):
        if p is None:
            prev = None
            continue
        cur = (int(p[0] * scale), int(p[1] * scale))
        if prev is not None:
            thick = max(1, int(1 + 3 * i / max(1, len(pts) - 1)))
            cv2.line(panel, prev, cur, (0, 255, 255), thick, cv2.LINE_AA)
        prev = cur
    if prev is not None:
        cv2.circle(panel, prev, 7, (0, 255, 255), -1)
        cv2.circle(panel, prev, 7, (255, 255, 255), 1)


def main(max_frames=0):
    cfgA = json.load(open("config-side1.json"))
    cfgB = json.load(open("config-side2.json"))
    offset = int(cfgA.get("sync", {}).get("offset_frames", 0))
    out_dir = cfgA.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "selector_render.mp4")

    print("pass 1: detect + select ...")
    cleanA = detect_and_select(cfgA, SEL["side-1"], max_frames)
    cleanB = detect_and_select(cfgB, SEL["side-2"], max_frames)

    print("pass 2: render ...")
    capA = cv2.VideoCapture(cfgA["source"])
    capB = cv2.VideoCapture(cfgB["source"])
    capB.set(cv2.CAP_PROP_POS_FRAMES, offset)
    fps = capA.get(cv2.CAP_PROP_FPS) or 20.0
    trailA, trailB = deque(maxlen=TRAIL_LEN), deque(maxlen=TRAIL_LEN)
    writer = None
    nA, nB = 0, offset
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        okA, fA = capA.read()
        okB, fB = capB.read()
        if not okA or not okB:
            break
        sA = PANEL_H / fA.shape[0]
        sB = PANEL_H / fB.shape[0]
        pA = cv2.resize(fA, (int(fA.shape[1] * sA), PANEL_H))
        pB = cv2.resize(fB, (int(fB.shape[1] * sB), PANEL_H))
        trailA.append(cleanA.get(nA))
        trailB.append(cleanB.get(nB))
        draw_trail(pA, trailA, sA)
        draw_trail(pB, trailB, sB)
        cv2.putText(pA, "SIDE 1 (selector)", (12, 30), font, 0.9, (0, 255, 255), 2)
        cv2.putText(pB, "SIDE 2 (selector)", (12, 30), font, 0.9, (0, 255, 255), 2)
        comp = cv2.hconcat([pA, pB])
        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     fps, (comp.shape[1], comp.shape[0]))
        writer.write(comp)
        nA += 1
        nB += 1
        if max_frames and nA >= max_frames:
            break
    capA.release()
    capB.release()
    if writer is not None:
        writer.release()
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
