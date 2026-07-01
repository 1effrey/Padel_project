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
from core.ball_trail import smooth_trail

PANEL_H = 540
TRAIL_LEN = 8       # Fadi-style: a SHORT comet (~0.4s). Long trails fold into loops at hits.
GAP_FILL_MAX = 5    # bridge detector misses up to this many frames (Fadi: 'inpaint')
SMOOTH_WIN = 7      # local order-2 (parabola) smoothing window -> projectile-shaped arc
# NEAR-HALF OWNERSHIP (CLAUDE.md decision #2, Fadi's cam*_side): each camera keeps ONLY ball
# candidates in its near half -- v >= net_line_y - NEAR_MARGIN_PX.
# DISABLED for now: on a SINGLE camera this drops the ball for most of its flight, because in
# image space the near half is only ~1/3 of the frame (perspective), and nothing covers the
# dropped far half yet. It becomes correct in Phase 4 (fusion), where the OTHER camera owns the
# far half and `owner_cam` arbitrates -- Fadi never hard-cuts, he tags side + picks the owner.
# Kept here, gated off, so it's ready to switch on WITH fusion.
NEAR_HALF_ENABLED = False
NEAR_MARGIN_PX = 150   # px ABOVE the net line still counted as near (a ball at the net)
# per-camera selector params (from the eval: side-1 likes a bigger tube, side-2 the default)
SEL = {
    "side-1": dict(lag=5, max_step_px=450.0, min_support=2, static_radius_px=20.0),
    "side-2": dict(lag=5, max_step_px=350.0, min_support=2, static_radius_px=20.0),
}


def detect_and_select(cfg, params, max_frames):
    """Pass 1 for one camera -> {frame_idx: (u,v) or None} clean, SMOOTHED track.
    detector -> fixed-lag selector (ghost rejection) -> Kalman (smoothing + short coast).
    Same as the production chain, so this is what the integrated output will look like."""
    from core.ball_detector import BallDetection
    from core.ball_eval import _build_tracker
    det = _build_detector(cfg)
    sel = FixedLagBallSelector(**params)
    cap = cv2.VideoCapture(cfg["source"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    trk, _ = _build_tracker(cfg, fps)
    clean = {}
    # near-half cutoff for THIS camera (None -> no filter, keep whole frame). Off until fusion.
    net_y = (cfg.get("court", {}) or {}).get("net_line_y")
    near_cut = (float(net_y) - NEAR_MARGIN_PX) if (NEAR_HALF_ENABLED and net_y is not None) else None

    def feed(pt):
        meas = ([BallDetection(found=True, u=pt.u, v=pt.v, confidence=pt.conf, reason="ok")]
                if pt.source == "detected" else [])
        t = trk.update_multi(meas)
        # smoothed Kalman position; trail hygiene: draw only measured or a SHORT coast
        if t.x is not None and t.status != "lost" and (t.measured or t.coast <= 2):
            clean[pt.frame] = (float(t.x), float(t.y))
        else:
            clean[pt.frame] = None

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det.detect(frame)
        cands = list(getattr(det, "last_candidates", []))
        if near_cut is not None:                       # keep only near-half candidates
            cands = [c for c in cands if float(c.v) >= near_cut]
        pt = sel.push(n, cands)
        if pt is not None:
            feed(pt)
        n += 1
        if max_frames and n >= max_frames:
            break
    for pt in sel.flush():
        feed(pt)
    cap.release()
    print(f"  {cfg['source']}: {sum(1 for v in clean.values() if v)} clean ball frames / {len(clean)}")
    return clean


def _cached_track(cfg, params, max_frames, cache_path):
    """Return the clean per-frame track, running the (GPU) detector pass ONLY if there is no
    valid cache. The detector is the slow part; caching it lets us re-tune the TRAIL LOOK
    (dots/length/smoothing) instantly without re-detecting. Cache is invalidated if the
    source or max_frames changes. Delete the cache file (or pass --redetect) to force a rerun."""
    meta = {"source": cfg["source"], "max_frames": int(max_frames), "params": params,
            "near_half": NEAR_HALF_ENABLED, "near_margin": NEAR_MARGIN_PX,
            "heatmap_threshold": cfg.get("ball", {}).get("heatmap_threshold")}
    if "--redetect" not in sys.argv and os.path.exists(cache_path):
        try:
            blob = json.load(open(cache_path))
            if blob.get("meta") == meta:
                track = {int(k): (tuple(v) if v else None) for k, v in blob["track"].items()}
                print(f"  cache hit {cache_path}: {sum(1 for v in track.values() if v)} ball frames")
                return track
            print(f"  cache stale ({cache_path}) -> re-detecting")
        except Exception as e:
            print(f"  cache unreadable ({e}) -> re-detecting")
    track = detect_and_select(cfg, params, max_frames)
    json.dump({"meta": meta, "track": {str(k): v for k, v in track.items()}},
              open(cache_path, "w"))
    print(f"  cached -> {cache_path}")
    return track


def draw_trail(panel, trail, scale):
    """Fadi-style light comet: a per-frame DOT at each gap-filled, parabola-smoothed point
    (NO thick line -> no heavy bars on fast balls, no folded loops at hits) joined by a hair-
    thin 1px connector for readability, plus a hollow-ring head at the current ball. None
    entries BREAK the trail (a true gap -> ball left this half / long miss, never bridged)."""
    pts = list(trail)
    prev = None
    n = len(pts)
    for i, p in enumerate(pts):
        if p is None:
            prev = None
            continue
        cur = (int(round(p[0] * scale)), int(round(p[1] * scale)))
        if prev is not None:
            cv2.line(panel, prev, cur, (0, 200, 255), 1, cv2.LINE_AA)   # hair-thin connector
        cv2.circle(panel, cur, 2 if i < n - 1 else 3, (0, 255, 255), -1, cv2.LINE_AA)
        prev = cur
    if prev is not None:                                   # hollow-ring target on the ball
        cv2.circle(panel, prev, 6, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(panel, prev, 1, (255, 255, 255), -1, cv2.LINE_AA)


def main(max_frames=0):
    cfgA = json.load(open("config-side1.json"))
    cfgB = json.load(open("config-side2.json"))
    offset = int(cfgA.get("sync", {}).get("offset_frames", 0))
    out_dir = cfgA.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "selector_render.mp4")

    print("pass 1: detect + select (cached) ...")
    cleanA = _cached_track(cfgA, SEL["side-1"], max_frames, os.path.join(out_dir, "clean_side1.json"))
    cleanB = _cached_track(cfgB, SEL["side-2"], max_frames, os.path.join(out_dir, "clean_side2.json"))

    # gap-fill short misses + parabola-smooth each flight segment -> clean projectile arcs
    smA = {f: (p.u, p.v) for f, p in
           smooth_trail(cleanA, GAP_FILL_MAX, SMOOTH_WIN).items()}
    smB = {f: (p.u, p.v) for f, p in
           smooth_trail(cleanB, GAP_FILL_MAX, SMOOTH_WIN).items()}
    print(f"  smoothed: side-1 {len(smA)} drawn pts, side-2 {len(smB)} drawn pts")

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
        trailA.append(smA.get(nA))
        trailB.append(smB.get(nB))
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
    # first numeric arg = max_frames (0 = whole video); --redetect forces a fresh detector pass
    nums = [a for a in sys.argv[1:] if a.isdigit()]
    main(int(nums[0]) if nums else 0)
