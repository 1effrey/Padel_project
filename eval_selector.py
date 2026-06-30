"""eval_selector.py -- A/B: does the fixed-lag selector clean the track WITHOUT losing recall?

Runs the detector ONCE over a clip, then scores two tracks against the labels:
  A) baseline:  candidates -> Kalman tracker                       (current behaviour)
  B) selector:  candidates -> FixedLagBallSelector -> Kalman tracker

Reports precision / recall / median localization / BIG-JUMP count (the zigzag metric) for each.

Usage (run on the pod -- GPU makes the detector pass fast):
  python eval_selector.py config-side1.json output/ball_labels_side-1-full-vid.csv
  # optional: max_frames lag max_step min_support static_radius
  python eval_selector.py config-side1.json output/ball_labels_side-1-full-vid.csv 0 5 350 2 20
"""
import csv
import json
import math
import statistics as st
import sys

import cv2

from core.ball_detector import BallDetection
from core.ball_eval import _build_detector, _build_tracker
from core.ball_selector import FixedLagBallSelector

JUMP_PX = 300.0   # frame-to-frame measured jump above this = a "big jump" (zigzag/teleport)


def load_labels(path):
    L = {}
    for r in csv.DictReader(open(path)):
        u, v = r["u"], r["v"]
        L[int(r["frame"])] = (r["visible"] == "1",
                              float(u) if u not in ("", "-1") else None,
                              float(v) if v not in ("", "-1") else None)
    return L


def measured_xy(t):
    """The track's MEASURED point this frame (None when coasting/lost) -- matches the
    precision harness, which scores accepted detections, not coasted predictions."""
    return (t.meas_u, t.meas_v) if t.measured else (None, None)


def score(track, L, tol=30.0):
    TP = FP = FN = 0
    errs = []
    jumps = 0
    prev = None
    for f in sorted(track):
        u, v = track[f]
        if u is not None:
            if prev is not None and math.hypot(u - prev[0], v - prev[1]) > JUMP_PX:
                jumps += 1
            prev = (u, v)
        if f not in L:
            continue
        vis, lu, lv = L[f]
        present = u is not None
        if vis and lu is not None:
            if present and math.hypot(u - lu, v - lv) <= tol:
                TP += 1
                errs.append(math.hypot(u - lu, v - lv))
            elif present:
                FP += 1
                FN += 1
            else:
                FN += 1
        else:
            if present:
                FP += 1
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    med = st.median(errs) if errs else float("nan")
    return prec, rec, med, jumps, TP, FP, FN


def run(config_path, label_path, max_frames=0, lag=5, max_step=350.0, min_support=2, static_r=20.0):
    cfg = json.load(open(config_path))
    det = _build_detector(cfg)
    cap = cv2.VideoCapture(cfg["source"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    trkA, _ = _build_tracker(cfg, fps)
    trkB, _ = _build_tracker(cfg, fps)
    sel = FixedLagBallSelector(lag=lag, max_step_px=max_step,
                               min_support=min_support, static_radius_px=static_r)

    trackA, trackB = {}, {}

    def feedB(pt):
        meas = ([BallDetection(found=True, u=pt.u, v=pt.v, confidence=pt.conf, reason="ok")]
                if pt.source == "detected" else [])
        trackB[pt.frame] = measured_xy(trkB.update_multi(meas))

    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det.detect(frame)
        cands = list(getattr(det, "last_candidates", []))
        trackA[n] = measured_xy(trkA.update_multi(cands))      # path A
        pt = sel.push(n, cands)                                 # path B (lagged)
        if pt is not None:
            feedB(pt)
        n += 1
        if max_frames and n >= max_frames:
            break
    for pt in sel.flush():
        feedB(pt)
    cap.release()

    L = load_labels(label_path)
    pa, ra, ea, ja, ta, fa, na = score(trackA, L)
    pb, rb, eb, jb, tb, fb, nb = score(trackB, L)
    print(f"scored {len([f for f in trackA if f in L])} labelled frames "
          f"(selector: lag={lag} max_step={max_step} min_support={min_support} static_r={static_r})")
    print(f"A baseline : precision={pa:.3f} recall={ra:.3f} med_err={ea:.1f}px  big_jumps={ja}  (TP={ta} FP={fa} FN={na})")
    print(f"B selector : precision={pb:.3f} recall={rb:.3f} med_err={eb:.1f}px  big_jumps={jb}  (TP={tb} FP={fb} FN={nb})")
    print(f"-> precision {pb-pa:+.3f}, recall {rb-ra:+.3f}, big_jumps {jb-ja:+d}  "
          f"(want: precision up, recall ~flat, jumps DOWN)")


if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2],
        int(a[3]) if len(a) > 3 else 0,
        int(a[4]) if len(a) > 4 else 5,
        float(a[5]) if len(a) > 5 else 350.0,
        int(a[6]) if len(a) > 6 else 2,
        float(a[7]) if len(a) > 7 else 20.0)
