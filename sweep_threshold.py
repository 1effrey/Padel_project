"""sweep_threshold.py -- find the best ball heatmap_threshold in ONE detector pass.

Raising the detector's `heatmap_threshold` trades recall for precision (a stricter peak
required to call a blob a ball). Instead of re-running the GPU detector once per threshold, we
run it ONCE at a LOW base threshold (capturing every candidate + its heatmap-peak confidence),
then SIMULATE each higher threshold by keeping only candidates with peak >= thr. Each simulated
threshold is scored through the SAME production path we ship (selector -> Kalman tracker) so the
precision/recall we read is the precision/recall of the real pipeline.

This is an approximation (the peak gate ~ the heatmap binarisation; blob area can shift a little
at the true threshold), so treat the winner as the value to CONFIRM with one real eval run at
that threshold in config. It's plenty to find the right ballpark cheaply.

Usage (pod):
  python sweep_threshold.py config-side1.json output/ball_labels_side-1-full-vid.csv 0 450
  python sweep_threshold.py config-side2.json output/ball_labels_side-2-full-vid.csv 0 350
  # args: config labels [max_frames] [selector_max_step_px] [base_thr]
"""
import json
import sys

import cv2

from core.ball_detector import BallDetection
from core.ball_eval import _build_detector, _build_tracker
from core.ball_selector import FixedLagBallSelector
from eval_selector import load_labels, measured_xy, score

BASE_THR = 0.30          # capture candidates down to here, then simulate stricter thresholds
SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def run(config_path, label_path, max_frames=0, max_step=350.0, base_thr=BASE_THR):
    cfg = json.load(open(config_path))
    cfg.setdefault("ball", {})["heatmap_threshold"] = float(base_thr)   # capture wide
    det = _build_detector(cfg)
    cap = cv2.VideoCapture(cfg["source"])
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0

    # ONE detector pass: store every frame's candidates as plain (u, v, conf) tuples
    per_frame = []
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        det.detect(frame)
        per_frame.append([(float(c.u), float(c.v), float(getattr(c, "confidence", 1.0)))
                          for c in getattr(det, "last_candidates", [])])
        n += 1
        if max_frames and n >= max_frames:
            break
    cap.release()
    L = load_labels(label_path)
    print(f"detector pass done: {n} frames, {sum(len(c) for c in per_frame)} raw candidates "
          f"(base_thr={base_thr}, selector max_step={max_step})\n")
    print(f"{'thr':>5} {'precision':>10} {'recall':>8} {'med_err':>8} {'big_jumps':>10} "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")

    for thr in SWEEP:
        if thr < base_thr:
            continue
        trk, _ = _build_tracker(cfg, fps)
        sel = FixedLagBallSelector(lag=5, max_step_px=max_step, min_support=2, static_radius_px=20.0)
        track = {}

        def feed(pt):
            meas = ([BallDetection(found=True, u=pt.u, v=pt.v, confidence=pt.conf, reason="ok")]
                    if pt.source == "detected" else [])
            track[pt.frame] = measured_xy(trk.update_multi(meas))

        for i, cands in enumerate(per_frame):
            kept = [BallDetection(found=True, u=u, v=v, confidence=c, reason="ok")
                    for (u, v, c) in cands if c >= thr]           # simulate this threshold
            pt = sel.push(i, kept)
            if pt is not None:
                feed(pt)
        for pt in sel.flush():
            feed(pt)

        prec, rec, med, jumps, TP, FP, FN = score(track, L)
        print(f"{thr:>5.2f} {prec:>10.3f} {rec:>8.3f} {med:>8.1f} {jumps:>10d} "
              f"{TP:>5d} {FP:>5d} {FN:>5d}")

    print("\npick the row with the best precision that still holds recall, then set that value as "
          "ball.heatmap_threshold in the config and confirm with one eval_selector.py run.")


if __name__ == "__main__":
    a = sys.argv
    run(a[1], a[2],
        int(a[3]) if len(a) > 3 else 0,
        float(a[4]) if len(a) > 4 else 350.0,
        float(a[5]) if len(a) > 5 else BASE_THR)
