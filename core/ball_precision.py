"""core/ball_precision.py
PRECISION test harness -- the validated accuracy number we were missing.

It runs the ball DETECTOR over the frames you LABELLED and compares each prediction
to the ground-truth click, reporting real precision / recall / localization error.

WHY HELD-OUT MATTERS
  The model trained on all your labels, so scoring it on ALL of them is optimistic.
  The trainer kept a deterministic 15% VAL split (random seed 0) it never trained on
  -- we reproduce that split and report it separately. The HELD-OUT number is the
  honest one; the all-labelled number is an optimistic upper bound.

DEFINITIONS (per labelled frame)
  visible + found within tol     -> TP   (correct detection)
  visible + missed/found-far     -> FN   (the real ball was not found)   [far also -> FP]
  not-visible + found            -> FP   (a hallucinated ball)
  not-visible + not found        -> TN
  precision = TP/(TP+FP)   recall = TP/(TP+FN)
  localization error is reported in PIXELS, and in court METRES via the homography
  (approximate for an airborne ball, but a useful real-world scale).

RUN
  python main.py --precision --config config-side1.json
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
from typing import Any, Dict, List, Optional, Tuple

from core.ball_eval import _build_detector
from core.ball_label import _csv_path, _load_existing
from utils.homography import Homography
from utils.metrics import NumpyEncoder
from utils.video_io import ThreadedVideoReader

# one record per labelled frame: (frame, visible, label_u, label_v, found, det_u, det_v, conf)
Record = Tuple[int, bool, Optional[float], Optional[float], bool, Optional[float], Optional[float], float]


def _val_frames(labels: Dict[int, Any], in_frames: int, val_frac: float) -> set:
    """Reproduce the trainer's deterministic val split -> the set of held-out frames."""
    samples = [f for f in labels if f >= in_frames - 1]      # same filter as BallDataset
    if len(samples) < 2:
        return set(samples)
    idxs = list(range(len(samples)))
    random.Random(0).shuffle(idxs)                            # same seed as train_ball.py
    n_val = min(max(1, int(len(samples) * val_frac)), len(samples) - 1)
    return {samples[i] for i in idxs[:n_val]}


def _stats(vals: List[float]) -> Optional[Dict[str, float]]:
    if not vals:
        return None
    s = sorted(vals)
    p90 = s[min(len(s) - 1, int(0.9 * len(s)))]
    return {"mean": round(statistics.mean(s), 1), "median": round(statistics.median(s), 1),
            "p90": round(p90, 1)}


def _score(records: List[Record], tol_px: float, homog: Optional[Homography],
           min_conf: float = 0.0) -> Dict[str, Any]:
    """Score the records; a detection only counts if its confidence >= min_conf
    (so we can sweep the threshold post-hoc from one pass)."""
    TP = FP = FN = TN = 0
    fp_on_empty = 0
    errs_px: List[float] = []
    errs_m: List[float] = []
    within = {15: 0, 30: 0, 50: 0}
    n_vis = sum(1 for r in records if r[1])
    n_empty = len(records) - n_vis

    for (_f, vis, lu, lv, found, du, dv, conf) in records:
        eff = found and conf is not None and conf >= min_conf
        if vis:
            if eff:
                e = math.hypot(du - lu, dv - lv)
                errs_px.append(e)
                if homog is not None:
                    mx, my = homog.pixel_to_meters((du, dv))
                    lx, ly = homog.pixel_to_meters((lu, lv))
                    errs_m.append(math.hypot(mx - lx, my - ly))
                for k in within:
                    if e <= k:
                        within[k] += 1
                if e <= tol_px:
                    TP += 1
                else:
                    FP += 1          # detection in the wrong place ...
                    FN += 1          # ... and the real ball was missed
            else:
                FN += 1
        else:
            if eff:
                FP += 1
                fp_on_empty += 1
            else:
                TN += 1

    prec = TP / (TP + FP) if (TP + FP) else None
    rec = TP / (TP + FN) if (TP + FN) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else None
    return {
        "visible_frames": n_vis, "not_visible_frames": n_empty,
        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
        "precision": round(prec, 3) if prec is not None else None,
        "recall": round(rec, 3) if rec is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "false_positive_rate_on_empty": round(fp_on_empty / n_empty, 3) if n_empty else None,
        "localization_px": _stats(errs_px),
        "localization_m": _stats(errs_m),
        "pct_visible_within_px": {str(k): (round(100 * within[k] / n_vis, 1) if n_vis else None)
                                  for k in within},
    }


def run_precision(config: Dict[str, Any], tol_px: float = 30.0,
                  val_frac: float = 0.15) -> Dict[str, Any]:
    """Score the detector against the labelled frames. Returns the summary dict."""
    out_dir = config.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)
    source = config["source"]
    in_frames = config.get("ball", {}).get("in_frames", 3)

    csv_path = _csv_path(out_dir, source)
    labels = _load_existing(csv_path)
    if not labels:
        print(f"[precision] no labels at {csv_path} -- label some frames first "
              f"(python main.py --config <cfg> --label-ball).")
        return {}
    val = _val_frames(labels, in_frames, val_frac)
    homog = Homography.from_config(config)
    detector = _build_detector(config)
    if not detector.operational:
        print("[precision] detector not operational (no weights) -- train a model first.")
        return {}

    max_f = max(labels)
    print(f"[precision] scoring {len(labels)} labelled frames (held-out val={len(val)}) "
          f"on {source}; sequential pass to frame {max_f} (~a few minutes)...")
    reader = ThreadedVideoReader(source, hw_accel=config.get("decode", {}).get("hw_accel", True))

    records: List[Record] = []
    n = 0
    while True:
        ok, frame = reader.read()
        if not ok or n > max_f:
            break
        det = detector.detect(frame)
        if n in labels and n >= in_frames - 1:
            vis, lu, lv = labels[n]
            records.append((n, vis == 1, lu, lv, det.found, det.u, det.v, det.confidence))
        n += 1
    reader.stop()

    held_records = [r for r in records if r[0] in val]
    thresholds = [round(0.50 + 0.05 * i, 2) for i in range(7)]    # 0.50 .. 0.80
    sweep = []
    for T in thresholds:
        s = _score(held_records, tol_px, homog, min_conf=T)
        sweep.append({"threshold": T, "precision": s["precision"], "recall": s["recall"],
                      "f1": s["f1"], "fp_rate": s["false_positive_rate_on_empty"]})
    best = max(sweep, key=lambda s: (s["f1"] or 0.0))

    summary = {
        "source": source, "tol_px": tol_px,
        "scored_frames": len(records),
        "held_out": _score(held_records, tol_px, homog),
        "all_labelled": _score(records, tol_px, homog),
        "threshold_sweep_held_out": sweep,
        "recommended_threshold": best["threshold"],
        "note": "held_out = the trainer's val split (never trained on) = the honest number; "
                "all_labelled is optimistic (includes training frames).",
    }
    path = os.path.join(out_dir, "precision_metrics.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    ho = summary["held_out"]
    print(f"\n[precision] ===== HELD-OUT (honest) =====")
    print(f"  precision={ho['precision']}  recall={ho['recall']}  f1={ho['f1']}  "
          f"(tol={tol_px:.0f}px)")
    print(f"  localization error px: {ho['localization_px']}   metres: {ho['localization_m']}")
    print(f"  visible frames detected within 15/30/50 px: "
          f"{ho['pct_visible_within_px']['15']}% / {ho['pct_visible_within_px']['30']}% / "
          f"{ho['pct_visible_within_px']['50']}%")
    print(f"  false-positive rate on 'no-ball' frames: {ho['false_positive_rate_on_empty']}")
    print("\n[precision] CONFIDENCE-THRESHOLD SWEEP (held-out) -- raise it to cut false positives:")
    print("  thresh  precision  recall   f1     fp_rate")
    for s in sweep:
        print(f"   {s['threshold']:.2f}     {str(s['precision']):<8} {str(s['recall']):<7} "
              f"{str(s['f1']):<6} {s['fp_rate']}")
    print(f"[precision] best F1 at ball.heatmap_threshold={best['threshold']} "
          f"(precision {best['precision']}, recall {best['recall']}, fp_rate {best['fp_rate']}).")
    al = summary["all_labelled"]
    print(f"[precision] (all-labelled, optimistic: precision={al['precision']} "
          f"recall={al['recall']} median_px={al['localization_px']['median'] if al['localization_px'] else None})")
    print(f"[precision] full report -> {path}")
    return summary
