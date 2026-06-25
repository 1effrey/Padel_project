"""core/bounce_eval.py
MEASURE the accuracy of floor-bounce detection (core/ball_events.py) against a
HAND-CONFIRMED set of true bounces. This is the QUALITY GATE for the bounce/landing
data: a bounce we cannot measure is a bounce we cannot trust.

Two modes, BOTH CSV-ONLY (no video, no GPU -- run anywhere, including a CPU laptop):

  propose : run the SAME Kalman tracker + BallEventDetector the pipeline uses, but on
            the HAND-LABELLED (u,v) track, and OVER-PROPOSE candidate floor bounces
            (a LOWERED velocity threshold so we miss few). You then eyeball the
            candidate frames (core/bounce_frames.py) and edit the file into a TRUTH
            csv: keep the real bounces, delete the false ones, ADD any that were
            missed (a row with at least a `frame`), and fix `in_court`.

  measure : compare PREDICTED bounces to your TRUTH csv -> recall / precision /
            timing error / landing error / in-out agreement. Predictions come from
            EITHER the hand track (default -- isolates the LOGIC, runs on the laptop)
            OR a ball_eval JSONL produced on the pod (--pred-jsonl -- the real
            end-to-end TrackNet detector; the heavy run is on the pod, but this only
            reads its small JSONL so the MEASURING still runs anywhere).

WHY TWO PREDICTION SOURCES
  hand track       -> "is the bounce LOGIC correct?"  (no detector noise)
  ball_eval JSONL  -> "how does it do on the REAL detections?" (end-to-end)
  Run both: if the logic is good but end-to-end is worse, the gap is DETECTION
  quality (compression / recall), not the bounce algorithm.

TRUTH / CANDIDATE CSV SCHEMA (the propose output IS the truth template)
  frame      (required)  video frame index of the bounce
  type       (optional)  "floor_bounce" (others ignored by measure)
  u, v       (optional)  image pixel (for reference / frame extraction)
  x_m, y_m   (optional)  court metres of the landing (compared if present in BOTH)
  in_court   (optional)  1/0 or true/false (compared if present in BOTH)

USAGE
  # 1) propose candidates from the hand labels (over-proposed, laptop):
  python -m core.bounce_eval propose \
      --labels results/ball_labels_side-1-full-vid.csv \
      --config config-side1.json --fps 20 --propose-min-vy 300 \
      --out results/bounce_candidates_side-1.csv

  # 2) (you review + edit that file into results/bounce_truth_side-1.csv)

  # 3a) measure the LOGIC on the hand track (laptop):
  python -m core.bounce_eval measure \
      --labels results/ball_labels_side-1-full-vid.csv \
      --config config-side1.json --fps 20 \
      --truth results/bounce_truth_side-1.csv --tol 3

  # 3b) measure END-TO-END from a pod ball_eval run:
  python -m core.bounce_eval measure \
      --pred-jsonl output/ball_eval_side-1.jsonl \
      --truth results/bounce_truth_side-1.csv --tol 3
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
Label = Tuple[int, bool, Optional[float], Optional[float]]   # frame, visible, u, v


def load_labels(path: str) -> List[Label]:
    """Read a ball_labels csv (frame, visible, u, v) in frame order."""
    out: List[Label] = []
    for r in csv.DictReader(open(path, newline="")):
        frame = int(float(r["frame"]))
        visible = str(r.get("visible", "")).strip() in ("1", "1.0", "true", "True")
        u = r.get("u", "").strip()
        v = r.get("v", "").strip()
        out.append((frame, visible,
                    float(u) if u else None,
                    float(v) if v else None))
    out.sort(key=lambda t: t[0])
    return out


def _fnum(v: Any) -> Optional[float]:
    s = str(v).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _fbool(v: Any) -> Optional[bool]:
    s = str(v).strip().lower()
    if s in ("1", "1.0", "true", "yes", "in"):
        return True
    if s in ("0", "0.0", "false", "no", "out"):
        return False
    return None


def load_event_csv(path: str) -> List[Dict[str, Any]]:
    """Read a candidate/truth csv into a list of dicts (only `frame` required)."""
    out: List[Dict[str, Any]] = []
    for r in csv.DictReader(open(path, newline="")):
        f = _fnum(r.get("frame"))
        if f is None:
            continue
        out.append({
            "frame": int(f),
            "x_m": _fnum(r.get("x_m")),
            "y_m": _fnum(r.get("y_m")),
            "in_court": _fbool(r.get("in_court")),
        })
    out.sort(key=lambda d: d["frame"])
    return out


def load_pred_jsonl(path: str) -> List[Dict[str, Any]]:
    """Pull floor_bounce events out of a ball_eval per-frame JSONL (pod output)."""
    out: List[Dict[str, Any]] = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        ev = rec.get("event")
        if not ev or ev.get("type") != "floor_bounce":
            continue
        out.append({
            "frame": int(ev.get("frame", rec.get("frame"))),
            "x_m": ev.get("x_m"), "y_m": ev.get("y_m"),
            "in_court": ev.get("in_court"),
        })
    out.sort(key=lambda d: d["frame"])
    return out


# --------------------------------------------------------------------------- #
# Run the SAME tracker + event detector the pipeline uses, on the HAND track
# --------------------------------------------------------------------------- #
def events_from_labels(labels: List[Label], config: Dict[str, Any], fps: float,
                       min_vy_override: Optional[float] = None
                       ) -> Tuple[List[Dict[str, Any]], float]:
    """Feed the hand (u,v) track through BallTracker + BallEventDetector and return
    the FLOOR-BOUNCE events as dicts. dt is the REAL time between consecutive label
    rows (labels are strided), so the px/s velocities -- and thus the thresholds --
    are correct."""
    from core.ball_detector import BallDetection
    from core.ball_tracker import BallTracker
    from core.ball_events import BallEventDetector
    from utils.homography import Homography

    t = config.get("ball", {}).get("tracker", {})
    gaps = [b[0] - a[0] for a, b in zip(labels, labels[1:]) if b[0] > a[0]]
    stride = st.median(gaps) if gaps else 1.0
    dt = t.get("dt") or (stride / fps if fps else 0.05)

    tracker = BallTracker(
        dt=dt,
        process_noise=t.get("process_noise", 50000.0),
        meas_noise=t.get("meas_noise", 25.0),
        conf_ref=t.get("conf_ref", 0.7),
        conf_floor=t.get("conf_floor", 0.05),
        max_coast_frames=t.get("max_coast_frames", 15),
        gate=t.get("gate", 0.0),
        min_updates_before_gating=t.get("min_updates_before_gating", 3),
        assoc_radius=t.get("assoc_radius_px", 600.0),
    )
    homog = Homography.from_config(config)
    e = config.get("ball", {}).get("events", {})
    det = BallEventDetector(
        homography=homog,
        min_vy_px_s=(min_vy_override if min_vy_override is not None
                     else e.get("min_vy_px_s", 500.0)),
        min_vx_px_s=e.get("min_vx_px_s", 500.0),
        wall_margin_m=e.get("wall_margin_m", 0.6),
        hit_angle_deg=e.get("hit_angle_deg", 70.0),
        hit_min_speed_px_s=e.get("hit_min_speed_px_s", 1500.0),
        refractory_frames=e.get("refractory_frames", 3),
        in_out_margin_m=e.get("in_out_margin_m", 0.1),
    )

    bounces: List[Dict[str, Any]] = []
    for frame, visible, u, v in labels:
        if visible and u is not None and v is not None:
            d = BallDetection(found=True, u=u, v=v, confidence=1.0, reason="hand")
        else:
            d = BallDetection(found=False, reason="not-visible")
        track = tracker.update(d)
        ev = det.update(frame, track)
        if ev is not None and ev.type == "floor_bounce":
            bounces.append({"frame": ev.frame, "x_m": ev.x_m, "y_m": ev.y_m,
                            "in_court": ev.in_court, "u": ev.u, "v": ev.v})
    return bounces, dt


# --------------------------------------------------------------------------- #
# Matching + report
# --------------------------------------------------------------------------- #
def match(pred: List[Dict[str, Any]], truth: List[Dict[str, Any]], tol: int
          ) -> Tuple[List[Tuple[Dict, Dict]], List[Dict], List[Dict]]:
    """Greedy nearest-frame matching within +/- tol frames. Returns (matched pairs,
    false positives, false negatives)."""
    used = [False] * len(truth)
    tp: List[Tuple[Dict, Dict]] = []
    fp: List[Dict] = []
    for p in sorted(pred, key=lambda d: d["frame"]):
        best, best_d = None, tol + 1
        for i, gt in enumerate(truth):
            if used[i]:
                continue
            d = abs(p["frame"] - gt["frame"])
            if d <= tol and d < best_d:
                best, best_d = i, d
        if best is None:
            fp.append(p)
        else:
            used[best] = True
            tp.append((p, truth[best]))
    fn = [gt for i, gt in enumerate(truth) if not used[i]]
    return tp, fp, fn


def report(pred: List[Dict[str, Any]], truth: List[Dict[str, Any]], tol: int,
           tag: str) -> None:
    tp, fp, fn = match(pred, truth, tol)
    nTP, nFP, nFN = len(tp), len(fp), len(fn)
    recall = nTP / (nTP + nFN) if (nTP + nFN) else 0.0
    prec = nTP / (nTP + nFP) if (nTP + nFP) else 0.0
    f1 = 2 * prec * recall / (prec + recall) if (prec + recall) else 0.0

    print(f"\n===== bounce accuracy [{tag}]  (match tol = +/-{tol} frames) =====")
    print(f"  truth bounces : {len(truth)}")
    print(f"  predicted     : {len(pred)}")
    print(f"  TP {nTP}   FP {nFP}   FN {nFN}")
    print(f"  recall   {recall:.2f}   precision {prec:.2f}   F1 {f1:.2f}")

    if tp:
        timing = [abs(p["frame"] - gt["frame"]) for p, gt in tp]
        print(f"  timing error (frames): med {st.median(timing):.1f}  "
              f"max {max(timing)}")
        land = [((p["x_m"] - gt["x_m"]) ** 2 + (p["y_m"] - gt["y_m"]) ** 2) ** 0.5
                for p, gt in tp
                if None not in (p["x_m"], p["y_m"], gt["x_m"], gt["y_m"])]
        if land:
            print(f"  landing error (m)    : med {st.median(land):.2f}  "
                  f"max {max(land):.2f}   (n={len(land)})")
        io = [(p["in_court"] == gt["in_court"]) for p, gt in tp
              if p["in_court"] is not None and gt["in_court"] is not None]
        if io:
            print(f"  in/out agreement     : {100 * sum(io) / len(io):.0f}% "
                  f"(n={len(io)})")
    if fn:
        print(f"  missed (FN) frames   : {[gt['frame'] for gt in fn][:20]}")
    if fp:
        print(f"  false-alarm frames   : {[p['frame'] for p in fp][:20]}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    pp = sub.add_parser("propose", help="over-propose candidate bounces from hand labels")
    pp.add_argument("--labels", required=True)
    pp.add_argument("--config", required=True)
    pp.add_argument("--fps", type=float, default=20.0)
    pp.add_argument("--propose-min-vy", type=float, default=300.0,
                    help="lowered vy threshold (px/s) to OVER-propose; config uses 500")
    pp.add_argument("--out", required=True)

    mp = sub.add_parser("measure", help="score predicted bounces vs a truth csv")
    mp.add_argument("--truth", required=True)
    mp.add_argument("--tol", type=int, default=3, help="match window in frames")
    mp.add_argument("--labels", help="hand labels -> predict on the hand track (logic)")
    mp.add_argument("--config", help="config (required with --labels)")
    mp.add_argument("--fps", type=float, default=20.0)
    mp.add_argument("--pred-jsonl", help="ball_eval JSONL -> end-to-end predictions")

    args = ap.parse_args()

    if args.mode == "propose":
        config = json.load(open(args.config))
        labels = load_labels(args.labels)
        bounces, dt = events_from_labels(labels, config, args.fps,
                                         min_vy_override=args.propose_min_vy)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["frame", "type", "u", "v", "x_m", "y_m", "in_court"])
            for b in bounces:
                w.writerow([b["frame"], "floor_bounce",
                            f"{b['u']:.1f}", f"{b['v']:.1f}",
                            "" if b["x_m"] is None else f"{b['x_m']:.2f}",
                            "" if b["y_m"] is None else f"{b['y_m']:.2f}",
                            "" if b["in_court"] is None else int(b["in_court"])])
        print(f"[propose] dt={dt:.3f}s  proposed {len(bounces)} candidate floor "
              f"bounces -> {args.out}")
        print("  NEXT: extract these frames (core/bounce_frames.py), review them, and")
        print("  edit the file into your TRUTH csv (keep real, delete false, add missed).")
        return

    # measure
    truth = load_event_csv(args.truth)
    if args.pred_jsonl:
        pred = load_pred_jsonl(args.pred_jsonl)
        report(pred, truth, args.tol, "end-to-end (detector JSONL)")
    if args.labels:
        if not args.config:
            ap.error("--labels requires --config")
        config = json.load(open(args.config))
        labels = load_labels(args.labels)
        pred, _ = events_from_labels(labels, config, args.fps)
        report(pred, truth, args.tol, "logic (hand track)")
    if not args.pred_jsonl and not args.labels:
        ap.error("measure needs --pred-jsonl and/or --labels")


if __name__ == "__main__":
    main()
