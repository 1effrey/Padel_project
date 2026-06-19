"""core/ball_3d.py
PHASE 4 + 5 INTEGRATED -- the end-to-end 3D ball pipeline.

This wires the dual-camera triangulation (Phase 4) into ONE synced pass together with
side-1 detection / tracking / bounce events, writing a single unified per-frame log
(output/ball_3d.jsonl). That log carries everything the Phase-5 physics needs -- the
side-1 track + events AND the triangulated 3D points -- so the projectile EKF + smoother
can run straight off it (next stage) to produce the final 3D trajectory.

It is the integrated successor to running --ball-eval + --ball-fuse + the standalone
`core.ball_physics` separately: one command, one pass.

  STAGE 4 (this file, so far): unified dual-cam detect -> track -> events -> triangulate
                               -> output/ball_3d.jsonl.
  STAGE 5 (next): feed that log to the projectile EKF + RTS smoother + write the 3D CSV.

The unified log is deliberately a superset of both old logs: it has ball_eval's per-frame
fields (top-level u/v + `track` + `event`) AND ball_fusion's `ball3d` -- so the existing
physics reads it with no change (`run_ekf(ball_3d.jsonl, fusion_path=ball_3d.jsonl)`).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

from core.ball_eval import _build_detector, _build_tracker, _build_events
from core.camera_calib import build_camera, triangulate_ball
from utils.homography import Homography
from utils.metrics import NumpyEncoder
from utils.video_io import ThreadedVideoReader


def run_stage4(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any],
               max_frames: Optional[int] = None) -> Tuple[str, Any, float]:
    """Phase-4 stage: one synced pass over both cameras (side-B frame = side-A + offset).
    Each frame runs side-1 detect/track/events and side-2 detect/track, then triangulates
    where BOTH cameras measured. Writes a unified output/ball_3d.jsonl. Returns
    (jsonl_path, side-1 camera, fps)."""
    out_dir = cfg_a.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)

    sync = cfg_a.get("sync")
    if not sync or sync.get("offset_frames") is None:
        raise RuntimeError("config-A needs a 'sync' block (side-B = side-A + offset). "
                           "Run: python main.py --sync --config ... --config2 ...")
    offset = int(sync["offset_frames"])

    dec = cfg_a.get("decode", {})
    readerA = ThreadedVideoReader(cfg_a["source"], hw_accel=dec.get("hw_accel", True),
                                  start_frame=0)
    readerB = ThreadedVideoReader(cfg_b["source"], hw_accel=dec.get("hw_accel", True),
                                  start_frame=offset)
    fps = readerA.fps or 20.0

    # Calibrate both cameras into the shared court frame (reuse side-1 focal for side-2).
    camA, dA = build_camera(cfg_a, readerA.width, readerA.height, is_side2=False)
    camB, dB = build_camera(cfg_b, readerB.width, readerB.height, is_side2=True,
                            focal_override=dA["f"])
    print(f"[ball-3d] calib: side-1 reproj={dA.get('reproj_px', float('nan')):.1f}px, "
          f"side-2 reproj={dB.get('reproj_px', float('nan')):.1f}px (sync offset {offset})")

    detA, detB = _build_detector(cfg_a), _build_detector(cfg_b)
    trkA, _ = _build_tracker(cfg_a, fps)
    trkB, _ = _build_tracker(cfg_b, fps)
    eventsA = _build_events(cfg_a, Homography.from_config(cfg_a))
    max_reproj = float(cfg_a.get("ball", {}).get("fusion", {}).get("max_reproj_px", 40.0))

    path = os.path.join(out_dir, "ball_3d.jsonl")
    jsonl = open(path, "w")
    n = n_tri = 0
    ev_counts: Dict[str, int] = {}
    while True:
        okA, frameA = readerA.read()
        okB, frameB = readerB.read()
        if not okA or not okB:
            break

        det_a = detA.detect(frameA)
        tA = trkA.update_multi(getattr(detA, "last_candidates", []))
        detB.detect(frameB)
        tB = trkB.update_multi(getattr(detB, "last_candidates", []))
        ev = eventsA.update(n, tA) if eventsA is not None else None
        if ev is not None:
            ev_counts[ev.type] = ev_counts.get(ev.type, 0) + 1

        rec: Dict[str, Any] = {"frame": n, **det_a.to_dict()}   # top-level u/v for physics
        if tA is not None:
            rec["track"] = tA.to_dict()
        if ev is not None:
            rec["event"] = ev.to_dict()
        # Triangulate where BOTH cameras measured the ball this synced frame.
        if (tA.measured and tA.meas_u is not None
                and tB.measured and tB.meas_u is not None):
            res = triangulate_ball(camA, camB, (float(tA.meas_u), float(tA.meas_v)),
                                   (float(tB.meas_u), float(tB.meas_v)), max_reproj)
            if res is not None:
                X, std = res["X"], res["std"]
                rec["ball3d"] = {"x": float(X[0]), "y": float(X[1]), "z": float(X[2]),
                                 "std": std.tolist(), "reproj_px": res["reproj_px"]}
                n_tri += 1
        jsonl.write(json.dumps(rec, cls=NumpyEncoder) + "\n")

        n += 1
        if max_frames is not None and n >= max_frames:
            break
    jsonl.close()
    readerA.stop()
    readerB.stop()
    print(f"[ball-3d] STAGE 4: {n} frames, triangulated 3D on {n_tri} "
          f"({100 * n_tri / max(1, n):.1f}%), events {ev_counts or 'none'}")
    print(f"[ball-3d]   unified log -> {path}  (side-1 track + events + 3D in one file)")
    return path, camA, fps


# --------------------------------------------------------------------------- #
# STAGE 5 -- physics + final output
# --------------------------------------------------------------------------- #
def _build_output_rows(traj, jsonl_path: str, fps: float) -> list:
    """Combine the physics 3D trajectory with the unified log into the final per-frame
    rows: 2D (u, v) + 3D (x, y, z) + speed (km/h) + provenance + event. This is the
    deliverable schema (docs/ball_tracking_output_template.csv)."""
    log: Dict[int, Dict[str, Any]] = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                r = json.loads(line)
                log[int(r["frame"])] = r
    dt = 1.0 / fps if fps else 0.05
    rows, prev = [], None
    for (f, x, y, z) in traj:
        r = log.get(f, {})
        tr = r.get("track", {}) or {}
        if tr.get("measured") and tr.get("meas_u") is not None:
            u, v = tr["meas_u"], tr["meas_v"]               # chosen detection (zero-lag)
        else:
            u, v = tr.get("x"), tr.get("y")                 # tracker fills the gap
        speed = ""
        if prev is not None:
            d = ((x - prev[0]) ** 2 + (y - prev[1]) ** 2 + (z - prev[2]) ** 2) ** 0.5
            speed = round(d / dt * 3.6, 1)                  # m/s -> km/h
        prev = (x, y, z)
        source = ("triangulated" if "ball3d" in r
                  else "measured" if tr.get("measured") else "gap")
        rows.append({
            "frame": f,
            "u": round(u, 1) if u is not None else "",
            "v": round(v, 1) if v is not None else "",
            "x_m": round(x, 3), "y_m": round(y, 3), "z_m": round(z, 3),
            "speed_kmh": speed, "source": source,
            "event": (r.get("event") or {}).get("type", ""),
        })
    return rows


def _write_csv(rows: list, path: str) -> None:
    import csv
    cols = ["frame", "u", "v", "x_m", "y_m", "z_m", "speed_kmh", "source", "event"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def _draw_overlay(cfg_a: Dict[str, Any], rows: list, cam, fps: float,
                  path: Optional[str], max_frames: Optional[int],
                  show: bool = False) -> None:
    """Back-project the recovered 3D ball onto side-1's video with height + speed. Writes
    `path` if given, and/or shows a live window if `show` (press q to quit)."""
    import cv2
    import numpy as np
    by_frame = {r["frame"]: r for r in rows}
    colors = {"triangulated": (0, 220, 0), "measured": (0, 220, 220), "gap": (0, 140, 255)}
    reader = ThreadedVideoReader(cfg_a["source"],
                                 hw_accel=cfg_a.get("decode", {}).get("hw_accel", True),
                                 start_frame=0)
    writer = None
    if show:
        cv2.namedWindow("Ball 3D", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ball 3D", 1280, 720)
    delay = max(1, int(1000.0 / fps)) if (show and fps) else 1   # ~real-time playback
    n = 0
    while True:
        ok, frame = reader.read()
        if not ok:
            break
        r = by_frame.get(n)
        if r is not None:
            uv = cam.project(np.array([r["x_m"], r["y_m"], r["z_m"]], dtype=float))
            if np.all(np.isfinite(uv)):
                p = (int(uv[0]), int(uv[1]))
                col = colors.get(r["source"], (200, 200, 200))
                cv2.circle(frame, p, 12, col, 3)
                txt = f"h={r['z_m']:.1f}m"
                if r["speed_kmh"] != "":
                    txt += f"  {r['speed_kmh']:.0f}km/h"
                cv2.putText(frame, txt, (p[0] + 16, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2)
                if r["event"]:
                    cv2.putText(frame, r["event"], (p[0] + 16, p[1] + 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        if path is not None:
            if writer is None:
                h, w = frame.shape[:2]
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(frame)
        if show:
            cv2.imshow("Ball 3D", frame)
            if (cv2.waitKey(delay) & 0xFF) == ord("q"):
                break
        n += 1
        if max_frames is not None and n >= max_frames:
            break
    reader.stop()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()


def run_ball_3d(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any],
                max_frames: Optional[int] = None, save_video: bool = False,
                show: bool = False) -> str:
    """END-TO-END 3D ball pipeline, one command:
        Phase 4  run_stage4  -- dual-cam detect/track/events/triangulate -> unified log
        Phase 5  run_ekf     -- projectile EKF + RTS smoother on that log -> 3D trajectory
        output   -> ball_3d_trajectory.csv  (+ optional ball_3d_overlay.mp4)
    Returns the CSV path."""
    from core.ball_physics import run_ekf
    out_dir = cfg_a.get("output", {}).get("dir", "output")
    jsonl_path, _camA, fps = run_stage4(cfg_a, cfg_b, max_frames)        # Phase 4
    traj, cam = run_ekf(jsonl_path, cfg_a, fps=fps, fusion_path=jsonl_path)   # Phase 5
    rows = _build_output_rows(traj, jsonl_path, fps)
    csv_path = os.path.join(out_dir, "ball_3d_trajectory.csv")
    _write_csv(rows, csv_path)
    print(f"[ball-3d] STAGE 5: wrote {len(rows)} frames -> {csv_path}")
    print(f"[ball-3d]   columns: frame,u,v,x_m,y_m,z_m,speed_kmh,source,event")
    if save_video or show:
        vid = os.path.join(out_dir, "ball_3d_overlay.mp4") if save_video else None
        if show:
            print("[ball-3d]   live window opening (press q to quit)...")
        _draw_overlay(cfg_a, rows, cam, fps, vid, max_frames, show=show)
        if vid is not None:
            print(f"[ball-3d]   overlay -> {vid}")
    return csv_path


if __name__ == "__main__":
    import sys

    cpa = sys.argv[1] if len(sys.argv) > 1 else "config-side1.json"
    cpb = sys.argv[2] if len(sys.argv) > 2 else "config-side2.json"
    mf = int(sys.argv[3]) if len(sys.argv) > 3 else None
    with open(cpa, "r", encoding="utf-8") as f:
        ca = json.load(f)
    with open(cpb, "r", encoding="utf-8") as f:
        cb = json.load(f)
    print(f"[ball-3d] END-TO-END: {ca['source']} + {cb['source']}")
    run_ball_3d(ca, cb, max_frames=mf, save_video=True)
