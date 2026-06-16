"""core/ball_fusion.py
Phase-4 DUAL-CAMERA BALL FUSION -- the first real 3D ball.

It drives BOTH camera clips in lockstep (side-B frame = side-A frame + sync offset),
detects + tracks the ball in each, and when BOTH cameras have the ball in the same
synced frame it TRIANGULATES the two pixels into a 3D court point (x, y, z) with a
height z the single cameras can never measure. Epipolar-inconsistent matches (the two
rays don't meet) are rejected.

HONEST EXPECTATION
  Triangulation needs BOTH cameras to detect the ball the same frame. With side-1 at
  ~67% and side-2 weak (untrained), the joint rate is low, so the 3D track is SPARSE
  until side-2 is labelled/trained. The covariance is anisotropic: court-length (y) is
  the least certain axis. This module proves the pipeline; density follows detection.

Reuses the per-camera detector/tracker builders from core.ball_eval and the calibration
from core.camera_calib. Mirrors core/fusion.py's synced lockstep convention.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import cv2
import numpy as np

from core.ball_eval import _build_detector, _build_tracker
from core.ball_tracker import BallTracker
from core.camera_calib import build_camera, triangulate_ball
from utils.display import PlaybackThrottle
from utils.metrics import NumpyEncoder
from utils.video_io import ThreadedVideoReader


def _draw_fused(frame: np.ndarray, cam, X: Optional[np.ndarray], std: Optional[np.ndarray],
                speed_kmh: Optional[float]) -> None:
    """Draw the triangulated 3D ball back onto camera-A's frame, with its height."""
    if X is None:
        cv2.putText(frame, "3D ball: waiting for BOTH cameras", (20, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
        return
    u, v = cam.project(X)
    u, v = int(u), int(v)
    cv2.circle(frame, (u, v), 12, (0, 255, 0), 3)
    cv2.circle(frame, (u, v), 2, (0, 255, 0), -1)
    txt = f"3D ({X[0]:.1f},{X[1]:.1f},{X[2]:.1f})m  Z={X[2]:.2f}m"
    if speed_kmh is not None:
        txt += f"  {speed_kmh:.0f} km/h"
    cv2.putText(frame, txt, (u + 16, v - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    if std is not None:
        cv2.putText(frame, f"std(x,y,z)=({std[0]:.2f},{std[1]:.2f},{std[2]:.2f})m  "
                    f"[y = least certain]", (u + 16, v + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 0), 1)


def run_ball_fusion(
    cfg_a: Dict[str, Any],
    cfg_b: Dict[str, Any],
    show: bool = False,
    save_video: bool = False,
    max_frames: Optional[int] = None,
    start_frame: int = 0,
) -> Dict[str, Any]:
    """Triangulate the ball into 3D across the two synced cameras."""
    out_dir = cfg_a.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)

    sync = cfg_a.get("sync")
    if not sync or sync.get("offset_frames") is None:
        raise RuntimeError("config-A needs a 'sync' block (side-B = side-A + offset). "
                           "Run: python main.py --sync --config ... --config2 ...")
    offset = int(sync["offset_frames"])

    dec = cfg_a.get("decode", {})
    readerA = ThreadedVideoReader(cfg_a["source"], hw_accel=dec.get("hw_accel", True),
                                  start_frame=start_frame)
    readerB = ThreadedVideoReader(cfg_b["source"], hw_accel=dec.get("hw_accel", True),
                                  start_frame=start_frame + offset)
    fps = readerA.fps or 20.0
    dt = 1.0 / fps if fps else 0.05

    # calibrate both cameras into the shared court frame (reuse side-1 focal for side-2)
    camA, dA = build_camera(cfg_a, readerA.width, readerA.height, is_side2=False)
    camB, dB = build_camera(cfg_b, readerB.width, readerB.height, is_side2=True,
                            focal_override=dA["f"])
    print(f"[ball-fuse] calib: side-1 reproj={dA.get('reproj_px', float('nan')):.1f}px "
          f"(f={dA['f']:.0f}, stable={dA['stable']}); side-2 reproj="
          f"{dB.get('reproj_px', float('nan')):.1f}px (f={dB['f']:.0f}, "
          f"stable={dB['stable']}, focal_reused={dB.get('focal_overridden')})")

    detA, detB = _build_detector(cfg_a), _build_detector(cfg_b)
    trkA, _ = _build_tracker(cfg_a, fps)
    trkB, _ = _build_tracker(cfg_b, fps)
    if trkA is None:
        trkA = BallTracker(dt=dt)
    if trkB is None:
        trkB = BallTracker(dt=dt)

    max_reproj = float(cfg_a.get("ball", {}).get("fusion", {}).get("max_reproj_px", 40.0))

    jsonl = open(os.path.join(out_dir, "ball_fusion.jsonl"), "w")
    writer = None
    writer_path = os.path.join(out_dir, "ball_fusion_overlay.mp4")
    throttle = PlaybackThrottle(cfg_a.get("display", {}).get("playback_fps", 0))
    if show:
        cv2.namedWindow("Ball Fusion (Phase 4)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ball Fusion (Phase 4)", 1280, 720)

    n = 0
    n_tri = 0
    reproj_list = []
    heights = []
    prev_X = None
    t0 = time.time()
    while True:
        loop_t0 = time.time()
        okA, frameA = readerA.read()
        okB, frameB = readerB.read()
        if not okA or not okB:
            break

        detA.detect(frameA)
        tA = trkA.update_multi(getattr(detA, "last_candidates", []))
        detB.detect(frameB)
        tB = trkB.update_multi(getattr(detB, "last_candidates", []))

        X = std = None
        speed_kmh = None
        rec: Dict[str, Any] = {"frame": start_frame + n}
        if (tA.measured and tA.meas_u is not None
                and tB.measured and tB.meas_u is not None):
            uvA = (float(tA.meas_u), float(tA.meas_v))
            uvB = (float(tB.meas_u), float(tB.meas_v))
            res = triangulate_ball(camA, camB, uvA, uvB, max_reproj)
            if res is not None:
                X, std = res["X"], res["std"]
                n_tri += 1
                reproj_list.append(res["reproj_px"])
                heights.append(float(X[2]))
                if prev_X is not None:
                    speed_kmh = float(np.linalg.norm(X - prev_X) / dt * 3.6)  # m/s -> km/h
                prev_X = X
                rec["uvA"], rec["uvB"] = uvA, uvB
                rec["ball3d"] = {"x": float(X[0]), "y": float(X[1]), "z": float(X[2]),
                                 "std": std.tolist(), "reproj_px": res["reproj_px"],
                                 "speed_kmh": speed_kmh}
            else:
                rec["rejected"] = "epipolar"     # rays didn't meet
        jsonl.write(json.dumps(rec, cls=NumpyEncoder) + "\n")

        if show or save_video:
            _draw_fused(frameA, camA, X, std, speed_kmh)
            if save_video:
                if writer is None:
                    fh, fw = frameA.shape[:2]
                    writer = cv2.VideoWriter(writer_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (fw, fh))
                    if not writer.isOpened():
                        print(f"[ball-fuse] WARNING: VideoWriter failed -> no overlay.")
                        writer, save_video = None, False
                if writer is not None:
                    writer.write(frameA)
            if show:
                cv2.imshow("Ball Fusion (Phase 4)", frameA)
                if throttle.wait((time.time() - loop_t0) * 1000.0) == ord("q"):
                    break

        n += 1
        if max_frames is not None and n >= max_frames:
            break

    readerA.stop()
    readerB.stop()
    jsonl.close()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    hz = np.array(heights) if heights else np.array([0.0])
    summary = {
        "frames": n,
        "triangulated": n_tri,
        "triangulation_rate": round(n_tri / max(1, n), 4),
        "mean_reproj_px": round(float(np.mean(reproj_list)), 2) if reproj_list else None,
        "height_z_m": {"min": round(float(hz.min()), 2), "median": round(float(np.median(hz)), 2),
                       "max": round(float(hz.max()), 2)},
        "calib_reproj_px": {"side1": round(dA.get("reproj_px", -1), 2),
                            "side2": round(dB.get("reproj_px", -1), 2)},
        "processing_fps": round(n / elapsed, 2) if elapsed > 0 else 0.0,
    }
    with open(os.path.join(out_dir, "ball_fusion_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    print(f"\n[ball-fuse] {n} frames, triangulated 3D on {n_tri} "
          f"({summary['triangulation_rate']*100:.1f}%) -- sparse until side-2 is trained.")
    if n_tri:
        print(f"[ball-fuse] height Z (m): min={summary['height_z_m']['min']} "
              f"median={summary['height_z_m']['median']} max={summary['height_z_m']['max']}; "
              f"mean reproj={summary['mean_reproj_px']}px")
    print(f"[ball-fuse] 3D log -> {os.path.join(out_dir, 'ball_fusion.jsonl')}")
    return summary
