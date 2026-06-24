"""core/ball_dual.py
DUAL-VIEW live ball tracking -- both cameras side by side + a top-down court (homography),
sharing ONE cross-camera ball.

  * ONE ball per camera: the Kalman tracker's single motion-locked track, plus an optional
    YELLOW colour gate (the padel ball is optic-yellow) to reject non-ball blobs -- lights,
    court lines, limbs. Together: "the yellow thing that is moving", not "a ball everywhere".
  * The TOP-DOWN court (built from the homography) shows the ball at its court position
    from the BEST available source: triangulated (BOTH cameras) -> side-1 floor back-
    projection -> side-2 floor back-projection. So when one camera loses the ball the other
    still places it -- the two sides are connected through the shared court frame.
  * ONE command:
      python main.py --ball-dual --config config-side1.json --config2 config-side2.json --show

Both cameras are calibrated into the SAME global court frame (side-2 composed with the 180
deg flip in build_camera), so triangulation and each camera's floor back-projection all land
in one consistent top-down map.
"""
from __future__ import annotations

import csv
import os
from collections import deque
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from core.ball_eval import _build_detector, _build_tracker
from core.camera_calib import build_camera, triangulate_ball
from utils.homography import Homography, draw_court_lines
from utils.minimap import Minimap
from utils.video_io import ThreadedVideoReader

# Optic-yellow padel ball in HSV (OpenCV H is 0-180). Lenient S/V -- the ball is small,
# bright and motion-blurred, so we only ask for a SLICE of yellow-green pixels.
_YELLOW_LO = np.array([22, 50, 50], dtype=np.uint8)
_YELLOW_HI = np.array([45, 255, 255], dtype=np.uint8)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _is_yellow(frame: np.ndarray, u: float, v: float, r: int = 12,
               frac: float = 0.06) -> bool:
    """True if the patch around (u, v) is yellowish enough to be the ball."""
    h, w = frame.shape[:2]
    x0, x1 = max(0, int(u - r)), min(w, int(u + r))
    y0, y1 = max(0, int(v - r)), min(h, int(v + r))
    if x1 <= x0 or y1 <= y0:
        return False
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    return float((cv2.inRange(hsv, _YELLOW_LO, _YELLOW_HI) > 0).mean()) >= frac


def _floor_point(cam, u: float, v: float) -> Optional[Tuple[float, float]]:
    """Back-project a pixel onto the court FLOOR (z=0) in the GLOBAL frame -- a single
    camera's best guess of where the ball is on court (valid when the ball is near the
    ground; for a high ball it's an approximation, fine for the map)."""
    d_cam = np.linalg.inv(cam.K) @ np.array([float(u), float(v), 1.0])
    d_world = cam.R.T @ d_cam                                # camera ray -> world
    if abs(d_world[2]) < 1e-9:
        return None
    C = cam.center
    s = -C[2] / d_world[2]                                   # hit z=0
    if s <= 0:
        return None
    P = C + s * d_world
    return float(P[0]), float(P[1])


def _ball_uv(track) -> Optional[Tuple[float, float]]:
    """The zero-lag 2D ball point from a tracker state (chosen detection, else predicted)."""
    if track is None:
        return None
    if track.measured and track.meas_u is not None:
        return float(track.meas_u), float(track.meas_v)
    if track.x is not None:
        return float(track.x), float(track.y)
    return None


def run_dual_view(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any],
                  max_frames: Optional[int] = None, show: bool = False,
                  save_video: bool = False, yellow_gate: bool = False,
                  panel_h: int = 540) -> str:
    """Run both cameras synced, side by side, with a shared top-down court showing one
    cross-camera ball. Returns the output video path (or '')."""
    out_dir = cfg_a.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)

    sync = cfg_a.get("sync")
    if not sync or sync.get("offset_frames") is None:
        raise RuntimeError("config-A needs a 'sync' block (side-B = side-A + offset).")
    offset = int(sync["offset_frames"])

    dec = cfg_a.get("decode", {})
    rA = ThreadedVideoReader(cfg_a["source"], hw_accel=dec.get("hw_accel", True), start_frame=0)
    rB = ThreadedVideoReader(cfg_b["source"], hw_accel=dec.get("hw_accel", True), start_frame=offset)
    fps = rA.fps or 20.0

    camA, dA = build_camera(cfg_a, rA.width, rA.height, is_side2=False)
    camB, dB = build_camera(cfg_b, rB.width, rB.height, is_side2=True, focal_override=dA["f"])
    homA, homB = Homography.from_config(cfg_a), Homography.from_config(cfg_b)
    detA, detB = _build_detector(cfg_a), _build_detector(cfg_b)
    trkA, _ = _build_tracker(cfg_a, fps)
    trkB, _ = _build_tracker(cfg_b, fps)
    mm = Minimap(scale_px_per_m=cfg_a.get("minimap", {}).get("scale_px_per_m", 30))
    trail = deque(maxlen=int(cfg_a.get("ball", {}).get("tracker", {}).get("trail_len", 30)))
    max_reproj = float(cfg_a.get("ball", {}).get("fusion", {}).get("max_reproj_px", 40.0))

    writer = None
    out_path = os.path.join(out_dir, "ball_dual.mp4")
    # per-frame ball-location log (ALWAYS written): each camera's image pixel + the shared
    # court position x/y (metres) and z (height, real only when both cameras triangulate).
    csv_path = os.path.join(out_dir, "ball_dual_locations.csv")
    cf = open(csv_path, "w", newline="")
    cw = csv.writer(cf)
    cw.writerow(["frame", "cam1_detected", "cam1_x_px", "cam1_y_px",
                 "cam2_detected", "cam2_x_px", "cam2_y_px",
                 "court_x_m", "court_y_m", "court_z_m", "source"])
    if show:
        cv2.namedWindow("Ball Dual (side-1 | side-2 | court)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ball Dual (side-1 | side-2 | court)", 1700, 600)
    delay = max(1, int(1000.0 / fps)) if show else 1

    def fit(img):
        return cv2.resize(img, (int(img.shape[1] * panel_h / img.shape[0]), panel_h))

    n = n_both = n_a = n_b = 0
    while True:
        okA, fA = rA.read()
        okB, fB = rB.read()
        if not okA or not okB:
            break

        detA.detect(fA)
        tA = trkA.update_multi(getattr(detA, "last_candidates", []))
        detB.detect(fB)
        tB = trkB.update_multi(getattr(detB, "last_candidates", []))

        uvA, uvB = _ball_uv(tA), _ball_uv(tB)
        measA = tA.measured and uvA is not None
        measB = tB.measured and uvB is not None
        seenA = measA and (not yellow_gate or _is_yellow(fA, *uvA))   # "yellow + moving"
        seenB = measB and (not yellow_gate or _is_yellow(fB, *uvB))

        # --- ball court position. TRIANGULATION (both cameras) is the reliable 3D: it places
        #     the ball correctly even when AIRBORNE, and gives x, y AND the real height z.
        #     A SINGLE camera can only FLOOR-project (homography, z=0) -- correct for a ball
        #     near the ground, but it flies far off-court for an airborne ball. So a single-
        #     cam position is kept ONLY if it lands on/near the court; a wildly off-court
        #     projection (an airborne ball) is dropped rather than placed wrong. z stays
        #     blank unless triangulated -- never assumed 0. ---
        court = source = color = None
        court_z = None
        if seenA and seenB:
            res = triangulate_ball(camA, camB, uvA, uvB, max_reproj)
            if res is not None:
                court = (float(res["X"][0]), float(res["X"][1]))    # triangulated x, y (in-bounds)
                court_z = float(res["X"][2])                        # real triangulated height (m)
                source, color, n_both = "both (3D)", (0, 220, 0), n_both + 1
        if court is None and seenA and homA is not None:
            p = homA.pixel_to_meters(uvA)
            if -2.0 <= p[0] <= 12.0 and -2.0 <= p[1] <= 22.0:       # on/near court only
                court, source, color, n_a = p, "side-1", (0, 220, 220), n_a + 1
        if court is None and seenB and homB is not None:
            p = homB.pixel_to_meters(uvB)
            if -2.0 <= p[0] <= 12.0 and -2.0 <= p[1] <= 22.0:
                court, source, color, n_b = p, "side-2", (255, 180, 0), n_b + 1

        # --- per-frame ball-location log: cam1/cam2 image px + shared court x/y/z (m) ---
        cw.writerow([
            n,
            1 if seenA else 0,
            f"{uvA[0]:.1f}" if seenA else "", f"{uvA[1]:.1f}" if seenA else "",
            1 if seenB else 0,
            f"{uvB[0]:.1f}" if seenB else "", f"{uvB[1]:.1f}" if seenB else "",
            f"{court[0]:.3f}" if court else "", f"{court[1]:.3f}" if court else "",
            f"{court_z:.3f}" if court_z is not None else "",
            source or "lost",
        ])

        # --- court overlay on the full-res frames, THEN scale to panels ---
        if homA is not None:
            draw_court_lines(fA, homA, thickness=2)
        if homB is not None:
            draw_court_lines(fB, homB, thickness=2)
        pA, pB = fit(fA), fit(fB)
        sA, sB = panel_h / fA.shape[0], panel_h / fB.shape[0]
        # ball drawn on the SCALED panels (with a crosshair) so it stays clearly visible
        if seenA:
            ca = (int(uvA[0] * sA), int(uvA[1] * sA))
            cv2.circle(pA, ca, 13, (255, 0, 255), 3)
            cv2.line(pA, (ca[0] - 22, ca[1]), (ca[0] + 22, ca[1]), (255, 0, 255), 1)
            cv2.line(pA, (ca[0], ca[1] - 22), (ca[0], ca[1] + 22), (255, 0, 255), 1)
        if seenB:
            cb = (int(uvB[0] * sB), int(uvB[1] * sB))
            cv2.circle(pB, cb, 13, (255, 0, 255), 3)
            cv2.line(pB, (cb[0] - 22, cb[1]), (cb[0] + 22, cb[1]), (255, 0, 255), 1)
            cv2.line(pB, (cb[0], cb[1] - 22), (cb[0], cb[1] + 22), (255, 0, 255), 1)
        cv2.putText(pA, "SIDE 1", (12, 32), _FONT, 1.0, (255, 255, 255), 2)
        cv2.putText(pB, "SIDE 2", (12, 32), _FONT, 1.0, (255, 255, 255), 2)

        # --- top-down court with the one shared ball + recent trajectory trail ---
        trail.append(court)                              # (x, y) m, or None when lost
        mimg = mm._base.copy()
        # draw the trail (recent positions), breaking the line at lost/off-court frames
        prev = None
        for p in trail:
            if p is not None and mm._on_canvas(p):
                cur = mm.m2px(p)
                if prev is not None:
                    cv2.line(mimg, prev, cur, (0, 165, 255), 2, cv2.LINE_AA)   # orange trail
                prev = cur
            else:
                prev = None                              # break trail across a gap
        if court is not None and mm._on_canvas(court):
            cx, cy = mm.m2px(court)
            cv2.circle(mimg, (cx, cy), 10, color, -1)
            cv2.circle(mimg, (cx, cy), 10, (255, 255, 255), 2)
        cv2.putText(mimg, f"ball: {source or 'lost'}", (6, mm.h - 8), _FONT, 0.5,
                    color or (120, 120, 120), 1)

        comp = cv2.hconcat([pA, pB,
                            cv2.resize(mimg, (int(mm.w * panel_h / mm.h), panel_h))])

        if save_video:
            if writer is None:
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                         (comp.shape[1], comp.shape[0]))
            writer.write(comp)
        if show:
            cv2.imshow("Ball Dual (side-1 | side-2 | court)", comp)
            if (cv2.waitKey(delay) & 0xFF) == ord("q"):
                break

        n += 1
        if max_frames is not None and n >= max_frames:
            break

    rA.stop()
    rB.stop()
    cf.close()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"[ball-dual] {n} frames: ball placed on court via both-cam 3D {n_both}, "
          f"side-1-only {n_a}, side-2-only {n_b} "
          f"(connected {100*(n_both+n_a+n_b)/max(1,n):.0f}% of frames).")
    print(f"[ball-dual]   locations -> {csv_path}")
    if save_video:
        print(f"[ball-dual]   video -> {out_path}")
    return out_path if save_video else ""


if __name__ == "__main__":
    import json
    import sys

    cpa = sys.argv[1] if len(sys.argv) > 1 else "config-side1.json"
    cpb = sys.argv[2] if len(sys.argv) > 2 else "config-side2.json"
    mf = int(sys.argv[3]) if len(sys.argv) > 3 else None
    with open(cpa, "r", encoding="utf-8") as f:
        ca = json.load(f)
    with open(cpb, "r", encoding="utf-8") as f:
        cb = json.load(f)
    run_dual_view(ca, cb, max_frames=mf, save_video=True)
