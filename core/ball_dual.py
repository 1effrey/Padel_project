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

from core.ball_eval import _build_detector, _build_events, _build_tracker, _wrists
from core.camera_calib import build_camera, triangulate_ball
from utils.homography import COURT_LENGTH_M, Homography, draw_court_lines
from utils.minimap import Minimap
from utils.video_io import ThreadedVideoReader

# Optic-yellow padel ball in HSV (OpenCV H is 0-180). Lenient S/V -- the ball is small,
# bright and motion-blurred, so we only ask for a SLICE of yellow-green pixels.
_YELLOW_LO = np.array([22, 50, 50], dtype=np.uint8)
_YELLOW_HI = np.array([45, 255, 255], dtype=np.uint8)
_FONT = cv2.FONT_HERSHEY_SIMPLEX

# hit-marker colours (BGR) + on-screen labels, by event type.
_HIT_COLORS = {
    "player_hit": (255, 0, 255),    # magenta
    "net_hit": (255, 255, 0),       # cyan
    "wall_bounce": (0, 165, 255),   # orange  -> WALL (in play)
    "fence_hit": (0, 0, 255),       # red     -> FENCE = OUT
}
_HIT_LABELS = {
    "player_hit": "player hit",
    "net_hit": "net hit",
    "wall_bounce": "wall hit",
    "fence_hit": "fence OUT",
}


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


def _draw_ball_trail(panel: np.ndarray, trail, scale: float,
                     color: Tuple[int, int, int] = (0, 255, 255)) -> None:
    """Draw the ball's recent path as a fading, tapering tail on a SCALED panel, with a
    bright head dot at the newest point -- easier to follow than a static marker.

    PURELY visual: it just reads a deque of full-res (u, v) ball pixels (None entries
    are frames the ball was not seen and BREAK the line). Detection/tracking untouched.
    Older segments are drawn thinner so the tail visibly fades toward the past."""
    pts = list(trail)
    n = len(pts)
    prev = None
    for i, p in enumerate(pts):
        if p is None:                                  # gap -> break the line here
            prev = None
            continue
        cur = (int(p[0] * scale), int(p[1] * scale))
        if prev is not None:
            thick = max(1, int(1 + 4 * (i / max(1, n - 1))))   # taper: old=thin, new=thick
            cv2.line(panel, prev, cur, color, thick, cv2.LINE_AA)
        prev = cur
    if prev is not None:                               # head dot on the current ball
        cv2.circle(panel, prev, 7, color, -1)
        cv2.circle(panel, prev, 7, (255, 255, 255), 1)


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
    _trail_len = int(cfg_a.get("ball", {}).get("tracker", {}).get("trail_len", 30))
    trail = deque(maxlen=_trail_len)
    # per-camera 2D pixel trails (full-res ball (u,v), None = not seen) -> the on-panel
    # ball tail. Purely visual; same length as the court trail.
    ball_trailA, ball_trailB = deque(maxlen=_trail_len), deque(maxlen=_trail_len)
    max_reproj = float(cfg_a.get("ball", {}).get("fusion", {}).get("max_reproj_px", 40.0))
    # per-camera bounce detectors. NEAR-HALF OWNERSHIP: each camera places a floor bounce
    # ONLY in the half nearest it (side-1: y<net_y, side-2: y>=net_y) where its floor
    # projection is accurate -- the far half is the OTHER camera's job (validated: side-1
    # fires ~all its bounces in its far half, where localisation is unreliable).
    evA, evB = _build_events(cfg_a, homA), _build_events(cfg_b, homB)
    net_y = COURT_LENGTH_M / 2.0
    bounces = []                                     # accumulated landing-map points (x, y, in)

    # pose -> wrists, needed for PLAYER HITS (a deflection only counts as a player hit at
    # racket reach from a wrist; a net volley benefits too). ONE model instance serves both
    # camera frames. Toggle via cfg_a ball.events.use_pose (default True).
    pose = None
    ev_cfg = cfg_a.get("ball", {}).get("events", {})
    if (evA is not None or evB is not None) and ev_cfg.get("use_pose", True):
        from core.detector import PoseDetector
        dc = cfg_a.get("detection", {})
        pose = PoseDetector(model_path=cfg_a["model"], device=cfg_a.get("device", "cuda"),
                            conf_threshold=dc.get("conf_threshold", 0.3),
                            iou_threshold=dc.get("iou_threshold", 0.5),
                            imgsz=dc.get("imgsz", 1280))
    kp_conf = cfg_a.get("skeleton", {}).get("keypoint_conf_threshold", 0.5)

    # HIT accumulators: drawn on the panels, tallied, and logged. De-dup is across BOTH
    # cameras -- one real hit can be seen by both -> count it ONCE. wall_bounce = a glass /
    # off-floor deflection (in play); fence_hit = a metal-fence deflection (OUT).
    _HIT_TYPES = ("player_hit", "net_hit", "wall_bounce", "fence_hit")
    hit_counts = {t: 0 for t in _HIT_TYPES}
    recentA, recentB = deque(maxlen=60), deque(maxlen=60)
    last_hit_frame = {t: -10 ** 9 for t in _HIT_TYPES}
    hit_dedup = int(cfg_a.get("ball", {}).get("fusion", {}).get("hit_dedup_frames", 5))
    hcsv_path = os.path.join(out_dir, "ball_dual_hits.csv")
    hcf = open(hcsv_path, "w", newline="")
    hcw = csv.writer(hcf)
    hcw.writerow(["frame", "camera", "type", "u_px", "v_px", "hand"])

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
    # accurate bounce/landing log -- floor bounces only, each from the camera that owns
    # that half (so the court x/y is the trustworthy ground-contact position).
    bcsv_path = os.path.join(out_dir, "ball_dual_bounces.csv")
    bf = open(bcsv_path, "w", newline="")
    bw = csv.writer(bf)
    bw.writerow(["frame", "camera", "court_x_m", "court_y_m", "in_court"])
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

        # wrists per camera (only when that camera has a ball -> skip pose on empty frames)
        wristsA = (_wrists(pose.detect(fA), kp_conf)
                   if (pose is not None and tA is not None and tA.x is not None) else None)
        wristsB = (_wrists(pose.detect(fB), kp_conf)
                   if (pose is not None and tB is not None and tB.x is not None) else None)

        # --- events: floor bounces (landing map) + player/net HITS, per camera ---
        eA = evA.update(n, tA, wristsA) if evA is not None else None
        eB = evB.update(n, tB, wristsB) if evB is not None else None
        for ev, cam_name in ((eA, "side-1"), (eB, "side-2")):
            if ev is None or ev.type != "floor_bounce" or ev.x_m is None:
                continue
            owns = (ev.y_m < net_y) if cam_name == "side-1" else (ev.y_m >= net_y)
            if owns:
                bounces.append((ev.x_m, ev.y_m, ev.in_court))
                bw.writerow([n, cam_name, f"{ev.x_m:.3f}", f"{ev.y_m:.3f}",
                             "" if ev.in_court is None else int(ev.in_court)])

        # --- PLAYER / NET / WALL / FENCE hits: counted ONCE across both cameras (a hit one
        #     camera sees, the other often sees too). The firing camera's panel shows it. ---
        for ev, cam_name, recent in ((eA, "side-1", recentA), (eB, "side-2", recentB)):
            if ev is None or ev.type not in _HIT_TYPES:
                continue
            if n - last_hit_frame[ev.type] < hit_dedup:
                continue                              # duplicate of the other camera's hit
            last_hit_frame[ev.type] = n
            hit_counts[ev.type] += 1
            recent.append((n, ev))
            hcw.writerow([n, cam_name, ev.type, f"{ev.u:.1f}", f"{ev.v:.1f}",
                          ev.player_hand or ""])

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
        # ball shown as a FADING TRAIL on each panel (newest = bright head dot) instead of
        # a static crosshair, so the moving ball is easy to follow. Visual only -- the
        # underlying detection/tracking is unchanged; we only record where it was seen.
        ball_trailA.append((uvA[0], uvA[1]) if seenA else None)
        ball_trailB.append((uvB[0], uvB[1]) if seenB else None)
        _draw_ball_trail(pA, ball_trailA, sA)
        _draw_ball_trail(pB, ball_trailB, sB)
        cv2.putText(pA, "SIDE 1", (12, 32), _FONT, 1.0, (255, 255, 255), 2)
        cv2.putText(pB, "SIDE 2", (12, 32), _FONT, 1.0, (255, 255, 255), 2)

        # recent hit markers on each panel, kept ~1 s. PLAYER=magenta, NET=cyan,
        # WALL=orange (in play), FENCE=red (OUT).
        for recent, panel, scale in ((recentA, pA, sA), (recentB, pB, sB)):
            for fno, ev in recent:
                age = n - fno
                if age > 20:
                    continue
                hcol = _HIT_COLORS.get(ev.type, (255, 255, 255))
                hx, hy = int(ev.u * scale), int(ev.v * scale)
                cv2.drawMarker(panel, (hx, hy), hcol, cv2.MARKER_DIAMOND, 26, 3)
                if age <= 7:
                    cv2.putText(panel, _HIT_LABELS.get(ev.type, ev.type).upper(), (hx + 14, hy),
                                _FONT, 0.6, hcol, 2)

        # --- top-down court with the one shared ball + recent trajectory trail ---
        # trail ONLY the RELIABLE (triangulated, both-camera) positions -- single-camera
        # floor projections jump for an airborne ball, so they are NOT trailed (the ball
        # DOT below still shows the best-available position every frame).
        trail.append(court if source == "both (3D)" else None)
        mimg = mm._base.copy()
        # accumulated bounce LANDINGS (persistent diamonds): green = in, red = out
        for bx, by, inc in bounces:
            if mm._on_canvas((bx, by)):
                bpx, bpy = mm.m2px((bx, by))
                bcol = (0, 0, 255) if inc is False else (0, 220, 0)
                cv2.drawMarker(mimg, (bpx, bpy), bcol, cv2.MARKER_DIAMOND, 12, 2)
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

        # running tally (top-right of the composite): hits + accumulated landings
        for i, (lab, val, hcol) in enumerate(
                (("PLAYER HITS", hit_counts["player_hit"], _HIT_COLORS["player_hit"]),
                 ("NET HITS", hit_counts["net_hit"], _HIT_COLORS["net_hit"]),
                 ("WALL HITS", hit_counts["wall_bounce"], _HIT_COLORS["wall_bounce"]),
                 ("FENCE OUT", hit_counts["fence_hit"], _HIT_COLORS["fence_hit"]),
                 ("BOUNCES", len(bounces), (0, 220, 0)))):
            txt = f"{lab}: {val}"
            (tw, _t), _bl = cv2.getTextSize(txt, _FONT, 0.8, 2)
            tx, ty = comp.shape[1] - tw - 16, 34 + i * 34
            cv2.putText(comp, txt, (tx, ty), _FONT, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(comp, txt, (tx, ty), _FONT, 0.8, hcol, 2, cv2.LINE_AA)

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
    bf.close()
    hcf.close()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"[ball-dual] {n} frames: ball placed on court via both-cam 3D {n_both}, "
          f"side-1-only {n_a}, side-2-only {n_b} "
          f"(connected {100*(n_both+n_a+n_b)/max(1,n):.0f}% of frames).")
    print(f"[ball-dual]   locations -> {csv_path}")
    print(f"[ball-dual]   bounces   -> {bcsv_path} ({len(bounces)} near-half landings)")
    print(f"[ball-dual]   hits      -> {hcsv_path} "
          f"({hit_counts['player_hit']} player, {hit_counts['net_hit']} net, "
          f"{hit_counts['wall_bounce']} wall, {hit_counts['fence_hit']} fence-OUT)")
    if save_video:
        print(f"[ball-dual]   video -> {out_path}")
    return out_path if save_video else ""


if __name__ == "__main__":
    import argparse
    import json

    # Flag-based CLI: the two config paths stay positional, but --show /
    # --save-video / --max-frames are real flags (not parsed by position).
    # On a headless box (RunPod) pass --save-video; on your laptop pass --show.
    ap = argparse.ArgumentParser(
        description="Dual-view ball tracking (side-1 | side-2 | top-down court)")
    ap.add_argument("config_a", nargs="?", default="config-side1.json",
                    help="side-1 config (has the sync block)")
    ap.add_argument("config_b", nargs="?", default="config-side2.json",
                    help="side-2 config")
    ap.add_argument("--show", action="store_true", help="display the live window")
    ap.add_argument("--save-video", action="store_true",
                    help="write output/ball_dual.mp4")
    ap.add_argument("--max-frames", type=int, default=None,
                    help="stop after N frames (quick spot-check)")
    args = ap.parse_args()

    with open(args.config_a, "r", encoding="utf-8") as f:
        ca = json.load(f)
    with open(args.config_b, "r", encoding="utf-8") as f:
        cb = json.load(f)
    run_dual_view(ca, cb, max_frames=args.max_frames,
                  show=args.show, save_video=args.save_video)
