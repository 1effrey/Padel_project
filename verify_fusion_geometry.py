"""verify_fusion_geometry.py  (read-only Phase-4 check, NOT a runtime module)

Question this answers:
    To fuse the two cameras into ONE global court frame we must map side-2's local
    meters into side-1's local meters. utils/homography.py claims the two frames are
    related by a 180-degree rotation about court center (5,10): (x,y) -> (10-x,20-y).
    But each camera defined its own "left/right" from opposite ends, so the true
    relation could instead be a pure mirror. If we pick the wrong one, fusion is
    garbage. So before building anything, we TEST it empirically.

How:
    At several SYNCED instants (side-1 frame F, side-2 frame F+offset) we detect
    players in both views and project their feet to each camera's local meters.
    A player standing near the NET is seen by BOTH cameras, so after the correct
    transform that player must land at (almost) the SAME global coordinate from
    both cameras. We try 4 candidate transforms and measure, for net-region
    players, the nearest-neighbour distance between side-1 points and transformed
    side-2 points. The correct transform is the one whose net-region points
    COINCIDE (smallest distance). We also save overlay minimaps for the eyeball gate.

This script imports only existing modules; it changes no production code.
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Tuple

import cv2
import numpy as np

from core.detector import PoseDetector
from utils import roi as roi_utils
from utils.court_position import player_court_position
from utils.homography import COURT_LENGTH_M, COURT_WIDTH_M, Homography
from utils.minimap import Minimap

Point = Tuple[float, float]

# candidate side-2-local -> global(side-1-local) transforms
TRANSFORMS: Dict[str, Callable[[float, float], Point]] = {
    "rot180":   lambda x, y: (COURT_WIDTH_M - x, COURT_LENGTH_M - y),  # doc's claim
    "flipY":    lambda x, y: (x, COURT_LENGTH_M - y),                  # mirror across net
    "flipX":    lambda x, y: (COURT_WIDTH_M - x, y),                   # mirror across center line
    "identity": lambda x, y: (x, y),                                   # sanity (no change)
}

NET_BAND = (6.0, 14.0)   # a player with global y in this band is near the net = shared


def _load(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _build_detector(cfg: Dict[str, Any]) -> PoseDetector:
    d = cfg["detection"]
    return PoseDetector(
        model_path=cfg["model"],
        device=cfg.get("device", "cuda"),
        conf_threshold=d["conf_threshold"],
        iou_threshold=d["iou_threshold"],
        imgsz=d.get("imgsz", 640),
        person_class=d.get("classes", [0])[0],
        enhance=d.get("enhance", False),
        tiling=d.get("tiling"),
    )


def _positions_at(cap, frame_idx: int, detector: PoseDetector, polygon,
                  homog: Homography, kp_thr: float) -> List[Point]:
    """Detect -> ROI filter -> feet-to-meters for one frame. Returns local-meter feet."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        return []
    raw = detector.detect(frame)
    kept, _ = roi_utils.filter_detections(raw, polygon)
    out = []
    for det in kept:
        pos = player_court_position(det, homog, kp_thr)
        out.append((float(pos["foot_m"][0]), float(pos["foot_m"][1])))
    return out


def _nn_dist(a: Point, pts: List[Point]) -> float:
    if not pts:
        return float("inf")
    return min(float(np.hypot(a[0] - p[0], a[1] - p[1])) for p in pts)


def main() -> None:
    cfg_a = _load("config-side1.json")
    cfg_b = _load("config-side2.json")
    offset = cfg_a.get("sync", {}).get("offset_frames", 60)
    kp_thr = cfg_a.get("skeleton", {}).get("keypoint_conf_threshold", 0.5)
    out_dir = cfg_a.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)

    det_a, det_b = _build_detector(cfg_a), _build_detector(cfg_b)
    poly_a = roi_utils.to_polygon(cfg_a.get("court", {}).get("polygon"))
    poly_b = roi_utils.to_polygon(cfg_b.get("court", {}).get("polygon"))
    hom_a = Homography.from_config(cfg_a)
    hom_b = Homography.from_config(cfg_b)
    if hom_a is None or hom_b is None:
        raise RuntimeError("Both configs need a homography for this check.")

    cap_a = cv2.VideoCapture(cfg_a["source"])
    cap_b = cv2.VideoCapture(cfg_b["source"])
    n_a = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))

    # sample synced frames spread across the clip (skip head/tail)
    frames = list(range(800, min(n_a - offset - 5, 26000), 800))
    print(f"[geo] offset={offset}  sampling {len(frames)} synced frames...")

    mm = Minimap(scale_px_per_m=30, margin_px=28)
    netband_dists: Dict[str, List[float]] = {k: [] for k in TRANSFORMS}
    saved = 0

    for F in frames:
        pa = _positions_at(cap_a, F, det_a, poly_a, hom_a, kp_thr)
        pb = _positions_at(cap_b, F + offset, det_b, poly_b, hom_b, kp_thr)
        if not pa or not pb:
            continue

        # for each candidate transform, gather net-region coincidence distances
        for name, T in TRANSFORMS.items():
            pb_g = [T(x, y) for (x, y) in pb]
            for a in pa:
                if NET_BAND[0] <= a[1] <= NET_BAND[1]:           # side-1 player near net
                    netband_dists[name].append(_nn_dist(a, pb_g))

        # save a few overlay minimaps (rot180) for the human eyeball gate
        if saved < 6:
            canvas = mm._base.copy()
            for (x, y) in pa:                                    # side-1 = green circles
                if mm._on_canvas((x, y)):
                    cv2.circle(canvas, mm.m2px((x, y)), 7, (0, 255, 0), -1)
            for (x, y) in pb:                                    # side-2 rotated = magenta squares
                gx, gy = TRANSFORMS["rot180"](x, y)
                if mm._on_canvas((gx, gy)):
                    px, py = mm.m2px((gx, gy))
                    cv2.rectangle(canvas, (px - 6, py - 6), (px + 6, py + 6), (255, 0, 255), 2)
            cv2.putText(canvas, f"f{F}  green=side1  magenta=side2(rot180)",
                        (6, mm.h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imwrite(os.path.join(out_dir, f"geo_overlay_{F}.png"), canvas)
            saved += 1

    cap_a.release()
    cap_b.release()

    print("\n[geo] net-region coincidence (side-1 near-net player -> nearest transformed side-2 player)")
    print(f"{'transform':>10} | {'samples':>7} | {'median_m':>8} | {'p25_m':>6} | {'min_m':>6}")
    best = None
    for name in TRANSFORMS:
        d = np.array([x for x in netband_dists[name] if np.isfinite(x)])
        if len(d) == 0:
            print(f"{name:>10} |   (no net-region samples)")
            continue
        med = float(np.median(d))
        p25 = float(np.percentile(d, 25))
        print(f"{name:>10} | {len(d):7d} | {med:8.2f} | {p25:6.2f} | {float(d.min()):6.2f}")
        if best is None or med < best[1]:
            best = (name, med)
    if best:
        print(f"\n[geo] BEST transform by net coincidence: '{best[0]}' (median {best[1]:.2f} m)")
        print("[geo] expect the correct one to be clearly smallest (sub-~1.5 m). "
              "Overlays saved to output/geo_overlay_*.png for the eyeball check.")


if __name__ == "__main__":
    main()
