"""core/pipeline.py
Phase-1 main loop on ONE feed:

    read -> detect -> ROI filter -> track -> draw -> (show / save) -> measure

This module only ORCHESTRATES. The real work lives in the small single-purpose
modules it calls (detector, roi, tracker, skeleton, colors, metrics), so this
file stays readable and there is no "god-file".
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Optional

import cv2

from core.detector import PoseDetector
from core.tracker import PlayerTracker
from utils import roi as roi_utils
from utils.colors import color_for_id
from utils.metrics import Metrics
from utils.skeleton import draw_skeleton


def _draw_overlay(frame, det: Dict[str, Any], color, kp_conf_threshold: float) -> None:
    """Draw one player's box + id label + skeleton."""
    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
    tid = det["track_id"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {tid}  {det['conf']:.2f}"
    cv2.putText(frame, label, (x1, max(0, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    draw_skeleton(frame, det["keypoints"], color, kp_conf_threshold)


def run(
    config: Dict[str, Any],
    show: bool = False,
    save_video: bool = False,
    max_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the Phase-1 pipeline. Returns the metrics summary dict."""
    det_cfg = config["detection"]
    trk_cfg = config["tracker"]
    court_cfg = config.get("court", {})
    kp_thr = config.get("skeleton", {}).get("keypoint_conf_threshold", 0.5)
    out_dir = config.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)

    # --- build the components from config (no hard-coded tunables) ----------
    detector = PoseDetector(
        model_path=config["model"],
        device=config.get("device", "cuda"),
        conf_threshold=det_cfg["conf_threshold"],
        iou_threshold=det_cfg["iou_threshold"],
        imgsz=det_cfg.get("imgsz", 640),
        person_class=det_cfg.get("classes", [0])[0],
        enhance=det_cfg.get("enhance", False),
        tiling=det_cfg.get("tiling"),
    )
    tracker = PlayerTracker(**trk_cfg)
    polygon = roi_utils.to_polygon(court_cfg.get("polygon"))
    if polygon is None:
        print("[pipeline] WARNING: no court polygon set -> ROI filter is "
              "PASS-THROUGH. Run `python main.py --calibrate-roi` to define it.")

    # --- open the video source ---------------------------------------------
    cap = cv2.VideoCapture(config["source"])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {config['source']}")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or trk_cfg.get("frame_rate", 30)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if save_video:
        out_path = os.path.join(out_dir, "phase1_annotated.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in, (w, h))

    if show:
        # WINDOW_NORMAL = resizable/draggable; the 4K frame would otherwise open
        # full-size and overflow the screen. Processing stays at full res.
        cv2.namedWindow("Padel Phase-1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Padel Phase-1", 1280, 720)

    metrics = Metrics()
    t0 = time.time()
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # the Phase-1 chain, one frame at a time
        raw = detector.detect(frame)
        kept, _removed = roi_utils.filter_detections(raw, polygon)
        tracked = tracker.update(kept)

        # draw everything we kept
        for det in tracked:
            _draw_overlay(frame, det, color_for_id(det["track_id"]), kp_thr)
        if polygon is not None:
            cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)  # court outline

        metrics.update(len(raw), len(kept), tracked)

        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("Padel Phase-1", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        n += 1
        if max_frames is not None and n >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    proc_fps = round(n / elapsed, 2) if elapsed > 0 else 0.0
    metrics_path = os.path.join(out_dir, "phase1_metrics.json")
    summary = metrics.save(metrics_path, extra={
        "source": config["source"],
        "source_fps": round(float(fps_in), 2),
        "processing_fps": proc_fps,
        "device": config.get("device", "cuda"),
    })
    print(f"[pipeline] processed {n} frames in {elapsed:.1f}s "
          f"({proc_fps} FPS). metrics -> {metrics_path}")
    return summary
