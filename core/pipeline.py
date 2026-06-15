"""core/pipeline.py
Phase-1 main loop on ONE feed:

    read -> detect -> ROI filter -> track -> draw -> (show / save) -> measure

This module only ORCHESTRATES. The real work lives in the small single-purpose
modules it calls (detector, roi, tracker, skeleton, colors, metrics), so this
file stays readable and there is no "god-file".
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import cv2

from core.detector import PoseDetector
from core.identity import IdentityManager
from core.tracker import PlayerTracker
from utils import roi as roi_utils
from utils.colors import color_for_id
from utils.court_position import player_court_position
from utils.display import PlaybackThrottle
from utils.homography import Homography, draw_court_lines
from utils.metrics import Metrics, NumpyEncoder
from utils.minimap import Minimap
from utils.skeleton import draw_skeleton


def _draw_overlay(frame, det: Dict[str, Any], disp_id: int, color,
                  kp_conf_threshold: float) -> None:
    """Draw one player's box + id label + skeleton.

    `disp_id` is the id to SHOW: the stable ReID id (1..4) when ReID is on, or the
    raw ByteTrack id otherwise. When a ReID id is present we also tack on the raw
    track id (t..) so a human can see which fragments were merged -- this is part
    of the quality gate.
    """
    x1, y1, x2, y2 = [int(v) for v in det["bbox"]]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    pid = det.get("player_id")
    if pid is not None:
        label = f"P{pid} (t{det['track_id']})  {det['conf']:.2f}"
    else:
        label = f"ID {disp_id}  {det['conf']:.2f}"
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

    # Optional: court markings + top-down view. None until calibrated -> skipped.
    homog = Homography.from_config(config)
    minimap = None
    inside_margin = 0.0
    positions_writer = None
    if homog is None:
        print("[pipeline] note: no homography set -> court lines / top-down view "
              "not drawn. Run `python main.py --calibrate-homography` to define it.")
    else:
        mm_cfg = config.get("minimap", {})
        inside_margin = mm_cfg.get("inside_margin_m", 0.0)
        minimap = Minimap(scale_px_per_m=mm_cfg.get("scale_px_per_m", 30),
                          margin_px=mm_cfg.get("margin_px", 28))
        positions_writer = open(os.path.join(out_dir, "phase1_positions.jsonl"), "w")

    # Optional: the 4-identity ReID layer. Toggled via config["reid"]["enabled"];
    # OFF by default so existing runs are unaffected. It needs court position, so
    # it is only created when a homography exists (otherwise we warn and skip).
    reid_cfg = config.get("reid", {})
    identity = None
    if reid_cfg.get("enabled", False):
        if homog is None:
            print("[pipeline] WARNING: reid enabled but no homography set -> "
                  "ReID skipped (needs court position). Calibrate homography first.")
        else:
            identity = IdentityManager(reid_cfg, homog, config["source"], out_dir, kp_thr)
            print(f"[pipeline] ReID ON -> 4 fixed identities "
                  f"(w_pos={identity.w_pos}, w_color={identity.w_color}, "
                  f"match_threshold={identity.match_threshold}).")

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

    # Playback-speed knob for the preview window. config["display"]["playback_fps"]:
    # 0/missing -> run as fast as possible (old behaviour); >0 -> throttle to that fps.
    throttle = PlaybackThrottle(config.get("display", {}).get("playback_fps", 0))

    if show:
        # WINDOW_NORMAL = resizable/draggable; the 4K frame would otherwise open
        # full-size and overflow the screen. Processing stays at full res.
        cv2.namedWindow("Padel Phase-1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Padel Phase-1", 1280, 720)

    metrics = Metrics()
    t0 = time.time()
    n = 0
    while True:
        loop_t0 = time.time()   # start of this frame's work (for playback throttle)
        ok, frame = cap.read()
        if not ok:
            break

        # the Phase-1 chain, one frame at a time
        raw = detector.detect(frame)
        kept, _removed = roi_utils.filter_detections(raw, polygon)
        tracked = tracker.update(kept)

        # ReID layer (if on): tag each detection with a stable player_id (1..4).
        # This consumes the tracker output; it never modifies the tracker.
        if identity is not None:
            identity.update(frame, tracked, n)

        # draw everything we kept (use the stable id for label + colour when ReID
        # is on, so a player keeps one colour/number across track fragments)
        for det in tracked:
            disp_id = det.get("player_id")
            if disp_id is None:
                disp_id = det["track_id"]
            _draw_overlay(frame, det, disp_id, color_for_id(disp_id), kp_thr)
        if polygon is not None:
            cv2.polylines(frame, [polygon], True, (0, 255, 255), 2)  # court outline
        if homog is not None:
            draw_court_lines(frame, homog)  # service/net/center/base/side lines

        # project players onto the court (meters) -> markers, minimap, log
        positions = []
        if homog is not None:
            for det in tracked:
                pos = player_court_position(det, homog, kp_thr, inside_margin)
                # colour/label the dot by the stable id when ReID is on
                disp_id = det.get("player_id")
                if disp_id is None:
                    disp_id = det["track_id"]
                pos["track_id"] = disp_id
                positions.append(pos)
                # show the chosen foot point + meters on the main frame so a human
                # can sanity-check the mapping (this IS the quality gate)
                fx, fy = int(pos["foot_px"][0]), int(pos["foot_px"][1])
                col = color_for_id(disp_id)
                cv2.circle(frame, (fx, fy), 5, col, -1)
                xm, ym = pos["foot_m"]
                cv2.putText(frame, f"({xm:.1f},{ym:.1f})m", (fx + 6, fy + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            if minimap is not None:
                mm = minimap.render(positions, color_fn=color_for_id)
                minimap.composite(frame, mm, corner="tr")
            if positions_writer is not None:
                positions_writer.write(
                    json.dumps({"frame": n, "players": positions}, cls=NumpyEncoder) + "\n")

        metrics.update(len(raw), len(kept), tracked)

        if writer is not None:
            writer.write(frame)
        if show:
            cv2.imshow("Padel Phase-1", frame)
            # pause only the LEFTOVER of the frame budget after the work above,
            # so playback actually hits the configured fps (see utils/display.py)
            work_ms = (time.time() - loop_t0) * 1000.0
            if throttle.wait(work_ms) == ord("q"):
                break

        n += 1
        if max_frames is not None and n >= max_frames:
            break

    cap.release()
    if writer is not None:
        writer.release()
    if positions_writer is not None:
        positions_writer.close()
    if identity is not None:
        identity.save()   # persist the 4 profiles + flush the assignment log
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
