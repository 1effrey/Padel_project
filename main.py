"""main.py -- Phase-1 entry point.

Two modes:
  1) Normal run:        python main.py --source side-1-full-vid.mp4 --show
  2) Define the court:  python main.py --calibrate-roi

Everything tunable comes from config.json; CLI flags only OVERRIDE a few of
those values for convenience. We never hard-code thresholds in logic.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import cv2

from core.pipeline import run


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_config(path: str, config: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def calibrate_roi(config: Dict[str, Any], config_path: str) -> None:
    """Click the court polygon on the first frame and save it to config.json.

    Controls:  Left-click = add point,  Right-click = undo,
               S = save & quit,         Q = quit without saving.

    We keep this human-in-the-loop on purpose: the court boundary is a judgment
    call best made by a person looking at the real court, not guessed by code.
    """
    cap = cv2.VideoCapture(config["source"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {config['source']}")

    # The frame is 4K. Downscale it to fit on screen for clicking, but remember
    # the scale so we can map every click back to FULL-resolution pixels (the
    # pipeline runs at full res, so the saved polygon must be in full-res coords).
    full_h, full_w = frame.shape[:2]
    max_w = 1280
    scale = min(1.0, max_w / full_w)
    disp_w, disp_h = int(full_w * scale), int(full_h * scale)
    base = cv2.resize(frame, (disp_w, disp_h))

    points: list[list[int]] = []  # stored in DISPLAY coords while clicking

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow("calibrate-roi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calibrate-roi", disp_w, disp_h)
    cv2.setMouseCallback("calibrate-roi", on_mouse)

    while True:
        disp = base.copy()
        for i, p in enumerate(points):
            cv2.circle(disp, (p[0], p[1]), 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(disp, tuple(points[i - 1]), tuple(p), (0, 255, 255), 2)
        if len(points) >= 3:  # show the closing edge as a hint
            cv2.line(disp, tuple(points[-1]), tuple(points[0]), (0, 255, 255), 1)
        cv2.putText(disp, "L:add  R:undo  S:save  Q:quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("calibrate-roi", disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("s") and len(points) >= 3:
            # scale DISPLAY clicks back up to full-resolution pixels before saving
            full_pts = [[int(round(px / scale)), int(round(py / scale))]
                        for px, py in points]
            config.setdefault("court", {})["polygon"] = full_pts
            save_config(config_path, config)
            print(f"[calibrate] saved {len(full_pts)} full-res points to {config_path}")
            break
        if key == ord("q"):
            print("[calibrate] cancelled, nothing saved.")
            break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Padel CV -- Phase 1")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--source", help="override config source (video path / index)")
    parser.add_argument("--model", help="override config model path")
    parser.add_argument("--show", action="store_true", help="display the annotated window")
    parser.add_argument("--save-video", action="store_true", help="write output/phase1_annotated.mp4")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames (for quick measuring)")
    parser.add_argument("--calibrate-roi", action="store_true", help="define the court polygon and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source:
        config["source"] = args.source
    if args.model:
        config["model"] = args.model

    if args.calibrate_roi:
        calibrate_roi(config, args.config)
        return

    run(config, show=args.show, save_video=args.save_video, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
