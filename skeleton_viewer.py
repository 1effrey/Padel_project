"""skeleton_viewer.py  --  STANDALONE skeleton playback from a per-player CSV.

This file is independent of the padel project: it imports nothing from core/ or
utils/, only `numpy` + `opencv-python`. Copy it anywhere.

It reads ONE per-player CSV (the playerN_*.csv export: player_id, frame, bbox*,
foot*, kp0_x..kp16_c) and animates the COCO-17 skeleton frame-by-frame on a blank
canvas, drawn at the players' real image pixel positions -- so you can watch the
tracked movement with your own eyes and judge whether it looks natural, using ONLY
the spreadsheet (no video needed).

Usage:
    python skeleton_viewer.py output/player3_fused_20260615_142500.csv
    python skeleton_viewer.py <csv> --fps 20 --conf 0.3 --width 1000
    python skeleton_viewer.py <csv> --check        # print stats and exit (no window)

Controls (in the window):
    SPACE  pause / resume
    .      step forward one frame (when paused)
    ,      step back one frame   (when paused)
    Q/Esc  quit
"""
from __future__ import annotations

import argparse
import csv
from typing import Dict, List

import cv2
import numpy as np

N_KP = 17

# COCO-17 skeleton bones (pairs of keypoint indices), grouped for colouring.
BONES = [
    # head (yellow)
    ((0, 1), (0, 255, 255)), ((0, 2), (0, 255, 255)),
    ((1, 3), (0, 255, 255)), ((2, 4), (0, 255, 255)),
    # arms (green)
    ((5, 7), (0, 255, 0)), ((7, 9), (0, 255, 0)),
    ((6, 8), (0, 255, 0)), ((8, 10), (0, 255, 0)),
    # torso (cyan)
    ((5, 6), (255, 255, 0)), ((5, 11), (255, 255, 0)),
    ((6, 12), (255, 255, 0)), ((11, 12), (255, 255, 0)),
    # legs (magenta)
    ((11, 13), (255, 0, 255)), ((13, 15), (255, 0, 255)),
    ((12, 14), (255, 0, 255)), ((14, 16), (255, 0, 255)),
]


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "kp0_x" not in rows[0]:
        raise SystemExit(
            f"'{path}' doesn't look like a per-player keypoint CSV "
            f"(no kp0_x column). Use a playerN_*.csv file.")
    return rows


def _f(val: str) -> float:
    """Parse a CSV cell to float; blanks -> 0.0."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return 0.0


def keypoints_of(row: Dict[str, str]) -> np.ndarray:
    """Pull the 17x3 (x, y, conf) keypoint array out of one CSV row."""
    kp = np.zeros((N_KP, 3), dtype=float)
    for i in range(N_KP):
        kp[i, 0] = _f(row.get(f"kp{i}_x"))
        kp[i, 1] = _f(row.get(f"kp{i}_y"))
        kp[i, 2] = _f(row.get(f"kp{i}_c"))
    return kp


def pick_source(rows: List[Dict[str, str]], chosen: str | None) -> List[Dict[str, str]]:
    """A fusion file can hold two rows per frame (one per camera). Keep a single
    source_video so the skeleton doesn't jump between camera views. Default = the
    source with the most rows."""
    srcs = [r.get("source_video", "") for r in rows]
    uniq = sorted(set(srcs))
    if len(uniq) <= 1:
        return rows
    if chosen is None:
        chosen = max(uniq, key=srcs.count)
    print(f"[viewer] file has {len(uniq)} source videos {uniq}; showing '{chosen}' "
          f"(override with --source). ")
    return [r for r in rows if r.get("source_video") == chosen]


def compute_bounds(rows, conf):
    """Image-pixel extent of all confident keypoints + bboxes, for scaling."""
    xs, ys = [], []
    for r in rows:
        kp = keypoints_of(r)
        for i in range(N_KP):
            if kp[i, 2] >= conf:
                xs.append(kp[i, 0]); ys.append(kp[i, 1])
        for kx, ky in (("bbox_x1", "bbox_y1"), ("bbox_x2", "bbox_y2")):
            xs.append(_f(r.get(kx))); ys.append(_f(r.get(ky)))
    if not xs:
        raise SystemExit("No confident keypoints found -- try a lower --conf.")
    return min(xs), max(xs), min(ys), max(ys)


def main() -> None:
    ap = argparse.ArgumentParser(description="Play back a player's skeleton from a CSV.")
    ap.add_argument("csv", help="path to a playerN_*.csv file")
    ap.add_argument("--fps", type=float, default=20.0, help="playback speed")
    ap.add_argument("--conf", type=float, default=0.3,
                    help="min keypoint confidence to draw a joint/bone")
    ap.add_argument("--width", type=int, default=1000, help="window width in pixels")
    ap.add_argument("--source", default=None,
                    help="for fusion files: which source_video to show")
    ap.add_argument("--check", action="store_true",
                    help="print stats and exit (no window)")
    args = ap.parse_args()

    rows = pick_source(load_rows(args.csv), args.source)
    x0, x1, y0, y1 = compute_bounds(rows, args.conf)
    pad = 40.0
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    scale = args.width / max(1.0, (x1 - x0))
    W = args.width
    H = max(1, int((y1 - y0) * scale))

    pid = rows[0].get("player_id", "?")
    print(f"[viewer] player {pid}: {len(rows)} frames, image extent "
          f"x[{x0:.0f},{x1:.0f}] y[{y0:.0f},{y1:.0f}] -> canvas {W}x{H}")
    if args.check:
        return

    def to_canvas(x: float, y: float):
        return int((x - x0) * scale), int((y - y0) * scale)

    win = f"skeleton  player {pid}  (SPACE pause | . , step | Q quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, W, H)
    delay = max(1, int(1000.0 / args.fps))

    idx, paused = 0, False
    while True:
        row = rows[idx]
        kp = keypoints_of(row)
        canvas = np.zeros((H, W, 3), dtype=np.uint8)

        # bbox (grey)
        bx1, by1 = to_canvas(_f(row["bbox_x1"]), _f(row["bbox_y1"]))
        bx2, by2 = to_canvas(_f(row["bbox_x2"]), _f(row["bbox_y2"]))
        cv2.rectangle(canvas, (bx1, by1), (bx2, by2), (90, 90, 90), 1)

        # bones
        for (a, b), col in BONES:
            if kp[a, 2] >= args.conf and kp[b, 2] >= args.conf:
                pa = to_canvas(kp[a, 0], kp[a, 1])
                pb = to_canvas(kp[b, 0], kp[b, 1])
                cv2.line(canvas, pa, pb, col, 2, cv2.LINE_AA)
        # joints
        for i in range(N_KP):
            if kp[i, 2] >= args.conf:
                cv2.circle(canvas, to_canvas(kp[i, 0], kp[i, 1]), 3, (0, 0, 255), -1)
        # foot point (the projected position), if present
        if "foot_img_x" in row:
            fx, fy = to_canvas(_f(row["foot_img_x"]), _f(row["foot_img_y"]))
            cv2.drawMarker(canvas, (fx, fy), (255, 255, 255), cv2.MARKER_TILTED_CROSS, 12, 1)

        hud = f"P{pid}  row {idx + 1}/{len(rows)}  frame {row.get('frame', '?')}"
        if paused:
            hud += "  [PAUSED]"
        cv2.putText(canvas, hud, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(0 if paused else delay) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("."):
            idx = min(idx + 1, len(rows) - 1); paused = True
        elif key == ord(","):
            idx = max(idx - 1, 0); paused = True
        elif not paused:
            idx = (idx + 1) % len(rows)   # loop so you can re-watch

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
