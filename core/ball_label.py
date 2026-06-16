"""core/ball_label.py
Phase-1 BALL labeling tool -- click the ball, frame by frame, to build the training
data TrackNetV2 needs. This is the human-in-the-loop step docs/ball_labeling.md
describes; the same click-on-a-frame idea as main.py's calibrate_roi.

OUTPUT (one CSV per clip, in config["output"]["dir"])
    output/ball_labels_<clip>.csv   with columns:  frame,visible,u,v
      visible=1 -> ball is on screen at full-res pixel (u, v)
      visible=0 -> ball not visible this frame (occluded / off-frame); u,v blank
  Only frames you actually label are written. Re-running RESUMES from the existing
  CSV so you can label in several sittings.

CONTROLS
    Left-click ... set the ball here  -> saves (visible=1) and auto-advances
    B ............ ball NOT visible    -> saves (visible=0) and auto-advances
    D / Space .... skip forward (no label saved for this frame)
    A ............ go back one step (to fix a mistake)
    X ............ clear this frame's label
    S ............ save the CSV now
    Q / Esc ...... save and quit

TIPS (what to label for a good detector)
    Label the HARD frames on purpose: serves and smashes (fast, blurred), the ball
    near vs far, and occlusions (mark those B = not visible). Aim for ~800-1500
    frames to start, balanced between "ball visible" and "not visible".
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import cv2


def _csv_path(out_dir: str, source: str) -> str:
    base = os.path.splitext(os.path.basename(str(source)))[0]
    return os.path.join(out_dir, f"ball_labels_{base}.csv")


def _load_existing(path: str) -> Dict[int, Tuple[int, Optional[float], Optional[float]]]:
    """Resume support: read any labels already saved for this clip."""
    labels: Dict[int, Tuple[int, Optional[float], Optional[float]]] = {}
    if not os.path.isfile(path):
        return labels
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            frame = int(row["frame"])
            visible = int(row["visible"])
            if visible and row.get("u") not in (None, ""):
                labels[frame] = (1, float(row["u"]), float(row["v"]))
            else:
                if visible:
                    print(f"[label-ball] WARNING: frame {frame} marked visible but has "
                          f"no (u,v) -> treating as NOT visible.")
                labels[frame] = (0, None, None)
    return labels


def _save(path: str, labels: Dict[int, Tuple[int, Optional[float], Optional[float]]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "visible", "u", "v"])
        for frame in sorted(labels):
            visible, u, v = labels[frame]
            if visible:
                w.writerow([frame, 1, f"{u:.1f}", f"{v:.1f}"])
            else:
                w.writerow([frame, 0, "", ""])


def read_frame_list(path: str) -> List[int]:
    """Read the 'frame' column from a CSV (e.g. output/hard_frames_*.csv from the
    miner) -> a sorted list of unique frame indices to label."""
    frames = set()
    with open(path, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                frames.add(int(row["frame"]))
            except (KeyError, ValueError):
                continue
    return sorted(frames)


def run_label_ball(config: Dict[str, Any], stride: Optional[int] = None,
                   start_frame: int = 0, frames: Optional[List[int]] = None) -> None:
    """Open config["source"] and label the ball -> CSV.

    `frames`: if given (e.g. from --label-from / the hard-example miner), step through
    ONLY those frames instead of striding the whole clip."""
    source = config["source"]
    out_dir = config.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)
    ball_cfg = config.get("ball", {})
    step = int(stride if stride is not None else ball_cfg.get("label_stride", 1))
    step = max(1, step)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    full_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    csv_path = _csv_path(out_dir, source)
    labels = _load_existing(csv_path)
    print(f"[label-ball] {source}: {total} frames, step={step}. "
          f"{len(labels)} labels already in {csv_path} (resuming).")

    # 4K -> downscale for display; remember scale to map clicks back to full res.
    scale = min(1.0, 1280 / full_w) if full_w else 1.0
    pending_click: Dict[str, Optional[Tuple[int, int]]] = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click["pt"] = (x, y)

    win = "label-ball"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # navigation: a frame LIST (from --label-from) overrides stride stepping
    nav: Optional[List[int]] = None
    if frames:
        nav = sorted(set(int(f) for f in frames))
        if total:
            nav = [f for f in nav if 0 <= f < total]
        if not nav:
            print("[label-ball] frame list empty / out of range; nothing to do.")
            cap.release()
            return
        print(f"[label-ball] frame-list mode: {len(nav)} target frames.")

    state = {
        "idx": nav[0] if nav else (max(0, min(start_frame, total - 1)) if total
                                   else max(0, start_frame)),
        "pos": 0,
    }

    def go(delta: int) -> None:
        """Advance/retreat: through the frame list if given, else by `step`."""
        if nav is not None:
            state["pos"] = max(0, min(len(nav) - 1, state["pos"] + delta))
            state["idx"] = nav[state["pos"]]
        else:
            ni = state["idx"] + delta * step
            state["idx"] = max(0, min(total - 1, ni)) if total else max(0, ni)

    last_idx = -1
    frame = None
    while True:
        idx = state["idx"]
        # (re)read the current frame only when the index changed (seeking 4K is slow)
        if idx != last_idx:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                print(f"[label-ball] could not read frame {idx}; stopping.")
                break
            last_idx = idx

        disp = cv2.resize(frame, (0, 0), fx=scale, fy=scale) if scale != 1.0 else frame.copy()

        # consume a click -> label this frame visible, then auto-advance
        if pending_click["pt"] is not None:
            cx, cy = pending_click["pt"]
            labels[idx] = (1, cx / scale, cy / scale)
            pending_click["pt"] = None
            go(1)
            continue

        # draw the existing label for this frame (if any)
        lab = labels.get(idx)
        if lab is not None and lab[0] == 1:
            dx, dy = int(lab[1] * scale), int(lab[2] * scale)
            cv2.circle(disp, (dx, dy), 9, (0, 255, 255), 2)
            cv2.circle(disp, (dx, dy), 2, (0, 255, 255), -1)
            tag = "VISIBLE"
        elif lab is not None:
            tag = "NOT VISIBLE"
        else:
            tag = "unlabeled"

        prog = (f"{state['pos'] + 1}/{len(nav)}  (frame {idx})" if nav
                else f"frame {idx}/{max(0, total - 1)}")
        cv2.putText(disp, f"{prog}   [{tag}]   labeled={len(labels)}",
                    (10, 28), font, 0.7, (255, 255, 255), 2)
        cv2.putText(disp, "L-click:ball  B:not-visible  D/Space:skip  A:back  X:clear  S:save  Q:quit",
                    (10, disp.shape[0] - 14), font, 0.55, (200, 200, 200), 2)
        cv2.imshow(win, disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("b"):
            labels[idx] = (0, None, None)
            go(1)
        elif key in (ord("d"), ord(" ")):
            go(1)
        elif key == ord("a"):
            go(-1)
        elif key == ord("x"):
            labels.pop(idx, None)
        elif key == ord("s"):
            _save(csv_path, labels)
            print(f"[label-ball] saved {len(labels)} labels -> {csv_path}")
        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
    _save(csv_path, labels)
    n_vis = sum(1 for v in labels.values() if v[0] == 1)
    print(f"[label-ball] done. {len(labels)} labels ({n_vis} visible, "
          f"{len(labels) - n_vis} not-visible) -> {csv_path}")
