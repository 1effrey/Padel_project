"""core/sync_manual.py
MANUAL sync tuner -- align the two cameras by eye.

Convention: side-B frame = side-A frame + offset_frames. The automatic --sync finds a
frame-exact INTEGER offset, but at 20 fps a sub-frame timing difference (~40 ms = 0.8 of a
frame) cannot be represented by an integer and shows as a slight lag in the side-by-side
view. This tool lets you scrub to a clear common event (a ball bounce, a sharp swing) and
nudge the offset until the two sides fire together, then save it to config-A. Choosing the
better integer (e.g. 55 instead of 54) can roughly halve the residual; killing sub-frame
error entirely would need frame interpolation (artefacts, not worth it).

Controls (the window must be focused):
    d / a : step BOTH forward / back 1 frame
    c / z : jump BOTH +10 / -10 frames
    l / j : nudge the OFFSET +1 / -1  (shifts side-2 relative to side-1)  <- the knob
    s     : SAVE the offset into config-A's sync block
    q/ESC : quit
"""
from __future__ import annotations

import json
from typing import Any, Dict

import cv2
import numpy as np

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _save_offset(config_path: str, offset: int, fps: float) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("sync", {})
    cfg["sync"]["offset_frames"] = int(offset)
    cfg["sync"]["method"] = "manual"
    cfg["sync"]["note"] = (f"manually aligned by eye: side-B frame = side-A + {int(offset)} "
                           f"({offset / fps:.3f}s @ {fps:.2f}fps).")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def run_manual_sync(cfg_a: Dict[str, Any], cfg_b: Dict[str, Any], config_a_path: str,
                    panel_h: int = 480) -> None:
    """Interactive side-by-side sync tuner. Saves the chosen offset to config-A."""
    capA = cv2.VideoCapture(cfg_a["source"])
    capB = cv2.VideoCapture(cfg_b["source"])
    if not capA.isOpened() or not capB.isOpened():
        print("[sync-manual] could not open one of the videos.")
        return
    nA = int(capA.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
    nB = int(capB.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
    fps = capA.get(cv2.CAP_PROP_FPS) or 20.0
    offset = int(cfg_a.get("sync", {}).get("offset_frames", 0))
    fA = 0

    def grab(cap, idx, nmax):
        idx = max(0, min(idx, nmax - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        return fr if ok else None

    def fit(img):
        if img is None:
            return np.zeros((panel_h, int(panel_h * 16 / 9), 3), np.uint8)
        return cv2.resize(img, (int(img.shape[1] * panel_h / img.shape[0]), panel_h))

    win = "Manual Sync"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1500, 560)
    print("[sync-manual] scrub to a clear common event (a bounce / sharp swing visible in "
          "BOTH), nudge the offset with l/j until the two sides fire together, then press s "
          "to save.  (d/a step, c/z jump 10, q quit)")

    while True:
        fA = max(0, min(fA, nA - 1))
        pA, pB = fit(grab(capA, fA, nA)), fit(grab(capB, fA + offset, nB))
        comp = cv2.hconcat([pA, pB])
        ms = offset / fps * 1000.0
        cv2.putText(comp, f"side-1  f={fA}", (12, 32), _FONT, 0.8, (255, 255, 255), 2)
        cv2.putText(comp, f"side-2  f={fA + offset}", (pA.shape[1] + 12, 32), _FONT, 0.8,
                    (255, 255, 255), 2)
        cv2.putText(comp, f"offset = {offset}  ({ms:+.0f} ms)     "
                          f"l/j nudge offset   d/a step   c/z jump10   s SAVE   q quit",
                    (12, comp.shape[0] - 14), _FONT, 0.7, (0, 255, 255), 2)
        cv2.imshow(win, comp)
        k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        elif k == ord("d"):
            fA += 1
        elif k == ord("a"):
            fA -= 1
        elif k == ord("c"):
            fA += 10
        elif k == ord("z"):
            fA -= 10
        elif k == ord("l"):
            offset += 1
        elif k == ord("j"):
            offset -= 1
        elif k == ord("s"):
            _save_offset(config_a_path, offset, fps)
            print(f"[sync-manual] SAVED offset={offset} ({offset / fps:.3f}s) -> "
                  f"{config_a_path}")
    capA.release()
    capB.release()
    cv2.destroyAllWindows()
    print(f"[sync-manual] final offset = {offset} "
          f"({offset / fps:.3f}s @ {fps:.2f}fps)")


if __name__ == "__main__":
    import sys

    cpa = sys.argv[1] if len(sys.argv) > 1 else "config-side1.json"
    cpb = sys.argv[2] if len(sys.argv) > 2 else "config-side2.json"
    with open(cpa, "r", encoding="utf-8") as f:
        ca = json.load(f)
    with open(cpb, "r", encoding="utf-8") as f:
        cb = json.load(f)
    run_manual_sync(ca, cb, cpa)
