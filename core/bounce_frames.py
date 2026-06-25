"""core/bounce_frames.py
Extract a short MONTAGE STRIP around each proposed bounce so you can VISUALLY confirm
it -- a single still can't show a bounce, but a strip of frames before/at/after lets
you SEE the ball come DOWN and go back UP.

To keep the BALL sharp (it is tiny in a 4K frame), we CROP a fixed window around the
bounce position at NATIVE resolution -- NOT downscale the whole court. The same crop
box is used for every tile in a strip, so the ball visibly moves DOWN then UP inside
the window. Crop center comes from the candidate's (u, v); the box is clamped to the
frame edges.

Needs the VIDEO, so run it on the pod (a laptop can also seek to a few hundred frames
fine -- it is CONTINUOUS 4K decode that is slow, not seeking to ~30 spots).

USAGE (on the pod)
  python -m core.bounce_frames \
      --video side-1-full-vid.mp4 \
      --candidates output/bounce_candidates_side-1.csv \
      --labels output/ball_labels_side-1-full-vid.csv \
      --crop 900 --span 2 --out-dir output/bounce_strips_side-1

Bump --crop if the ball's arc gets cut off at the top/bottom of the window.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List, Optional, Tuple

import cv2


def _load_ball_positions(path: str) -> Dict[int, Tuple[float, float]]:
    """frame -> (u, v) for visible hand labels (used to mark the ball on each tile)."""
    out: Dict[int, Tuple[float, float]] = {}
    for r in csv.DictReader(open(path, newline="")):
        if str(r.get("visible", "")).strip() not in ("1", "1.0", "true", "True"):
            continue
        u, v = r.get("u", "").strip(), r.get("v", "").strip()
        if u and v:
            out[int(float(r["frame"]))] = (float(u), float(v))
    return out


def _load_candidates(path: str) -> List[Tuple[int, Optional[float], Optional[float]]]:
    """(frame, u, v) per candidate; u,v center the crop (fall back to frame centre)."""
    out = []
    for r in csv.DictReader(open(path, newline="")):
        if not r.get("frame"):
            continue
        u, v = r.get("u", "").strip(), r.get("v", "").strip()
        out.append((int(float(r["frame"])),
                    float(u) if u else None,
                    float(v) if v else None))
    return sorted(out, key=lambda t: t[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--candidates", required=True, help="csv with frame[,u,v]")
    ap.add_argument("--labels", help="ball_labels csv -> mark the ball on each tile")
    ap.add_argument("--stride", type=int, default=2, help="label cadence (frames)")
    ap.add_argument("--span", type=int, default=2,
                    help="tiles each side of the bounce (total = 2*span+1)")
    ap.add_argument("--crop", type=int, default=900,
                    help="native-resolution crop window (px) centered on the ball")
    ap.add_argument("--out-dir", default="output/bounce_strips")
    args = ap.parse_args()

    positions = _load_ball_positions(args.labels) if args.labels else {}
    cands = _load_candidates(args.candidates)
    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half = args.crop // 2

    made = 0
    for cf, cu, cv_ in cands:
        # crop box centered on the bounce position (clamped to the frame)
        ccx = int(cu) if cu is not None else W // 2
        ccy = int(cv_) if cv_ is not None else H // 2
        x0 = max(0, min(ccx - half, W - args.crop))
        y0 = max(0, min(ccy - half, H - args.crop))
        x1, y1 = x0 + args.crop, y0 + args.crop

        tiles = []
        for fr in [cf + k * args.stride for k in range(-args.span, args.span + 1)]:
            if fr < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                continue
            crop = img[y0:y1, x0:x1].copy()           # native-resolution window
            # mark the ball (green ring) at its position WITHIN the crop
            if fr in positions:
                u, v = positions[fr]
                if x0 <= u < x1 and y0 <= v < y1:
                    cv2.circle(crop, (int(u - x0), int(v - y0)), 16, (0, 255, 0), 3)
            if fr == cf:                                # the proposed bounce tile
                cv2.rectangle(crop, (0, 0), (crop.shape[1] - 1, crop.shape[0] - 1),
                              (0, 255, 255), 10)
            cv2.putText(crop, f"f{fr}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 255), 2)
            tiles.append(crop)
        if not tiles:
            print(f"  frame {cf}: no readable frames")
            continue
        strip = cv2.hconcat(tiles)
        path = os.path.join(args.out_dir, f"bounce_{cf:06d}.jpg")
        cv2.imwrite(path, strip, [cv2.IMWRITE_JPEG_QUALITY, 92])
        made += 1

    cap.release()
    print(f"saved {made} montage strips -> {args.out_dir}")
    print("  yellow border = the proposed bounce frame; green ring = labelled ball.")
    print("  a REAL floor bounce: the ball descends, touches low, then rises.")
    print(f"  (crop window = {args.crop}px native; bump --crop if the arc is cut off.)")


if __name__ == "__main__":
    main()
