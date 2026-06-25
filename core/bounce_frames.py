"""core/bounce_frames.py
Extract a short MONTAGE STRIP around each proposed bounce so you can VISUALLY confirm
it -- a single still can't show a bounce, but a strip of frames before/at/after lets
you SEE the ball come DOWN and go back UP.

Needs the VIDEO, so run it on the pod (a laptop can also seek to a few hundred frames
fine -- it is CONTINUOUS 4K decode that is slow, not seeking to ~30 spots).

For each candidate frame f it grabs frames  f-2*stride .. f+2*stride  (the label
cadence), marks the hand-labelled ball position on each, and h-stacks them into one
downscaled JPG named by the candidate frame. ~30 small images you can flip through.

USAGE (on the pod)
  python -m core.bounce_frames \
      --video side-1-full-vid.mp4 \
      --candidates results/bounce_candidates_side-1.csv \
      --labels results/ball_labels_side-1-full-vid.csv \
      --stride 2 --span 2 --out-dir output/bounce_strips_side-1

Then transfer the small out-dir back (runpodctl / rclone) and review.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, Optional, Tuple

import cv2


def _load_ball_positions(path: str) -> Dict[int, Tuple[float, float]]:
    """frame -> (u, v) for visible hand labels."""
    out: Dict[int, Tuple[float, float]] = {}
    for r in csv.DictReader(open(path, newline="")):
        if str(r.get("visible", "")).strip() not in ("1", "1.0", "true", "True"):
            continue
        u, v = r.get("u", "").strip(), r.get("v", "").strip()
        if u and v:
            out[int(float(r["frame"]))] = (float(u), float(v))
    return out


def _load_frames(path: str) -> list:
    return sorted({int(float(r["frame"]))
                   for r in csv.DictReader(open(path, newline="")) if r.get("frame")})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--candidates", required=True, help="csv with a `frame` column")
    ap.add_argument("--labels", help="ball_labels csv -> mark the ball on each tile")
    ap.add_argument("--stride", type=int, default=2, help="label cadence (frames)")
    ap.add_argument("--span", type=int, default=2,
                    help="tiles each side of the bounce (total = 2*span+1)")
    ap.add_argument("--tile-w", type=int, default=360, help="tile width (px)")
    ap.add_argument("--out-dir", default="output/bounce_strips")
    args = ap.parse_args()

    positions = _load_ball_positions(args.labels) if args.labels else {}
    cands = _load_frames(args.candidates)
    os.makedirs(args.out_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")

    made = 0
    for cf in cands:
        wanted = [cf + k * args.stride for k in range(-args.span, args.span + 1)]
        tiles = []
        for fr in wanted:
            if fr < 0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                continue
            # mark the ball (green ring) if we have a label for this frame
            if fr in positions:
                u, v = positions[fr]
                cv2.circle(img, (int(u), int(v)), 18, (0, 255, 0), 3)
            # the bounce frame itself gets a yellow border so it stands out
            if fr == cf:
                cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1),
                              (0, 255, 255), 14)
            h, w = img.shape[:2]
            scale = args.tile_w / w
            tile = cv2.resize(img, (args.tile_w, int(h * scale)))
            cv2.putText(tile, f"f{fr}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            tiles.append(tile)
        if not tiles:
            print(f"  frame {cf}: no readable frames")
            continue
        strip = cv2.hconcat(tiles)
        path = os.path.join(args.out_dir, f"bounce_{cf:06d}.jpg")
        cv2.imwrite(path, strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
        made += 1

    cap.release()
    print(f"saved {made} montage strips -> {args.out_dir}")
    print("  yellow border = the proposed bounce frame; green ring = labelled ball.")
    print("  a REAL floor bounce: the ball descends, touches low, then rises.")


if __name__ == "__main__":
    main()
