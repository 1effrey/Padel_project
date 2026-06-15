"""utils/sync.py
Estimate the TIME OFFSET (in frames) between two camera recordings of the same
match, so the (Phase-4) fusion step can combine the SAME instant from both views.

WHY THIS EXISTS
  The two clips were started at slightly different moments -- their frame counts
  differ -- so frame N in one video is NOT the same instant as frame N in the
  other. Before we can fuse the two views into a single top-down court, we must
  know that offset.

THE IDEA (cheap, no model, no labels)
  Both cameras film the SAME rallies. When players sprint the whole picture
  changes a lot; when a point ends and they reset, it changes little. That
  "motion energy" over time is a 1-D signal that rises and falls TOGETHER in both
  videos. We build it for each clip (mean absolute frame-to-frame difference on a
  small grayscale copy) and cross-correlate the two signals. The lag that lines
  them up best is the time offset.

OUTPUT
  A single integer `offset_frames` with the convention:

      real instant at  side-A frame f   <-->   side-B frame  f + offset_frames

  plus the correlation score (how confident the match is). The caller saves this
  into config and ALSO writes a side-by-side montage at the detected offset so a
  human can eyeball that the same moment really shows in both views -- that visual
  check IS the quality gate, exactly like the homography preview.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


def motion_energy(
    source: str,
    stride: int = 10,
    width: int = 160,
    height: int = 90,
    max_frames: Optional[int] = None,
    log_every: int = 2000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the per-clip motion-energy signal.

    We read frames sequentially (mp4 random-seek is unreliable), and on every
    `stride`-th frame we shrink it to a tiny grayscale image and record how much
    it changed from the PREVIOUS sampled frame. That change (mean absolute pixel
    difference) is the activity at that moment.

    Returns (sample_frame_indices, energy) as two equal-length arrays. The first
    sample has energy 0 (nothing to diff against yet).
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    idxs = []
    energy = []
    prev_small: Optional[np.ndarray] = None
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if n % stride == 0:
            small = cv2.cvtColor(cv2.resize(frame, (width, height)), cv2.COLOR_BGR2GRAY)
            if prev_small is None:
                e = 0.0
            else:
                e = float(np.mean(np.abs(small.astype(np.int16) - prev_small.astype(np.int16))))
            idxs.append(n)
            energy.append(e)
            prev_small = small
        n += 1
        if log_every and n % log_every == 0:
            print(f"[sync] {source}: scanned {n} frames, {len(idxs)} samples")
        if max_frames is not None and n >= max_frames:
            break
    cap.release()
    return np.asarray(idxs, dtype=np.int64), np.asarray(energy, dtype=np.float64)


def _zscore(x: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance so the correlation ignores absolute brightness /
    exposure differences between the two cameras."""
    sd = x.std()
    return (x - x.mean()) / sd if sd > 1e-9 else x - x.mean()


def estimate_offset(
    energy_a: np.ndarray,
    energy_b: np.ndarray,
    stride: int,
    max_lag_frames: int = 600,
) -> Dict[str, Any]:
    """Cross-correlate the two motion signals and return the best frame offset.

    Both signals are sampled on the SAME uniform `stride`, so a lag of `k` samples
    means `k * stride` frames. We slide B against A over +/- max_lag and pick the
    lag with the highest Pearson correlation on the overlapping region.

    Offset convention (see module docstring):
        side-A frame f   <-->   side-B frame  f + offset_frames
    A POSITIVE offset means B is delayed relative to A (B started earlier / shows
    a given instant at a higher frame number).

    Returns a dict: offset_frames, corr (best), corr_runner_up, margin, lag_samples.
    """
    a = _zscore(energy_a)
    b = _zscore(energy_b)
    max_lag = max(1, int(max_lag_frames // stride))

    scores: Dict[int, float] = {}
    for lag in range(-max_lag, max_lag + 1):
        # convention: sample a[i] pairs with b[i + lag] (frame_b = frame_a + offset)
        i0 = max(0, -lag)
        i1 = min(len(a), len(b) - lag)
        if i1 - i0 < 30:                # too little overlap to trust
            continue
        av = a[i0:i1]
        bv = b[i0 + lag:i1 + lag]
        denom = np.linalg.norm(av) * np.linalg.norm(bv)
        scores[lag] = float(np.dot(av, bv) / denom) if denom > 1e-9 else 0.0

    if not scores:
        raise RuntimeError("[sync] no usable overlap to correlate.")

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_lag, best_corr = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0.0
    return {
        "offset_frames": int(best_lag * stride),
        "lag_samples": int(best_lag),
        "corr": round(best_corr, 4),
        "corr_runner_up": round(runner, 4),
        "margin": round(best_corr - runner, 4),
        "stride": stride,
        "max_lag_frames": max_lag_frames,
    }


def save_alignment_montage(
    source_a: str,
    source_b: str,
    offset_frames: int,
    out_path: str,
    sample_fracs: Tuple[float, ...] = (0.25, 0.5, 0.75),
    panel_w: int = 640,
) -> bool:
    """Write a stacked montage of paired frames at the detected offset so a human
    can confirm the alignment (the quality gate).

    For each fraction p of the overlapping timeline we grab side-A frame fa and
    side-B frame fa + offset, shrink both to panel_w wide, place them side by side
    (A | B), and stack the rows. Returns False if frames could not be read.
    """
    cap_a = cv2.VideoCapture(source_a)
    cap_b = cv2.VideoCapture(source_b)
    na = int(cap_a.get(cv2.CAP_PROP_FRAME_COUNT))
    nb = int(cap_b.get(cv2.CAP_PROP_FRAME_COUNT))

    def grab(cap, idx):
        if idx < 0:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        return fr if ok else None

    def panelize(fr, label):
        h, w = fr.shape[:2]
        ph = int(panel_w * h / w)
        p = cv2.resize(fr, (panel_w, ph))
        cv2.putText(p, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return p

    rows = []
    for p in sample_fracs:
        # choose fa so that both fa and fa+offset are in range
        lo = max(0, -offset_frames)
        hi = min(na, nb - offset_frames)
        if hi <= lo:
            continue
        fa = int(lo + p * (hi - lo))
        fb = fa + offset_frames
        ia, ib = grab(cap_a, fa), grab(cap_b, fb)
        if ia is None or ib is None:
            continue
        pa = panelize(ia, f"A f{fa}")
        pb = panelize(ib, f"B f{fb}")
        h = min(pa.shape[0], pb.shape[0])
        rows.append(np.hstack([pa[:h], pb[:h]]))
    cap_a.release()
    cap_b.release()

    if not rows:
        return False
    w = min(r.shape[1] for r in rows)
    montage = np.vstack([r[:, :w] for r in rows])
    cv2.imwrite(out_path, montage)
    return True
