"""core/ball_selector.py
FIXED-LAG BUFFERED BALL SELECTOR.

Pick the real ball from the detector's per-frame CANDIDATES using a SHORT LOOKAHEAD, so an
isolated false positive (a one-frame blip on the fence / a limb / a light) and a static
distractor (a held ball, a logo, a reflection) are rejected by SUPPORT -- "does this
detection continue across neighbouring frames along a plausible path?" -- the one thing a
causal tracker structurally cannot check (it has no future).

REAL-TIME: this is fixed-lag, NOT offline. It commits frame (N - lag) the moment frame N
arrives, i.e. a constant ~lag-frame latency (lag=5 @ 20fps = 0.25s). It never needs the whole
clip; it keeps only a 2*lag+1 frame rolling buffer. Identical behaviour live (Jetson) or on a
recorded clip.

OUTPUT: at most ONE clean detection per frame -- a BallPoint with source "detected" (a
supported candidate) or "none" (ghost / no ball). The existing Kalman tracker then SMOOTHS and
COASTS on top of this, so gap-filling stays where it already works; the selector's only job is
to hand the tracker a clean, ghost-free measurement (or nothing).

WHY THIS EXISTS: our causal tracker had to commit each frame instantly, so it followed isolated
FPs (the trail zigzags / teleport-to-fence) or, when we rejected far candidates blindly, dropped
the real post-hit ball (recall crash). A few frames of lookahead resolves both: a real ball has
SUPPORT in its neighbours; an FP does not.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class BallPoint:
    """One committed per-frame decision."""
    frame: int
    u: Optional[float]
    v: Optional[float]
    conf: float
    source: str            # "detected" (a supported candidate) | "none" (ghost / no ball)


class FixedLagBallSelector:
    """Support-based ball selection over a fixed-lag window. Feed one frame's candidates at a
    time with push(); it returns the committed decision for the frame `lag` frames back."""

    def __init__(self, lag: int = 5, max_step_px: float = 350.0, min_support: int = 2,
                 static_radius_px: float = 20.0) -> None:
        self.lag = max(0, int(lag))                 # lookahead in frames (== latency)
        self.max_step_px = float(max_step_px)       # per-frame motion tube radius
        self.min_support = int(min_support)         # neighbour frames that must agree
        self.static_radius_px = float(static_radius_px)  # below this spread -> static distractor
        self._cands: Dict[int, List[Any]] = {}      # frame -> candidates (objects with .u,.v,.confidence)
        self._last_pt: Optional[BallPoint] = None   # last DETECTED point, a tie-break anchor

    # ------------------------------------------------------------------ public
    def push(self, frame_idx: int, candidates: List[Any]) -> Optional[BallPoint]:
        """Feed frame `frame_idx`'s detector candidates. Returns the committed BallPoint for
        frame (frame_idx - lag), or None while the lookahead window is still filling."""
        self._cands[frame_idx] = list(candidates or [])
        # keep only the rolling window we still need: [frame_idx-2*lag .. frame_idx]
        cutoff = frame_idx - 2 * self.lag
        for f in [f for f in self._cands if f < cutoff]:
            del self._cands[f]
        t = frame_idx - self.lag
        return self._decide(t) if t >= 0 else None

    def flush(self) -> List[BallPoint]:
        """End-of-stream: decide the last `lag` frames that never got a full forward window
        (using whatever lookahead is available)."""
        if not self._cands:
            return []
        last = max(self._cands)
        return [self._decide(t) for t in range(last - self.lag + 1, last + 1) if t >= 0]

    # ---------------------------------------------------------------- internal
    def _decide(self, t: int) -> BallPoint:
        best = None
        best_support = -1
        best_spread = 0.0
        for c in self._cands.get(t, []):
            support, spread = self._support(c, t)
            better = (support > best_support
                      or (support == best_support and best is not None and self._prefer(c, best)))
            if better:
                best, best_support, best_spread = c, support, spread
        # accept only a SUPPORTED, NON-STATIC candidate; everything else is a ghost -> none
        if (best is not None and best_support >= self.min_support
                and best_spread > self.static_radius_px):
            pt = BallPoint(t, float(best.u), float(best.v),
                           float(getattr(best, "confidence", 1.0)), "detected")
            self._last_pt = pt
            return pt
        return BallPoint(t, None, None, 0.0, "none")

    def _support(self, c, t: int) -> Tuple[int, float]:
        """Count neighbouring frames that have a candidate consistent with `c` under bounded
        motion (within max_step_px per frame of separation), and the spatial SPREAD of those
        matches (used to reject a static path)."""
        xs, ys = [float(c.u)], [float(c.v)]
        support = 0
        for f in range(t - self.lag, t + self.lag + 1):
            if f == t or f not in self._cands:
                continue
            budget = self.max_step_px * abs(f - t)
            hit, hd = None, budget
            for o in self._cands[f]:
                d = math.hypot(o.u - c.u, o.v - c.v)
                if d <= budget and (hit is None or d < hd):
                    hit, hd = o, d
            if hit is not None:
                support += 1
                xs.append(float(hit.u))
                ys.append(float(hit.v))
        spread = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return support, spread

    def _prefer(self, c, best) -> bool:
        """Tie-break on equal support: higher confidence, else nearer the last detected point."""
        cc = float(getattr(c, "confidence", 1.0))
        bc = float(getattr(best, "confidence", 1.0))
        if cc != bc:
            return cc > bc
        if self._last_pt is not None and self._last_pt.u is not None:
            dc = math.hypot(c.u - self._last_pt.u, c.v - self._last_pt.v)
            db = math.hypot(best.u - self._last_pt.u, best.v - self._last_pt.v)
            return dc < db
        return False


# --------------------------------------------------------------------------- #
# Unit tests:  python -m core.ball_selector
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from collections import namedtuple
    C = namedtuple("C", "u v confidence")

    def run(frames, **kw):
        sel = FixedLagBallSelector(**kw)
        out: Dict[int, BallPoint] = {}
        for i, cands in enumerate(frames):
            pt = sel.push(i, cands)
            if pt is not None:
                out[pt.frame] = pt
        for pt in sel.flush():
            out[pt.frame] = pt
        return out

    KW = dict(lag=3, max_step_px=100.0, min_support=2, static_radius_px=15.0)

    # 1) smooth ball + a one-frame isolated blip at frame 5 -> ball kept, blip rejected
    frames = [[C(100 + 12 * i, 200, 0.8)] for i in range(10)]
    frames[5].append(C(2000, 50, 0.95))          # far, high-conf, ONE frame only
    o = run(frames, **KW)
    det = {f for f, p in o.items() if p.source == "detected"}
    assert det == set(range(10)), det
    assert abs(o[5].u - (100 + 12 * 5)) < 1, o[5]   # chose the REAL ball, not the blip
    print("test1 PASS: smooth ball kept, isolated blip rejected")

    # 2) static distractor (same spot 10 frames) -> all rejected
    o = run([[C(500, 500, 0.9)] for _ in range(10)], **KW)
    assert all(p.source == "none" for p in o.values()), o
    print("test2 PASS: static distractor rejected")

    # 3) post-hit reversal (moves right, then reverses left, continuous) -> kept throughout
    us = [100, 130, 160, 190, 220, 190, 160, 130, 100, 70]
    o = run([[C(u, 300, 0.8)] for u in us], **KW)
    assert all(p.source == "detected" for p in o.values()), o
    print("test3 PASS: post-hit reversal kept (no recall loss)")

    # 4) genuine miss (empty frame in the middle) -> that frame is 'none', neighbours detected
    frames = [[C(100 + 12 * i, 200, 0.8)] for i in range(10)]
    frames[5] = []                                # detector missed this frame
    o = run(frames, **KW)
    assert o[5].source == "none" and o[4].source == "detected" and o[6].source == "detected", o
    print("test4 PASS: true miss -> none (Kalman coasts downstream)")

    print("\nAll selector unit tests passed.")
