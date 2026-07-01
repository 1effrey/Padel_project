"""core/ball_trail.py
PROJECTILE TRAIL SMOOTHING -- turn the selector's clean per-frame points into smooth,
physics-shaped arcs for rendering (and for any downstream geometry that wants a clean path).

Why this exists: the selector/tracker gives us an ACCURATE per-frame point, but rendering it
as a raw point-to-point polyline looks jagged (sparse/blinking detections -> long straight
segments and kinks) and BREAKS on every 1-3 frame detector miss ("track not drawn"). Fadi's
reference output avoids both by (a) inpainting short gaps and (b) drawing a dense, smooth arc.
A free-flying ball follows a parabola (gravity), so a LOCAL QUADRATIC fit is the physically
correct smoother -- it removes jitter without inventing motion the ball can't make.

This is a post-process on the clean track. It is real-time-safe: every operation uses only a
small centered window (<= the fixed-lag we already buffer), never the whole clip.

Pipeline:
    detector -> FixedLagBallSelector -> (Kalman) -> {frame: (u,v)|None}
                                                       |
                                                       v   THIS MODULE
                              split into flight segments (break on long gaps)
                              inpaint short gaps (linear)         -> continuous arc
                              local quadratic smoothing (order 2) -> projectile shape
                                                       |
                                                       v
                              {frame: TrailPoint(u, v, src)}  -> draw a clean arc
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TrailPoint:
    u: float
    v: float
    src: str            # "detected" | "inpaint" (a bridged short gap)


def _segments(frames: List[int], present: Dict[int, bool], gap_fill_max: int
              ) -> List[List[int]]:
    """Group the present frames into segments, splitting only where the gap to the next
    present frame is LONGER than gap_fill_max (a real break, e.g. ball left the half / a hit
    far away). Gaps <= gap_fill_max are kept inside one segment so we can bridge them."""
    segs: List[List[int]] = []
    cur: List[int] = []
    last = None
    for f in frames:
        if not present.get(f):
            continue
        if last is not None and f - last > gap_fill_max + 1:
            segs.append(cur)
            cur = []
        cur.append(f)
        last = f
    if cur:
        segs.append(cur)
    return segs


def _inpaint(seg: List[int], pts: Dict[int, Tuple[float, float]]
             ) -> Dict[int, TrailPoint]:
    """Linearly fill the (short) gaps between consecutive detected frames in one segment."""
    out: Dict[int, TrailPoint] = {}
    for a, b in zip(seg, seg[1:] + [seg[-1]]):
        ua, va = pts[a]
        out[a] = TrailPoint(ua, va, "detected")
        if b == a:
            continue
        ub, vb = pts[b]
        span = b - a
        for f in range(a + 1, b):                      # interior gap frames
            t = (f - a) / span
            out[f] = TrailPoint(ua + (ub - ua) * t, va + (vb - va) * t, "inpaint")
    return out


# A parabola is used to smooth a window ONLY when the quadratic term is this many times more
# significant than noise (partial F-test on the curvature term). Otherwise we smooth with a
# straight LINE -- so a straight-flying ball is NOT bent into an invented curve. High enough that
# only genuine arcs (lobs/apex) get curved; a real lob's F is huge so it always passes.
CURV_F = 6.0


def _smooth_segment(seg_pts: Dict[int, TrailPoint], window: int
                    ) -> Dict[int, TrailPoint]:
    """Local least-squares smoothing of u(t) and v(t) over a centered window. For each axis we
    fit a straight LINE and a PARABOLA and keep the parabola only if its curvature is
    statistically justified (partial F-test) -- so a straight ball stays straight (no invented
    bow) while a real arc still gets its curve. Falls back to the raw point near the ends."""
    fs = sorted(seg_pts)
    if len(fs) < 5 or window < 5:
        return seg_pts
    half = window // 2
    idx = {f: i for i, f in enumerate(fs)}
    us = [seg_pts[f].u for f in fs]
    vs = [seg_pts[f].v for f in fs]
    out: Dict[int, TrailPoint] = {}
    for f in fs:
        i = idx[f]
        lo, hi = i - half, i + half
        if lo < 0 or hi >= len(fs):                    # window doesn't fit -> keep raw
            out[f] = seg_pts[f]
            continue
        wf = [float(x) for x in fs[lo:hi + 1]]
        su = _fit_eval(wf, us[lo:hi + 1], float(f))
        sv = _fit_eval(wf, vs[lo:hi + 1], float(f))
        out[f] = TrailPoint(su, sv, seg_pts[f].src)
    return out


def _fit_eval(xs: List[float], ys: List[float], x0: float) -> float:
    """Smooth y at x0, choosing a LINE unless a PARABOLA is statistically justified.
    Fit both by least squares (centered for conditioning); keep the parabola only if the
    partial F-test on its quadratic term exceeds CURV_F -- i.e. there is real curvature, not
    just noise a parabola can always over-fit. Pure Python, tiny window, no numpy."""
    n = len(xs)
    mx = sum(xs) / n
    X = [x - mx for x in xs]
    dx = x0 - mx

    # --- straight-line fit (centered: sum(X)=0 so slope/intercept decouple) ---
    Sxx = sum(x * x for x in X)
    b1 = sum(ys) / n                                   # intercept at the window centre
    a1 = (sum(x * y for x, y in zip(X, ys)) / Sxx) if Sxx > 1e-12 else 0.0
    y_line = a1 * dx + b1
    sse1 = sum((y - (a1 * x + b1)) ** 2 for x, y in zip(X, ys))

    if n < 4:                                          # too few points to justify a curve
        return y_line

    # --- parabola fit y = a x^2 + b x + c ---
    s1, s2 = sum(X), Sxx
    s3 = sum(x ** 3 for x in X)
    s4 = sum(x ** 4 for x in X)
    t0, t1, t2 = sum(ys), sum(x * y for x, y in zip(X, ys)), sum(x * x * y for x, y in zip(X, ys))
    A = [[s4, s3, s2, t2], [s3, s2, s1, t1], [s2, s1, float(n), t0]]
    for col in range(3):                               # Gaussian elimination, partial pivot
        piv = max(range(col, 3), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            return y_line                              # degenerate -> trust the line
        A[col], A[piv] = A[piv], A[col]
        for r in range(3):
            if r == col:
                continue
            fac = A[r][col] / A[col][col]
            for k in range(col, 4):
                A[r][k] -= fac * A[col][k]
    a = A[0][3] / A[0][0]
    b = A[1][3] / A[1][1]
    c = A[2][3] / A[2][2]
    sse2 = sum((y - (a * x * x + b * x + c)) ** 2 for x, y in zip(X, ys))

    # partial F-test: is the quadratic term worth it, or is the line enough?
    if sse2 <= 1e-9:                                   # perfect (or near) parabola fit
        return a * dx * dx + b * dx + c
    F = (sse1 - sse2) / (sse2 / (n - 3))
    return (a * dx * dx + b * dx + c) if F > CURV_F else y_line


def smooth_trail(track: Dict[int, Optional[Tuple[float, float]]],
                 gap_fill_max: int = 5, window: int = 7) -> Dict[int, TrailPoint]:
    """Clean per-frame track -> smooth projectile trail.

    track: {frame: (u,v)} for a present ball, {frame: None} (or missing) for no ball.
    Returns {frame: TrailPoint(u, v, src)} only for frames we draw (detected or bridged).
    Long gaps stay absent -> the caller breaks the line there (a true gap, not invented).
    """
    if not track:
        return {}
    frames = sorted(track)
    present = {f: track.get(f) is not None for f in frames}
    pts = {f: track[f] for f in frames if track.get(f) is not None}
    out: Dict[int, TrailPoint] = {}
    for seg in _segments(frames, present, gap_fill_max):
        if len(seg) < 2:                               # a lone point: keep as-is
            f = seg[0]
            out[f] = TrailPoint(pts[f][0], pts[f][1], "detected")
            continue
        filled = _inpaint(seg, pts)
        for f, p in _smooth_segment(filled, window).items():
            out[f] = p
    return out


# --------------------------------------------------------------------------- #
# Unit tests:  python -m core.ball_trail
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import math

    # 1) a parabola with a 2-frame gap + jitter -> gap bridged, arc recovered smoothly
    def parab(f):
        return (100.0 + 8.0 * f, 400.0 - 18.0 * f + 0.9 * f * f)   # u linear, v parabolic

    track = {}
    for f in range(0, 20):
        u, v = parab(f)
        # add small alternating jitter the smoother should remove
        track[f] = (u + (3 if f % 2 else -3), v + (3 if f % 2 else -3))
    track[8] = None                                    # detector miss
    track[9] = None                                    # 2-frame gap
    out = smooth_trail(track, gap_fill_max=5, window=7)
    assert 8 in out and out[8].src == "inpaint", "short gap not bridged"
    assert 9 in out and out[9].src == "inpaint"
    # smoothed interior point should sit close to the TRUE parabola (jitter removed)
    tu, tv = parab(10)
    err = math.hypot(out[10].u - tu, out[10].v - tv)
    assert err < 4.0, f"interior point not smoothed onto the arc: err={err:.1f}"
    print(f"test1 PASS: 2-frame gap bridged, jitter removed (interior err={err:.1f}px)")

    # 2) a long gap (> gap_fill_max) splits into two segments -> NOT bridged
    track = {f: parab(f) for f in range(0, 8)}
    for f in range(8, 16):
        track[f] = None                                # 8-frame hole
    for f in range(16, 24):
        track[f] = parab(f)
    out = smooth_trail(track, gap_fill_max=5, window=7)
    assert all(f not in out for f in range(8, 16)), "long gap wrongly inpainted"
    assert 7 in out and 16 in out
    print("test2 PASS: long gap left as a real break (no invented arc)")

    # 3) a single isolated point survives (no crash, kept as detected)
    out = smooth_trail({5: (10.0, 20.0)}, gap_fill_max=5, window=7)
    assert out[5].src == "detected"
    print("test3 PASS: lone point handled")

    # 4) a STRAIGHT (but noisy) ball must NOT be bent into a curve
    def line(f):
        return (100.0 + 20.0 * f, 300.0 + 14.0 * f)    # straight diagonal, constant velocity
    track = {}
    for f in range(0, 20):
        u, v = line(f)
        track[f] = (u + (4 if f % 2 else -4), v - (4 if f % 2 else -4))   # zig-zag noise
    out = smooth_trail(track, gap_fill_max=5, window=7)
    # collinearity: the smoothed interior points must lie on the straight line (no bow).
    # measure max deviation of smoothed pts from the true line direction
    import math as _m
    devs = []
    for f in range(4, 16):
        tu, tv = line(f)
        devs.append(_m.hypot(out[f].u - tu, out[f].v - tv))
    maxdev = max(devs)
    assert maxdev < 3.0, f"straight ball got bowed: max deviation {maxdev:.1f}px"
    print(f"test4 PASS: straight ball stays straight (max dev {maxdev:.1f}px, no invented curve)")

    print("\nAll ball_trail unit tests passed.")
