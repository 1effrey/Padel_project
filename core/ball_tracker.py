"""core/ball_tracker.py
Phase-2 2D ball tracker -- a constant-velocity KALMAN FILTER on top of the
TrackNet detector. It does four things the raw per-frame detector cannot:

  1. SMOOTHS the noisy (u, v) detections into a steady track.
  2. PREDICTS through occlusion / missing frames -- when the detector says
     "no ball", the filter keeps coasting on its last velocity instead of
     dropping the track. State is NEVER reset on a gap; the covariance simply
     GROWS each missed frame (we get less and less sure where the ball is), which
     is exactly what should happen.
  3. RAISES the measurement noise R when a detection's confidence is low, so a
     shaky 0.5-confidence point nudges the track less than a crisp 0.9 one.
  4. GATES OUT outliers (OPTIONAL, OFF by default) -- a Mahalanobis check can reject
     a detection wildly inconsistent with the motion. It is OFF by default (gate<=0)
     because a constant-velocity model mispredicts the ball through hits/bounces, so
     a tight gate wrongly rejects the real POST-HIT detections -- the most important
     frames. Proper motion-based outlier rejection arrives with the Phase-5 physics
     model that can actually predict a ballistic arc. Enable here only if your motion
     is smooth.

WHY CONSTANT VELOCITY (and not gravity yet)
  In the IMAGE this is a 2D point that moves smoothly between hits/bounces. A
  constant-velocity model with process noise handles the curvature as "noise".
  The real gravity/drag physics that recovers HEIGHT and the court-length axis is
  Phase 5 (3D) -- deliberately NOT done here. This stays honest 2D image tracking.

STATE  x = [px, py, vx, vy]   (pixels, pixels/second)
  px,py : filtered ball position in FULL-frame pixels
  vx,vy : image-space velocity (px/s)

The tracker is per-camera and stateful: feed it ONE BallDetection per frame in
order (exactly like the detector). It returns a TrackedBall every frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.ball_detector import BallDetection


@dataclass
class TrackedBall:
    """One frame's tracker output. (x, y) is the FILTERED position -- present even
    while coasting through a gap; None only when the track is lost/not yet acquired."""

    x: Optional[float]
    y: Optional[float]
    vx: float = 0.0
    vy: float = 0.0
    std_x: float = 0.0           # 1-sigma position uncertainty (px), grows in gaps
    std_y: float = 0.0
    status: str = "lost"         # "tracking" | "coasting" | "lost"
    measured: bool = False       # a detection was ACCEPTED and fused this frame
    gated: bool = False          # a detection was present but REJECTED as an outlier
    coast: int = 0               # consecutive frames without an accepted measurement
    meas_u: Optional[float] = None   # raw (u,v) of the CHOSEN candidate this frame
    meas_v: Optional[float] = None   # (None while coasting) -> zero-lag display point

    @property
    def speed_px_s(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": None if self.x is None else float(self.x),
            "y": None if self.y is None else float(self.y),
            "vx": float(self.vx), "vy": float(self.vy),
            "speed_px_s": self.speed_px_s,
            "std_x": float(self.std_x), "std_y": float(self.std_y),
            "status": self.status, "measured": bool(self.measured),
            "gated": bool(self.gated), "coast": int(self.coast),
            "meas_u": None if self.meas_u is None else float(self.meas_u),
            "meas_v": None if self.meas_v is None else float(self.meas_v),
        }


class BallTracker:
    """Constant-velocity Kalman filter for the 2D ball position."""

    # measurement matrix: we observe position (px, py) only
    _H = np.array([[1.0, 0.0, 0.0, 0.0],
                   [0.0, 1.0, 0.0, 0.0]])

    def __init__(
        self,
        dt: float = 0.05,
        process_noise: float = 50000.0,
        meas_noise: float = 25.0,
        conf_ref: float = 0.7,
        conf_floor: float = 0.05,
        max_coast_frames: int = 15,
        gate: float = 0.0,
        min_updates_before_gating: int = 3,
        assoc_radius: float = 600.0,
        reject_unassociated: bool = False,
        max_step_px: float = 0.0,
    ) -> None:
        self.dt = float(dt)                       # seconds per frame (1/fps)
        self.q = float(process_noise)             # acceleration PSD (px/s^2 scale)
        self.r_base = float(meas_noise)           # base position variance (px^2)
        self.conf_ref = float(conf_ref)           # confidence that maps to r_base
        self.conf_floor = float(conf_floor)       # never divide by ~0 confidence
        self.max_coast = int(max_coast_frames)    # give up after this many gap frames
        self.gate = float(gate)                   # chi-square(2) gate; <=0 DISABLES it
        self.min_updates_before_gating = int(min_updates_before_gating)
        # among multiple candidates, prefer those within this many px of the predicted
        # position (i.e. consistent with the ball's motion). A soft preference, never
        # a hard reject of the only candidate.
        self.assoc_radius = float(assoc_radius)
        # OPT-IN hard ghost rejection (Phase-2 graft). When True, an ESTABLISHED track
        # that sees NO motion-consistent candidate coasts instead of fusing a far one.
        # OFF by default -> behaviour identical to before (see update_multi for the
        # post-hit-frame caveat). Enable only after measuring hit recall.
        self.reject_unassociated = bool(reject_unassociated)
        # OPT-IN jump gate (default OFF, 0 = disabled): reject a chosen candidate that
        # TELEPORTS from the last MEASURED position by more than max_step_px (scaled up
        # after coast gaps). A real ball moves CONTINUOUSLY -- even a post-hit ball reverses
        # direction but stays NEAR where it just was -- so a huge one-frame jump to a far
        # blob is almost always a false positive (a detection on the fence / a light / a
        # limb). Gating on the last MEASURED position (not the KF prediction, which can point
        # the wrong way right after a hit) is what makes this SAFE -- unlike reject_unassociated
        # it does NOT drop the real post-hit ball. This is the cure for the "ball teleports to
        # the fence" fake-trajectory artifact.
        self.max_step_px = float(max_step_px)
        self._reset()

    # ------------------------------------------------------------------ public
    def update(self, det: BallDetection) -> TrackedBall:
        """Single-detection convenience wrapper around update_multi()."""
        return self.update_multi([det] if det.found else [])

    def update_multi(self, candidates: List[BallDetection]) -> TrackedBall:
        """Advance one frame given a LIST of detection candidates (may be empty).

        Among the candidates we keep the one that best CONTINUES the ball's motion
        (closest to the predicted position), so a MOVING ball wins over a static
        light / limb. This only DISCRIMINATES between candidates -- it never rejects
        the only one, so it cannot lose the track the way hard gating did."""
        # --- not tracking yet: initialise on the strongest candidate ---
        if self._x is None:
            if candidates:
                best = max(candidates, key=lambda c: c.confidence)
                self._init(best)
                return self._out("tracking", measured=True, meas=(best.u, best.v))
            return self._out("lost")

        # --- predict (always): x = F x ; P = F P F^T + Q  (covariance GROWS) ---
        F = self._F()
        self._x = F @ self._x
        self._P = F @ self._P @ F.T + self._Q()

        gated = False
        if candidates:
            # prefer candidates near the predicted position (motion-consistent); fall
            # back to ALL candidates if none are near (e.g. a fast / post-hit ball)
            px, py = float(self._x[0]), float(self._x[1])
            r2 = self.assoc_radius ** 2
            near = [c for c in candidates
                    if (c.u - px) ** 2 + (c.v - py) ** 2 <= r2]
            # OPT-IN hard ghost rejection (reject_unassociated, default OFF): when the
            # track is ESTABLISHED and NOTHING is motion-consistent, treat every candidate
            # as a ghost and COAST rather than fuse a far one. WARNING: a real fast / post-
            # hit ball also jumps far from a constant-velocity prediction, so this can DROP
            # the crucial post-hit frame -- measure hit recall before enabling. Default OFF
            # keeps the original "fall back to ALL candidates" behaviour exactly.
            if (not near and self.reject_unassociated
                    and self._n_updates >= self.min_updates_before_gating):
                pool: List[BallDetection] = []                # all far -> ghosts -> coast
            else:
                pool = near or candidates
            chosen = max(pool, key=lambda c: c.confidence) if pool else None
            # OPT-IN jump gate: reject a chosen candidate that teleports from the last
            # MEASURED position (budget grows with the coast gap, since the ball can move
            # further the longer we have not seen it). Continuous ball motion passes; a
            # one-frame leap to a far false positive (the fence / a light) is rejected -> coast.
            if (chosen is not None and self.max_step_px > 0.0
                    and self._last_meas is not None
                    and self._n_updates >= self.min_updates_before_gating):
                step = float(np.hypot(chosen.u - self._last_meas[0],
                                      chosen.v - self._last_meas[1]))
                if step > self.max_step_px * (1 + self._coast):
                    chosen = None                             # teleport -> reject, coast
            if chosen is not None:
                z = np.array([chosen.u, chosen.v], dtype=float)
                R = self._R(chosen.confidence)
                innov = z - self._H @ self._x                 # measurement residual
                S = self._H @ self._P @ self._H.T + R
                d2 = float(innov @ np.linalg.solve(S, innov)) # Mahalanobis^2
                if (self.gate > 0.0
                        and self._n_updates >= self.min_updates_before_gating
                        and d2 > self.gate):
                    gated = True                              # optional outlier reject
                else:
                    K = self._P @ self._H.T @ np.linalg.inv(S)  # fuse the measurement
                    self._x = self._x + K @ innov
                    self._P = (np.eye(4) - K @ self._H) @ self._P
                    self._coast = 0
                    self._n_updates += 1
                    self._last_meas = (float(chosen.u), float(chosen.v))
                    return self._out("tracking", measured=True, meas=(chosen.u, chosen.v))

        # --- no usable measurement -> coast on the prediction ---
        self._coast += 1
        if self._coast > self.max_coast:
            self._reset()                                     # lost -> re-acquire
            return self._out("lost", gated=gated)
        return self._out("coasting", gated=gated)

    def reset(self) -> None:
        """Drop the track (call between independent clips)."""
        self._reset()

    @staticmethod
    def interpolate_gaps(
        points: List[Optional[Tuple[float, float]]], max_gap: int
    ) -> List[Optional[Tuple[float, float]]]:
        """OFFLINE helper: given a per-frame list of (x, y) or None, linearly fill
        each None-run of length <= max_gap between two known endpoints. Longer gaps
        are left as None. Used for post-hoc smoothing of a finished track (reports);
        the online filter above fills gaps causally via prediction."""
        out = list(points)
        n = len(out)
        i = 0
        while i < n:
            if out[i] is not None:
                i += 1
                continue
            j = i
            while j < n and out[j] is None:
                j += 1
            # gap is out[i..j-1]; bounded by out[i-1] and out[j]
            if 0 < i and j < n and (j - i) <= max_gap:
                (x0, y0), (x1, y1) = out[i - 1], out[j]
                span = j - (i - 1)
                for k in range(i, j):
                    t = (k - (i - 1)) / span
                    out[k] = (x0 + t * (x1 - x0), y0 + t * (y1 - y0))
            i = j
        return out

    # ----------------------------------------------------------------- private
    def _reset(self) -> None:
        self._x: Optional[np.ndarray] = None      # state (4,)
        self._P: Optional[np.ndarray] = None      # covariance (4,4)
        self._coast = 0
        self._n_updates = 0
        self._last_meas: Optional[Tuple[float, float]] = None  # last ACCEPTED (u,v), for the jump gate

    def _init(self, det: BallDetection) -> None:
        self._x = np.array([det.u, det.v, 0.0, 0.0], dtype=float)
        # trust position ~ measurement noise; velocity unknown -> huge variance
        self._P = np.diag([self.r_base, self.r_base, 1e6, 1e6]).astype(float)
        self._coast = 0
        self._n_updates = 1
        self._last_meas = (float(det.u), float(det.v))

    def _F(self) -> np.ndarray:
        dt = self.dt
        return np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

    def _Q(self) -> np.ndarray:
        """Discrete white-noise-acceleration process noise."""
        dt = self.dt
        dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
        return self.q * np.array([
            [dt4 / 4, 0, dt3 / 2, 0],
            [0, dt4 / 4, 0, dt3 / 2],
            [dt3 / 2, 0, dt2, 0],
            [0, dt3 / 2, 0, dt2]], dtype=float)

    def _R(self, conf: float) -> np.ndarray:
        """Position measurement noise; LOW confidence -> LARGER R (trust it less)."""
        scale = self.conf_ref / max(conf, self.conf_floor)
        return np.eye(2) * (self.r_base * scale)

    def _out(self, status: str, measured: bool = False, gated: bool = False,
             meas: Optional[Tuple[float, float]] = None) -> TrackedBall:
        mu, mv = meas if meas is not None else (None, None)
        if self._x is None:
            return TrackedBall(x=None, y=None, status=status, measured=measured,
                               gated=gated, coast=self._coast, meas_u=mu, meas_v=mv)
        std = np.sqrt(np.clip(np.diag(self._P), 0.0, None))
        return TrackedBall(
            x=float(self._x[0]), y=float(self._x[1]),
            vx=float(self._x[2]), vy=float(self._x[3]),
            std_x=float(std[0]), std_y=float(std[1]),
            status=status, measured=measured, gated=gated, coast=self._coast,
            meas_u=mu, meas_v=mv)


# --------------------------------------------------------------------------- #
# Smoke test: feed a synthetic moving ball with a gap + an outlier.
#   python -m core.ball_tracker
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # gate=9.21 ON here to DEMONSTRATE outlier rejection on smooth (constant-
    # velocity) motion -- it is OFF in the real config (agile ball, see class docstring).
    trk = BallTracker(dt=0.05, max_coast_frames=10, gate=9.21)
    print("frame  status     x     y    coast  measured gated")
    for f in range(20):
        if 6 <= f <= 9:                         # 4-frame occlusion gap
            det = BallDetection(found=False, reason="no-ball")
        elif f == 13:                           # one wild outlier
            det = BallDetection(found=True, u=3000.0, v=200.0, confidence=0.6, reason="ok")
        else:                                   # ball moving right at 40 px/frame
            det = BallDetection(found=True, u=100.0 + 40 * f, v=500.0,
                                confidence=0.8, reason="ok")
        t = trk.update(det)
        xs = "  -- " if t.x is None else f"{t.x:6.0f}"
        ys = "  -- " if t.y is None else f"{t.y:5.0f}"
        print(f"{f:>4}  {t.status:<9}{xs}{ys}   {t.coast:>3}    "
              f"{str(t.measured):<7} {t.gated}")
    print("\nExpect: coasting through frames 6-9, frame 13 GATED (outlier rejected).")
