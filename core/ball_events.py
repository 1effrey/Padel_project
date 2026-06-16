"""core/ball_events.py
Phase-3 BALL EVENTS -- find the moments that BREAK free flight and segment the
trajectory into ballistic arcs. These events are the physics anchors Phase 5 needs.

THREE EVENT TYPES (detected from the smoothed 2D track's velocity)
  * floor_bounce -- the ball was falling and now rises: the VERTICAL image velocity
    flips from DOWN (+vy) to UP (-vy) while the ball maps INSIDE the court footprint.
    At that instant the ball is on the floor, so the homography maps it ACCURATELY to
    court metres -> this is a Z=0 anchor and gives a reliable in/out call.
  * wall_bounce  -- the HORIZONTAL velocity reverses while the ball is at the court's
    side boundary (the glass). Distinct from a floor bounce (horizontal vs vertical
    reversal) so we never confuse the two.
  * hit          -- a sharp, impulsive change of direction at speed that is NOT a
    floor or wall bounce (a serve / smash / volley). It starts a NEW arc.

WHY VELOCITY, NOT POSITION
  A bounce/hit is a sudden change in the ball's MOTION, so the velocity signal is
  where it shows up cleanest. We use the KALMAN velocity (already smoothed), and only
  compare CONSECUTIVE MEASURED frames -- never across an occlusion gap (the velocity
  is just a held prediction during a gap, so comparing across it would fire false
  events). A short refractory period stops one bounce from firing several frames.

SINGLE-CAMERA LIMITS (be honest)
  Back-glass (court-length) bounces are unreliable here -- this camera measures the
  court-length axis poorly. Bounces that happen DURING an occlusion are missed (no
  velocity to read). Both improve with two-camera fusion (Phase 4) and the physics
  model (Phase 5). What we get reliably: floor bounces in view, side-wall bounces,
  and clear hits -- enough to segment the arcs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

from utils.homography import (COURT_LENGTH_M, COURT_WIDTH_M, Homography,
                              is_inside_court)


@dataclass
class BallEvent:
    """One detected event. Court metres + in/out are filled only when a homography
    is available (and, for in/out, only for floor bounces which sit on the floor)."""

    frame: int
    type: str                       # "floor_bounce" | "wall_bounce" | "hit"
    u: float                        # image pixel of the event
    v: float
    x_m: Optional[float] = None     # court metres (floor-accurate at a bounce)
    y_m: Optional[float] = None
    in_court: Optional[bool] = None # in/out, for floor bounces

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": int(self.frame), "type": self.type,
            "u": float(self.u), "v": float(self.v),
            "x_m": None if self.x_m is None else float(self.x_m),
            "y_m": None if self.y_m is None else float(self.y_m),
            "in_court": self.in_court,
        }


class BallEventDetector:
    """Consumes the per-frame TrackedBall and emits at most one event per frame.

    SWING-based, not single-frame. The Kalman velocity is smoothed, so at a bounce it
    ramps through ~0 rather than jumping +big -> -big. So we track the PEAK speed of
    each velocity 'run' and fire when the sign flips AFTER a fast-enough run -- which
    catches the real reversals a single-frame threshold misses."""

    def __init__(
        self,
        homography: Optional[Homography] = None,
        min_vy_px_s: float = 500.0,
        min_vx_px_s: float = 500.0,
        wall_margin_m: float = 0.6,
        hit_angle_deg: float = 70.0,
        hit_min_speed_px_s: float = 1500.0,
        refractory_frames: int = 3,
        in_out_margin_m: float = 0.1,
    ) -> None:
        self.h = homography
        self.min_vy = float(min_vy_px_s)
        self.min_vx = float(min_vx_px_s)
        self.wall_margin_m = float(wall_margin_m)
        self.cos_hit = math.cos(math.radians(float(hit_angle_deg)))
        self.hit_min_speed = float(hit_min_speed_px_s)
        self.refractory = int(refractory_frames)
        self.in_out_margin_m = float(in_out_margin_m)
        self._reset_runs()
        self._prev_vel: Optional[tuple] = None
        self._prev_measured = False
        self._last_event_frame = -10 ** 9

    def _reset_runs(self) -> None:
        self._vy_sign = 0
        self._vx_sign = 0
        self._peak_down_vy = 0.0    # peak downward (+vy) speed in the current run
        self._peak_vx = 0.0         # peak |vx| in the current vx-sign run

    def reset(self) -> None:
        self._reset_runs()
        self._prev_vel = None
        self._prev_measured = False

    def update(self, frame: int, track) -> Optional[BallEvent]:
        """Feed this frame's TrackedBall; return a BallEvent or None."""
        # only on consecutive MEASURED frames -- never diff velocity across an occlusion
        if track is None or track.x is None or not track.measured:
            self._prev_measured = False
            return None

        vx, vy = float(track.vx), float(track.vy)
        u, v = float(track.x), float(track.y)
        x_m = y_m = None
        if self.h is not None:
            x_m, y_m = self.h.pixel_to_meters((u, v))

        prev_vel, prev_meas = self._prev_vel, self._prev_measured
        self._prev_vel, self._prev_measured = (vx, vy), True

        # accumulate the peak speed of the CURRENT run
        if vy > 0:
            self._peak_down_vy = max(self._peak_down_vy, vy)
        self._peak_vx = max(self._peak_vx, abs(vx))

        sy = 1 if vy > 0 else (-1 if vy < 0 else 0)
        sx = 1 if vx > 0 else (-1 if vx < 0 else 0)

        if not prev_meas:                       # first frame of a new measured segment
            self._vy_sign, self._vx_sign = sy, sx
            self._peak_down_vy = max(0.0, vy)
            self._peak_vx = abs(vx)
            return None

        ev: Optional[BallEvent] = None
        refr_ok = frame - self._last_event_frame >= self.refractory

        # FLOOR BOUNCE: a fast DOWNWARD run (vy>0) then UP (vy<0), ball on the court
        if refr_ok and self._vy_sign > 0 and sy < 0 and self._peak_down_vy >= self.min_vy:
            inside = (is_inside_court((x_m, y_m), self.in_out_margin_m)
                      if x_m is not None else None)
            if x_m is None or inside:
                ev = BallEvent(frame, "floor_bounce", u, v, x_m, y_m, inside)

        # SIDE-WALL BOUNCE: a fast horizontal run reverses near the x boundary
        if (ev is None and refr_ok and sx != 0 and self._vx_sign != 0
                and sx != self._vx_sign and self._peak_vx >= self.min_vx):
            # the airborne ball maps to garbage via the floor homography, so sanity-
            # check the mapped point is on the court lengthwise before trusting it as
            # a side-wall bounce (single-camera wall localisation is approximate).
            if (x_m is not None and 0.0 <= y_m <= COURT_LENGTH_M
                    and (x_m < self.wall_margin_m
                         or x_m > COURT_WIDTH_M - self.wall_margin_m)):
                ev = BallEvent(frame, "wall_bounce", u, v, x_m, y_m, None)

        # HIT: sharp redirection at speed, not a floor/wall bounce
        if ev is None and refr_ok and prev_vel is not None:
            pvx, pvy = prev_vel
            speed = math.hypot(vx, vy)
            pspeed = math.hypot(pvx, pvy)
            if speed > self.hit_min_speed and pspeed > self.hit_min_speed:
                if (pvx * vx + pvy * vy) / (pspeed * speed) < self.cos_hit:
                    ev = BallEvent(frame, "hit", u, v, x_m, y_m, None)

        # advance run state; reset a run's peak when its sign flips
        if sy != 0 and sy != self._vy_sign:
            self._vy_sign = sy
            if sy < 0:
                self._peak_down_vy = 0.0
        if sx != 0 and sx != self._vx_sign:
            self._vx_sign = sx
            self._peak_vx = abs(vx)

        if ev is not None:
            self._last_event_frame = frame
        return ev
