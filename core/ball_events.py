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
  * hit          -- a sharp, impulsive change of direction OR a sudden speed GAIN at
    speed that is NOT a floor or wall bounce (a serve / smash / volley). It starts a
    NEW arc. A bounce LOSES energy; a player stroke ADDS it -- so a speed gain is a
    strong hit cue on top of the turn angle.
  * player_hit   -- a hit whose contact point is within racket reach of a player's
    WRIST keypoint (so we know a PLAYER struck it, not a free-air redirect). Pass the
    per-frame wrist points to update(); trustworthy only for the NEAR player -- far-side
    keypoints are unreliable, which is the other camera's job.

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
    type: str                       # "floor_bounce" | "wall_bounce" | "hit" | "player_hit"
    u: float                        # image pixel of the event
    v: float
    x_m: Optional[float] = None     # court metres (floor-accurate at a bounce)
    y_m: Optional[float] = None
    in_court: Optional[bool] = None # in/out, for floor bounces
    player_hand: Optional[str] = None    # "left"/"right"/"" for a player_hit; else None
    speed_change: Optional[float] = None  # px/s speed change at a hit (>0 = sped up)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": int(self.frame), "type": self.type,
            "u": float(self.u), "v": float(self.v),
            "x_m": None if self.x_m is None else float(self.x_m),
            "y_m": None if self.y_m is None else float(self.y_m),
            "in_court": self.in_court,
            "player_hand": self.player_hand,
            "speed_change": None if self.speed_change is None else float(self.speed_change),
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
        hit_min_speed_gain_px_s: float = 800.0,
        racket_reach_px: float = 150.0,
        refractory_frames: int = 3,
        in_out_margin_m: float = 0.1,
    ) -> None:
        self.h = homography
        self.min_vy = float(min_vy_px_s)
        self.min_vx = float(min_vx_px_s)
        self.wall_margin_m = float(wall_margin_m)
        self.cos_hit = math.cos(math.radians(float(hit_angle_deg)))
        self.hit_min_speed = float(hit_min_speed_px_s)
        self.hit_min_speed_gain = float(hit_min_speed_gain_px_s)
        self.racket_reach = float(racket_reach_px)
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

    def _nearest_wrist(self, u: float, v: float, wrists) -> Optional[str]:
        """Match the event point to a player's WRIST. `wrists` is this frame's wrist
        points as (u, v) or (u, v, hand). Returns the nearest wrist's hand label ("" if
        unlabeled) when one is within racket reach of (u, v), else None. Reliable only
        for the NEAR player -- the caller should pass near-player wrists only."""
        if not wrists:
            return None
        best_d = self.racket_reach
        best_hand: Optional[str] = None
        for w in wrists:
            d = math.hypot(u - float(w[0]), v - float(w[1]))
            if d <= best_d:
                best_d, best_hand = d, (str(w[2]) if len(w) > 2 else "")
        return best_hand

    def update(self, frame: int, track, wrists=None) -> Optional[BallEvent]:
        """Feed this frame's TrackedBall (+ optional player wrist points for this frame
        as a list of (u, v) or (u, v, hand)); return a BallEvent or None."""
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

        # HIT: a sharp redirection OR a sudden speed GAIN, at speed. A bounce loses
        # energy; a player stroke adds it -- so a speed gain is a hit cue on its own.
        if ev is None and refr_ok and prev_vel is not None:
            pvx, pvy = prev_vel
            speed = math.hypot(vx, vy)
            pspeed = math.hypot(pvx, pvy)
            dspeed = speed - pspeed
            sharp_turn = (pspeed > self.hit_min_speed and speed > self.hit_min_speed
                          and (pvx * vx + pvy * vy) / (pspeed * speed) < self.cos_hit)
            speed_gain = speed > self.hit_min_speed and dspeed > self.hit_min_speed_gain
            if sharp_turn or speed_gain:
                # a deflection within racket reach of a wrist is a PLAYER hit
                hand = self._nearest_wrist(u, v, wrists)
                etype = "player_hit" if hand is not None else "hit"
                ev = BallEvent(frame, etype, u, v, x_m, y_m, None,
                               player_hand=hand, speed_change=dspeed)

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


if __name__ == "__main__":  # synthetic self-test -- no weights / footage needed
    from types import SimpleNamespace

    def _trk(x, y, vx, vy):
        return SimpleNamespace(x=x, y=y, vx=vx, vy=vy, measured=True)

    def _run(seq, wrists_seq=None):
        det = BallEventDetector(homography=None)
        out = []
        for i, t in enumerate(seq):
            w = wrists_seq[i] if wrists_seq else None
            ev = det.update(i, t, wrists=w)
            out.append(ev.type if ev else None)
        return out

    # floor bounce: a fast DOWN run (vy>0) then UP (vy<0)
    floor = [_trk(100, 100, 0, 600), _trk(100, 160, 0, 650),
             _trk(100, 230, 0, 700), _trk(100, 250, 0, -650)]
    assert "floor_bounce" in _run(floor), _run(floor)

    # hit: sharp reversal at speed; same point + a wrist on it -> player_hit
    hit = [_trk(500, 300, 2000, 0), _trk(540, 300, -2000, 80)]
    assert _run(hit)[-1] == "hit", _run(hit)
    assert _run(hit, [None, [(540, 300, "right")]])[-1] == "player_hit"

    # energy cue: a clear speed gain with little turn is still a hit
    energy = [_trk(500, 300, 1600, 0), _trk(560, 300, 2600, 0)]
    assert _run(energy)[-1] == "hit", _run(energy)

    # a smooth slow arc fires nothing
    smooth = [_trk(100, 100, 500, 100), _trk(130, 106, 500, 120), _trk(160, 113, 500, 140)]
    assert all(x is None for x in _run(smooth)), _run(smooth)

    print("core.ball_events self-test: PASS")
