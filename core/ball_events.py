"""core/ball_events.py
Phase-3 BALL EVENTS -- find the moments that BREAK free flight and segment the
trajectory into ballistic arcs. These events are the physics anchors Phase 5 needs.

THREE EVENT TYPES (detected from the smoothed 2D track's velocity)
  * floor_bounce -- the ball was falling and now rises: the VERTICAL image velocity
    flips from DOWN (+vy) to UP (-vy) while the ball maps INSIDE the court footprint.
    At that instant the ball is on the floor, so the homography maps it ACCURATELY to
    court metres -> this is a Z=0 anchor and gives a reliable in/out call.
  * wall_bounce  -- the HORIZONTAL velocity reverses while the ball is at the glass:
    either inside a drawn image-space GLASS REGION (parallax-free, from --calibrate-walls)
    or, lacking that, near the court's side boundary via the homography. Distinct from a
    floor bounce (horizontal vs vertical reversal) so we never confuse the two.
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
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from utils.homography import (COURT_LENGTH_M, COURT_WIDTH_M, Homography,
                              is_inside_court)


def _point_in_poly(u: float, v: float, poly) -> bool:
    """Ray-casting point-in-polygon. `poly` is a list of [x, y] in the same (full-res
    image) coordinates as the ball track. Pure-python -- no cv2/numpy dependency."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i][0], poly[i][1]
        xj, yj = poly[j][0], poly[j][1]
        if (yi > v) != (yj > v) and u < (xj - xi) * (v - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


@dataclass
class BallEvent:
    """One detected event. Court metres + in/out are filled only when a homography
    is available (and, for in/out, only for floor bounces which sit on the floor)."""

    frame: int
    type: str                       # "floor_bounce"|"wall_bounce"|"hit"|"player_hit"|"net_hit"
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
        racket_reach_px: float = 220.0,
        racket_min_px: float = 70.0,
        player_turn_deg: float = 60.0,
        player_min_speed_px_s: float = 300.0,
        player_speed_drop: float = 0.5,
        player_drive_ratio: float = 3.0,
        player_cooldown_frames: int = 10,
        net_polygon: Optional[List] = None,
        net_min_speed_px_s: float = 600.0,
        net_speed_drop: float = 0.45,
        net_turn_deg: float = 40.0,
        net_margin_px: float = 120.0,
        refractory_frames: int = 3,
        in_out_margin_m: float = 0.1,
        detect_bounces: bool = True,
        glass_regions=None,
    ) -> None:
        self.h = homography
        self.min_vy = float(min_vy_px_s)
        self.min_vx = float(min_vx_px_s)
        self.wall_margin_m = float(wall_margin_m)
        self.cos_hit = math.cos(math.radians(float(hit_angle_deg)))
        self.hit_min_speed = float(hit_min_speed_px_s)
        self.hit_min_speed_gain = float(hit_min_speed_gain_px_s)
        self.racket_reach = float(racket_reach_px)
        self.racket_min = float(racket_min_px)
        self.player_cos_turn = math.cos(math.radians(float(player_turn_deg)))
        self.player_min_speed = float(player_min_speed_px_s)
        self.player_speed_drop = float(player_speed_drop)
        self.player_drive_ratio = float(player_drive_ratio)
        self.player_cooldown = int(player_cooldown_frames)
        # net region in IMAGE pixels (a calibrated polygon); used to gate net hits.
        self._net_poly = (np.array(net_polygon, dtype=np.int32).reshape(-1, 1, 2)
                          if net_polygon else None)
        self.net_min_speed = float(net_min_speed_px_s)
        self.net_speed_drop = float(net_speed_drop)
        self.net_cos_turn = math.cos(math.radians(float(net_turn_deg)))
        self.net_margin_px = float(net_margin_px)
        self.refractory = int(refractory_frames)
        self.in_out_margin_m = float(in_out_margin_m)
        # when False, floor_bounce / wall_bounce are NOT emitted -- use this when another
        # part of the pipeline already owns bounce detection (we still emit player/net hits).
        self.detect_bounces = bool(detect_bounces)
        # drawn image-space glass/wall polygons (full-res coords), for wall-hit gating
        self._glass = [list(map(list, poly)) for poly in (glass_regions or [])]
        self._reset_runs()
        self._prev_vel: Optional[tuple] = None
        self._prev_measured = False
        self._last_event_frame = -10 ** 9
        self._net_peak = 0.0          # peak speed during the current net-band visit
        self._net_fired = False       # one net event per net-band visit
        self._rk_peak = 0.0           # racket zone: peak/min speed, entry dir, min point
        self._rk_min = 0.0
        self._rk_prev = 0.0
        self._rk_entry: Optional[tuple] = None
        self._rk_min_uv = (0.0, 0.0)
        self._rk_min_frame = 0
        self._rk_near = False
        self._rk_fired = False        # one player hit per racket-zone visit
        self._last_player_frame = -10 ** 9   # cooldown between consecutive player hits

    def _reset_runs(self) -> None:
        self._vy_sign = 0
        self._vx_sign = 0
        self._peak_down_vy = 0.0    # peak downward (+vy) speed in the current run
        self._peak_vx = 0.0         # peak |vx| in the current vx-sign run

    def reset(self) -> None:
        self._reset_runs()
        self._prev_vel = None
        self._prev_measured = False
        self._net_peak = 0.0
        self._net_fired = False
        self._rk_peak = self._rk_min = self._rk_prev = 0.0
        self._rk_entry = None
        self._rk_min_uv = (0.0, 0.0)
        self._rk_min_frame = 0
        self._rk_near = self._rk_fired = False
        self._last_player_frame = -10 ** 9

    def _nearest_wrist(self, u: float, v: float, wrists) -> Optional[str]:
        """Hand label of the nearest wrist IF the ball sits at RACKET-HEAD distance from it
        (racket_min .. racket_reach). A ball IN the hand (< racket_min, e.g. a bare-hand
        grab off the ground) or too far (> racket_reach) returns None -- so only the
        racket-holding hand, meeting the ball at the racket head, counts as a contact.
        `wrists` is this frame's wrist points as (u, v) or (u, v, hand)."""
        if not wrists:
            return None
        best_d, best = 1e18, None
        for w in wrists:
            d = math.hypot(u - float(w[0]), v - float(w[1]))
            if d < best_d:
                best_d, best = d, w
        if best is None or not (self.racket_min <= best_d <= self.racket_reach):
            return None                      # in-hand grab (too close) or too far -> not a contact
        return str(best[2]) if len(best) > 2 else ""

    def _in_glass(self, u: float, v: float) -> bool:
        """True if (u, v) falls inside any drawn glass/wall region."""
        return any(_point_in_poly(u, v, poly) for poly in self._glass)

    def _in_net(self, u: float, v: float) -> bool:
        """True if (u, v) is inside the calibrated NET region, or within net_margin_px of
        it. The margin gives a little tolerance for contacts just outside the band."""
        if self._net_poly is None:
            return False
        d = cv2.pointPolygonTest(self._net_poly, (float(u), float(v)), True)
        return d >= -self.net_margin_px

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
        speed = math.hypot(vx, vy)
        pspeed = math.hypot(*prev_vel) if prev_vel is not None else 0.0
        cosang = 1.0
        if prev_vel is not None and pspeed > 0.0 and speed > 0.0:
            pvx, pvy = prev_vel
            cosang = (pvx * vx + pvy * vy) / (pspeed * speed)

        # Track the PEAK speed during a continuous stay in the net band, so a GRADUAL
        # slow-down (a ball trickling down the net over several 20-fps frames) still reads
        # as a stop. Reset on leaving the band; fire at most ONE net event per visit.
        in_net = self._in_net(u, v)
        if in_net:
            self._net_peak = max(self._net_peak, speed, pspeed)
        else:
            self._net_peak = 0.0
            self._net_fired = False

        # PLAYER hit -- ONE event per racket-zone visit, PINNED TO THE DEFLECTION POINT
        # (the contact: where the ball's speed bottoms out / its direction flips at the
        # racket). The racket is ~0.45 m past the wrist and SWINGS, so racket_reach is a
        # generous zone around the wrist. While the ball stays in the zone we track its
        # PEAK speed, its ENTRY direction, and the running speed MINIMUM (with its u,v).
        # We fire ONCE -- not every frame -- when the racket has clearly acted: the ball
        # REVERSED direction, or it slowed to the contact and is now LEAVING (rising), or
        # it was DRIVEN up from a low -- and we report the event AT that minimum point.
        hand = self._nearest_wrist(u, v, wrists)
        near_wrist = hand is not None
        player_hit = None                              # (frame, u, v, hand) when confirmed
        if near_wrist and not self._rk_near:           # ENTERED the racket zone
            self._rk_peak = self._rk_min = self._rk_prev = speed
            self._rk_entry = (vx / speed, vy / speed) if speed > 0 else None
            self._rk_min_uv = (u, v)
            self._rk_min_frame = frame
            self._rk_fired = False
        elif near_wrist:                               # still in the zone
            if speed > self._rk_peak:
                self._rk_peak = speed
                if speed > 0:
                    self._rk_entry = (vx / speed, vy / speed)
            if speed < self._rk_min:
                self._rk_min = speed
                self._rk_min_uv = (u, v)
                self._rk_min_frame = frame
            if not self._rk_fired and self._rk_peak >= self.player_min_speed:
                rising = speed > self._rk_prev + 1.0
                came_down = self._rk_min <= self.player_speed_drop * self._rk_peak
                redirected = (speed > 0 and self._rk_entry is not None
                              and (self._rk_entry[0] * vx + self._rk_entry[1] * vy) / speed
                              < self.player_cos_turn)
                drove = (self._rk_min > 0 and speed >= self.player_min_speed
                         and speed >= self.player_drive_ratio * self._rk_min)
                if ((redirected or (rising and came_down) or drove)
                        and frame - self._last_player_frame >= self.player_cooldown):
                    mu, mv = self._rk_min_uv
                    player_hit = (self._rk_min_frame, mu, mv, hand)
                    self._rk_fired = True
                    self._last_player_frame = frame
            self._rk_prev = speed
        else:                                          # left the zone -> reset
            self._rk_peak = self._rk_min = self._rk_prev = 0.0
            self._rk_entry = None
            self._rk_fired = False
        self._rk_near = near_wrist

        if in_net:
            # IN THE NET BAND only a real CONTACT fires (floor/wall/generic-hit suppressed
            # -- the homography is unreliable and balls just pass over). The ball must have
            # come in FAST (peak), then either DEFLECT (changed direction AND slowed) or
            # STOP (slowed to a fraction of its in-net peak, even gradually). A player
            # VOLLEY (near a wrist) is a player_hit. A smooth pass-over keeps its speed and
            # leaves the band -> nothing.
            came_fast = self._net_peak >= self.net_min_speed
            net_deflect = came_fast and cosang < self.net_cos_turn and speed < pspeed
            net_stop = came_fast and speed <= self.net_speed_drop * self._net_peak
            if refr_ok and not self._net_fired and player_hit is not None:
                pf, pu, pv, ph = player_hit
                ev = BallEvent(pf, "player_hit", pu, pv, None, None, None, player_hand=ph)
                self._net_fired = True
            elif refr_ok and not self._net_fired and (net_deflect or net_stop):
                ev = BallEvent(frame, "net_hit", u, v, x_m, y_m, None,
                               speed_change=speed - pspeed)
                self._net_fired = True
        else:
            # FLOOR BOUNCE: a fast DOWNWARD run (vy>0) then UP (vy<0), ball on the court
            if (self.detect_bounces and refr_ok and self._vy_sign > 0 and sy < 0
                    and self._peak_down_vy >= self.min_vy):
                inside = (is_inside_court((x_m, y_m), self.in_out_margin_m)
                          if x_m is not None else None)
                if x_m is None or inside:
                    ev = BallEvent(frame, "floor_bounce", u, v, x_m, y_m, inside)

            # SIDE-WALL BOUNCE: a fast horizontal run reverses AT the glass. Prefer the
            # drawn image-space glass region (parallax-free for an airborne ball); fall
            # back to the homography side-boundary when no regions are drawn.
            if (ev is None and self.detect_bounces and refr_ok and sx != 0
                    and self._vx_sign != 0 and sx != self._vx_sign
                    and self._peak_vx >= self.min_vx):
                near_boundary = (x_m is not None and 0.0 <= y_m <= COURT_LENGTH_M
                                 and (x_m < self.wall_margin_m
                                      or x_m > COURT_WIDTH_M - self.wall_margin_m))
                if self._in_glass(u, v) or near_boundary:
                    ev = BallEvent(frame, "wall_bounce", u, v, x_m, y_m, None)

            # PLAYER HIT: the racket acted on the ball (one per visit, at the deflection
            # point computed above). A change with NO racket nearby is dropped -- a ball
            # only deflects at a player / net / wall / floor; a mid-air "hit" is noise.
            if ev is None and refr_ok and player_hit is not None:
                pf, pu, pv, ph = player_hit
                ev = BallEvent(pf, "player_hit", pu, pv, None, None, None, player_hand=ph)

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

    # detect_bounces=False: the SAME floor motion emits nothing (another pipeline owns
    # bounces) -- but player/net hits still fire (tested below run on the default detector).
    dnb = BallEventDetector(homography=None, detect_bounces=False)
    assert all(dnb.update(i, t) is None for i, t in enumerate(floor)), "bounce not suppressed"

    # PLAYER HIT: exactly ONE per racket-zone visit, at the deflection point. With NO
    # wrist the same motion is dropped as noise.
    wr = [(350, 300, "right")]                  # wrist; ball at x~500 -> ~150 px = racket head
    def wn(seq):
        return [wr] * len(seq)
    redir = [_trk(500, 300, 2000, 0), _trk(515, 300, 1800, 0), _trk(525, 300, -1700, 0)]
    o = _run(redir, wn(redir))                                       # REDIRECT (reversed)
    assert o.count("player_hit") == 1 and o[-1] == "player_hit", o
    assert _run(redir)[-1] is None, _run(redir)                      # no wrist -> noise
    stop = [_trk(500, 300, 1800, 0), _trk(512, 300, 1000, 0),
            _trk(518, 300, 150, 0), _trk(524, 300, 500, 0)]          # STOP then leave
    assert _run(stop, wn(stop)).count("player_hit") == 1, _run(stop, wn(stop))
    drive = [_trk(500, 300, 200, 0), _trk(515, 300, 250, 0), _trk(545, 300, 1200, 0)]
    assert _run(drive, wn(drive)).count("player_hit") == 1, _run(drive, wn(drive))   # DRIVE
    # GRAB: ball IN the hand (AT the wrist, < racket_min) stopping -> NOT a hit
    gw = [(508, 300, "right")]
    grab = [_trk(500, 300, 800, 0), _trk(505, 300, 350, 0), _trk(508, 300, 40, 0)]
    assert _run(grab, [gw, gw, gw]).count("player_hit") == 0, _run(grab, [gw, gw, gw])

    # a smooth slow arc fires nothing
    smooth = [_trk(100, 100, 500, 100), _trk(130, 106, 500, 120), _trk(160, 113, 500, 140)]
    assert all(x is None for x in _run(smooth)), _run(smooth)

    # wall bounce via a drawn glass region: horizontal reversal inside it (no homography)
    glass = [[[750, 250], [850, 250], [850, 350], [750, 350]]]
    dw = BallEventDetector(homography=None, glass_regions=glass)
    dw.update(0, _trk(800, 300, 600, 0))
    evw = dw.update(1, _trk(800, 300, -600, 0))
    assert evw is not None and evw.type == "wall_bounce", evw

    # NET HIT: a fast ball inside the net region whose speed COLLAPSES (stops at the net)
    netbox = [[0, 0], [100, 0], [100, 100], [0, 100]]
    dn = BallEventDetector(homography=None, net_polygon=netbox,
                           net_min_speed_px_s=600, net_speed_drop=0.45)
    dn.update(0, _trk(50, 50, 1500, 0))
    evn = dn.update(1, _trk(55, 50, 60, 0))            # 1500 -> 60 px/s inside the box
    assert evn is not None and evn.type == "net_hit", evn
    # a SMOOTH pass-over (speed kept) across the same box must NOT fire net_hit
    dp = BallEventDetector(homography=None, net_polygon=netbox,
                           net_min_speed_px_s=600, net_speed_drop=0.45)
    dp.update(0, _trk(40, 50, 1500, 0))
    evp = dp.update(1, _trk(70, 50, 1500, 0))          # speed maintained -> not a net hit
    assert evp is None or evp.type != "net_hit", evp
    # NET HIT via DEFLECTION: came in fast, turned ~90 deg, and SLOWED (but did NOT stop)
    dd = BallEventDetector(homography=None, net_polygon=netbox,
                           net_min_speed_px_s=600, net_speed_drop=0.45, net_turn_deg=40)
    dd.update(0, _trk(50, 50, 2000, 0))
    evdf = dd.update(1, _trk(55, 50, 0, 900))          # 2000->900, turned 90 deg
    assert evdf is not None and evdf.type == "net_hit", evdf

    print("core.ball_events self-test: PASS")
