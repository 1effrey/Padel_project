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
    type: str                       # "floor_bounce"|"wall_bounce"|"fence_hit"|"hit"|"player_hit"|"net_hit"
    u: float                        # image pixel of the event
    v: float
    x_m: Optional[float] = None     # court metres (floor-accurate at a bounce)
    y_m: Optional[float] = None
    in_court: Optional[bool] = None # in/out, for floor bounces
    player_hand: Optional[str] = None    # "left"/"right"/"" for a player_hit; else None
    speed_change: Optional[float] = None  # px/s speed change at a hit (>0 = sped up)
    # incoming (pre-contact) and outgoing (post-contact) velocity in px/s, attached for
    # downstream feature logging (speeds / angles). Pure data -- no effect on detection.
    vx_in: Optional[float] = None
    vy_in: Optional[float] = None
    vx_out: Optional[float] = None
    vy_out: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": int(self.frame), "type": self.type,
            "u": float(self.u), "v": float(self.v),
            "x_m": None if self.x_m is None else float(self.x_m),
            "y_m": None if self.y_m is None else float(self.y_m),
            "in_court": self.in_court,
            "player_hand": self.player_hand,
            "speed_change": None if self.speed_change is None else float(self.speed_change),
            "vx_in": self.vx_in, "vy_in": self.vy_in,
            "vx_out": self.vx_out, "vy_out": self.vy_out,
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
        player_contact_px: float = 90.0,
        player_pass_frames: int = 4,
        net_polygon: Optional[List] = None,
        net_min_speed_px_s: float = 600.0,
        net_speed_drop: float = 0.45,
        net_turn_deg: float = 40.0,
        net_margin_px: float = 120.0,
        refractory_frames: int = 3,
        in_out_margin_m: float = 0.1,
        detect_bounces: bool = True,
        glass_regions=None,
        fence_regions=None,
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
        # STRICT player-hit rule:
        #   player_contact_px ... TINY radius around the racket-holding wrist; the ball must
        #                         be this close to count as a contact (no large "reach zone").
        #   player_pass_frames .. how many frames after the contact we wait for a deflection
        #                         before declaring a pass-through (a MISS).
        self.player_contact_px = float(player_contact_px)
        self.player_pass_frames = int(player_pass_frames)
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
        # drawn image-space GLASS wall polygons (full-res coords) -> a deflection here is a
        # WALL hit (the ball stays in play).
        self._glass = [list(map(list, poly)) for poly in (glass_regions or [])]
        # drawn image-space METAL FENCE polygons -> a deflection here is a FENCE hit = OUT.
        self._fence = [list(map(list, poly)) for poly in (fence_regions or [])]
        self._reset_runs()
        self._prev_vel: Optional[tuple] = None
        self._prev_measured = False
        self._last_event_frame = -10 ** 9
        self._net_peak = 0.0          # peak speed during the current net-band visit
        self._net_fired = False       # one net event per net-band visit
        # pending-contact state for the STRICT player-hit rule (see _update_player_hit)
        self._pc_active = False       # a wrist contact is being evaluated for a deflection
        self._pc_frame = 0            # frame the ball first touched the wrist
        self._pc_uv = (0.0, 0.0)      # contact point (where the marker goes)
        self._pc_hand: Optional[str] = None   # which wrist (the racket-holding one)
        self._pc_vel_in: Optional[tuple] = None   # incoming (approach) velocity vector
        self._pc_age = 0              # frames elapsed since the contact began
        self._player_lock_until = -10 ** 9   # no new player hit before this frame (10-frame lock)

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
        self._pc_active = False
        self._pc_frame = 0
        self._pc_uv = (0.0, 0.0)
        self._pc_hand = None
        self._pc_vel_in = None
        self._pc_age = 0
        self._player_lock_until = -10 ** 9

    def _nearest_wrist_dist(self, u: float, v: float, wrists):
        """(distance, hand) of the wrist NEAREST the ball this frame, or (inf, None).

        ISOLATING THE RACKET-HOLDING ARM: we have no racket detector, so we take the wrist
        the ball actually CONTACTS (the nearest one) as the racket-holding arm -- a ball only
        deflects where the racket meets it. The OTHER (non-dominant) arm's wrist is simply
        never the nearest at a real contact, so it is ignored by construction.
        `wrists` is this frame's wrist points as (u, v) or (u, v, hand)."""
        if not wrists:
            return 1e18, None
        best_d, best = 1e18, None
        for w in wrists:
            d = math.hypot(u - float(w[0]), v - float(w[1]))
            if d < best_d:
                best_d, best = d, w
        hand = (str(best[2]) if best is not None and len(best) > 2 else "")
        return best_d, (hand if best is not None else None)

    @staticmethod
    def _deflected(vel_in, vx: float, vy: float, cos_turn: float) -> bool:
        """True if the trajectory vector TURNED more than the threshold between the incoming
        (approach) velocity `vel_in` and the current velocity (vx, vy).

        We compare the two vectors by the cosine of the angle between them:
            cos(angle) = (a . b) / (|a| |b|)
        cos ~ +1 -> same heading (the ball passed straight through -> NOT a hit),
        cos < cos_turn -> it bent more than `player_turn_deg` (a real deflection -> a hit).
        A sign flip in DX or DY (e.g. vx: +1500 -> -1500) drives cos negative, so this one
        test captures both 'reversed X' and 'reversed Y' as well as any sharp redirect.
        A ball that merely STOPS (current speed ~ 0) gives no direction -> not a deflection
        (so a bare-hand grab / catch is not mistaken for a racket hit)."""
        if vel_in is None:
            return False
        pvx, pvy = vel_in
        ps = math.hypot(pvx, pvy)
        cs = math.hypot(vx, vy)
        if ps <= 1e-6 or cs <= 1e-6:
            return False
        cosang = (pvx * vx + pvy * vy) / (ps * cs)
        return cosang < cos_turn

    def _update_player_hit(self, frame: int, u: float, v: float, vx: float, vy: float,
                           prev_vel, wrists) -> Optional[BallEvent]:
        """STRICT player-hit detection -- the three rules, in order:

        1) PROXIMAL DETECTION  -- the ball must be within player_contact_px of the racket-
           holding wrist (the nearest wrist). A larger 'reach' is deliberately NOT used, so
           the hit cannot fire while the ball is merely approaching the player.
        2) DEFLECTION VALIDATION -- proximity alone is NOT enough. We remember the incoming
           velocity when the ball first touches the wrist, then on the following frame(s)
           check whether the trajectory actually bent (see _deflected). A ball that passes
           through the wrist with its DX/DY heading unchanged is a MISS.
        3) NO DOUBLE-HITS -- once a deflection registers, we LOCK player hits for
           player_cooldown frames so a single swing cannot fire several times.

        Returns a player_hit BallEvent (pinned to the CONTACT point/frame, so the marker
        lands exactly where the ball met the racket) or None.
        """
        # rule 3: still inside the post-hit lock window -> emit nothing.
        if frame < self._player_lock_until:
            return None

        d, hand = self._nearest_wrist_dist(u, v, wrists)
        near = hand is not None and d <= self.player_contact_px

        # rule 1: open a pending contact the first frame the ball touches the wrist,
        # recording the INCOMING velocity (the approach) to compare against later.
        if not self._pc_active:
            if not near:
                return None
            self._pc_active = True
            self._pc_frame = frame
            self._pc_uv = (u, v)
            self._pc_hand = hand
            self._pc_vel_in = prev_vel
            self._pc_age = 0
        else:
            self._pc_age += 1

        # rule 2: did the trajectory bend right after contact? -> a real hit.
        if self._deflected(self._pc_vel_in, vx, vy, self.player_cos_turn):
            # floor-project the contact pixel to court metres (approximate for an airborne
            # contact, but the best single-camera estimate; height comes from triangulation).
            cx = cy = None
            if self.h is not None:
                cx, cy = self.h.pixel_to_meters(self._pc_uv)
            ev = BallEvent(self._pc_frame, "player_hit", self._pc_uv[0], self._pc_uv[1],
                           cx, cy, None, player_hand=self._pc_hand)
            self._player_lock_until = frame + self.player_cooldown   # rule 3: lock the swing
            self._pc_active = False
            return ev

        # no deflection yet: if the ball has LEFT the wrist or we have waited long enough,
        # it passed through -> a MISS; drop the pending contact.
        if (not near) or self._pc_age >= self.player_pass_frames:
            self._pc_active = False
        return None

    def _in_glass(self, u: float, v: float) -> bool:
        """True if (u, v) falls inside any drawn GLASS wall region (in-play surface)."""
        return any(_point_in_poly(u, v, poly) for poly in self._glass)

    def _in_fence(self, u: float, v: float) -> bool:
        """True if (u, v) falls inside any drawn METAL FENCE region (a hit here = OUT)."""
        return any(_point_in_poly(u, v, poly) for poly in self._fence)

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

        # PLAYER HIT (strict rule): tiny wrist proximity + a real post-contact deflection,
        # locked for a few frames after firing. Evaluated EVERY frame by its own helper so
        # the contact/deflection state is tracked cleanly. See _update_player_hit.
        player_ev = self._update_player_hit(frame, u, v, vx, vy, prev_vel, wrists)

        # A confirmed PLAYER HIT wins over everything else (a volley at the net is a player
        # hit, not a net hit). Otherwise fall through to the net / floor / wall logic.
        if player_ev is not None and refr_ok:
            ev = player_ev
        elif in_net:
            # IN THE NET BAND only a real CONTACT fires (floor/wall suppressed -- the
            # homography is unreliable and balls just pass over). The ball must have come in
            # FAST (peak), then either DEFLECT (changed direction AND slowed) or STOP (slowed
            # to a fraction of its in-net peak, even gradually). A smooth pass-over keeps its
            # speed and leaves the band -> nothing.
            came_fast = self._net_peak >= self.net_min_speed
            net_deflect = came_fast and cosang < self.net_cos_turn and speed < pspeed
            net_stop = came_fast and speed <= self.net_speed_drop * self._net_peak
            if refr_ok and not self._net_fired and (net_deflect or net_stop):
                ev = BallEvent(frame, "net_hit", u, v, x_m, y_m, None,
                               speed_change=speed - pspeed)
                self._net_fired = True
        else:
            # CLASSIFY A DEFLECTION BY WHERE IT HAPPENS (player hits were handled above, so
            # anything here is FAR FROM A HAND). A "deflection" is a sharp change of heading
            # between consecutive measured frames -- read from the robust per-axis "fast run
            # then reverse" signals, plus a general sharp-turn for off-axis (e.g. back-wall)
            # reversals.
            vert_rev = (self._vy_sign > 0 and sy < 0 and self._peak_down_vy >= self.min_vy)
            horiz_rev = (sx != 0 and self._vx_sign != 0 and sx != self._vx_sign
                         and self._peak_vx >= self.min_vx)
            sharp_turn = (pspeed >= self.min_vx and speed >= 0.3 * pspeed
                          and cosang < self.cos_hit)
            deflect = vert_rev or horiz_rev or sharp_turn

            # is the contact ON the court floor? (the homography maps a floor point in-bounds;
            # a ball up on the glass/fence projects OUTSIDE the court). None = no homography.
            inside = (is_inside_court((x_m, y_m), self.in_out_margin_m)
                      if x_m is not None else None)

            if self.detect_bounces and refr_ok and deflect:
                if self._in_fence(u, v):
                    # deflected into the metal fence -> OUT (the round ends). in_court=False.
                    ev = BallEvent(frame, "fence_hit", u, v, x_m, y_m, in_court=False)
                elif self._in_glass(u, v):
                    # deflected in a marked GLASS panel -> WALL hit (ball stays in play).
                    ev = BallEvent(frame, "wall_bounce", u, v, x_m, y_m, None)
                elif inside is False:
                    # a deflection OUTSIDE the court floor, far from any hand: it bounced off
                    # something off the floor -> treat as a WALL hit even with no drawn region.
                    ev = BallEvent(frame, "wall_bounce", u, v, x_m, y_m, None)
                elif vert_rev and (inside is None or inside):
                    # a falling ball bounced UP while ON the court floor -> floor bounce
                    # (the in/out call comes from the court lines).
                    ev = BallEvent(frame, "floor_bounce", u, v, x_m, y_m, inside)
                # else: a deflection INSIDE the court that is not a floor bounce -> noise.

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
            # attach incoming/outgoing velocity (px/s) for downstream feature logging only.
            # player_hit's incoming is the stored APPROACH velocity (the event is pinned to
            # the contact frame); every other event diffs the consecutive measured frames.
            vin = self._pc_vel_in if ev.type == "player_hit" else prev_vel
            if vin is not None:
                ev.vx_in, ev.vy_in = float(vin[0]), float(vin[1])
            ev.vx_out, ev.vy_out = float(vx), float(vy)
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

    # PLAYER HIT (strict): ball must come CLOSE to the wrist (<= player_contact_px) AND its
    # trajectory must DEFLECT just after. wrist at (500,300); contact_px default 90.
    wr = [(500, 300, "right")]
    def wn(seq):
        return [wr] * len(seq)
    # 1) REVERSAL at the wrist -> exactly ONE hit, pinned to the contact point/frame.
    redir = [_trk(300, 300, 1500, 0),    # far (d=200) -> init
             _trk(450, 300, 1500, 0),    # d=50 near -> contact opens, approach vel stored
             _trk(500, 300, -1500, 0)]   # reversed at the wrist -> deflection -> HIT
    o = _run(redir, wn(redir))
    assert o.count("player_hit") == 1 and o[-1] == "player_hit", o
    assert _run(redir)[-1] is None, _run(redir)                      # NO wrist -> no hit
    # 2) PASS-THROUGH: near the wrist but heading unchanged -> MISS (no hit).
    passt = [_trk(300, 300, 1500, 0), _trk(460, 300, 1500, 0),
             _trk(560, 300, 1500, 0), _trk(720, 300, 1500, 0)]
    assert _run(passt, wn(passt)).count("player_hit") == 0, _run(passt, wn(passt))
    # 3) GRAB: ball slows to a stop at the wrist but never reverses -> NOT a hit.
    grab = [_trk(300, 300, 800, 0), _trk(470, 300, 400, 0),
            _trk(500, 300, 30, 0), _trk(500, 300, 10, 0)]
    assert _run(grab, wn(grab)).count("player_hit") == 0, _run(grab, wn(grab))
    # 4) NO DOUBLE-HIT: a second reversal within the lock window is ignored.
    dbl = [_trk(300, 300, 1500, 0), _trk(450, 300, 1500, 0), _trk(500, 300, -1500, 0),
           _trk(450, 300, -1500, 0), _trk(500, 300, 1500, 0)]
    assert _run(dbl, wn(dbl)).count("player_hit") == 1, _run(dbl, wn(dbl))

    # a smooth slow arc fires nothing
    smooth = [_trk(100, 100, 500, 100), _trk(130, 106, 500, 120), _trk(160, 113, 500, 140)]
    assert all(x is None for x in _run(smooth)), _run(smooth)

    # wall bounce via a drawn glass region: horizontal reversal inside it (no homography)
    glass = [[[750, 250], [850, 250], [850, 350], [750, 350]]]
    dw = BallEventDetector(homography=None, glass_regions=glass)
    dw.update(0, _trk(800, 300, 600, 0))
    evw = dw.update(1, _trk(800, 300, -600, 0))
    assert evw is not None and evw.type == "wall_bounce", evw

    # FENCE HIT (OUT): a deflection inside a drawn FENCE region -> fence_hit, in_court False
    fence = [[[750, 250], [850, 250], [850, 350], [750, 350]]]
    df = BallEventDetector(homography=None, fence_regions=fence)
    df.update(0, _trk(800, 300, 600, 0))
    evf = df.update(1, _trk(800, 300, -600, 0))
    assert evf is not None and evf.type == "fence_hit" and evf.in_court is False, evf
    # fence takes PRIORITY over glass when a point is in both
    dfp = BallEventDetector(homography=None, glass_regions=glass, fence_regions=fence)
    dfp.update(0, _trk(800, 300, 600, 0))
    evfp = dfp.update(1, _trk(800, 300, -600, 0))
    assert evfp is not None and evfp.type == "fence_hit", evfp

    # WALL HIT with NO drawn region: a sharp deflection that maps OUTSIDE the court floor.
    # A tiny homography (1 px == 1 m, court 0..10 x 0..20) -> pixel (2000,2000) is far out.
    from utils.homography import Homography as _H
    import numpy as _np
    Hm = _np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    homo = _H(Hm, _np.linalg.inv(Hm))
    dwall = BallEventDetector(homography=homo)              # ball at (2000,2000)=(2000m,2000m) -> outside
    dwall.update(0, _trk(2000, 2000, 1500, 0))
    evwall = dwall.update(1, _trk(2000, 2000, -1500, 0))    # reversal, far outside court
    assert evwall is not None and evwall.type == "wall_bounce", evwall

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
