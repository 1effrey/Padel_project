"""core/ball_physics.py
PHASE 5 -- physics-based 3D trajectory reconstruction.

The single end-camera measures the ball's image position (u, v) well, but it cannot
directly see the ball's HEIGHT (Z) or DEPTH (Y). Phase 5 recovers those by enforcing
that a flying ball obeys projectile physics and touches the floor (Z=0) at bounces.

It is built in GATED steps -- each is measured on real footage before the next:

  STEP 1  (done)  -- SEGMENTATION
      Split the per-frame ball track into free-flight SEGMENTS bounded by events
      (floor bounce / wall bounce / hit). A clean parabola only holds BETWEEN impacts
      -- at each event the velocity jumps, so every arc is fit on its own. Each segment
      also remembers its boundary events, because a FLOOR BOUNCE is a Z=0 anchor (the
      ball is on the ground there) -- those anchors are what let us solve for the height
      the camera never saw.

  STEP 2  (this file, so far)  -- BALLISTIC RECONSTRUCTION (recovers Z)
      For each segment, split it into individual free-flight ARCS at floor contacts
      (the confirmed bounces PLUS intermediate bounces Phase 3 missed, re-found from the
      2D track), then for each arc between two floor points solve the unique gravity
      parabola that leaves the floor and returns to it -- giving the HEIGHT (Z) the
      single camera cannot measure. Validated by reprojecting the recovered 3D arc
      through the Phase-4 camera and comparing to what the camera actually saw.

  STEP 3  (this file, so far)  -- PROJECTILE EKF  -- the robust reconstructor
      An Extended Kalman Filter on the 3D state [x, y, z, vx, vy, vz]. It PREDICTS with
      gravity, UPDATES from the 2D detections through the (nonlinear) camera projection,
      and FILLS gaps by predicting when no detection arrives. Floor bounces anchor Z=0
      (confirmed events = strong 3D anchor; undetected bounces = a lightweight cusp cue
      that flips vertical velocity, gated on the filter's OWN height estimate so a false
      cusp can't corrupt it). Unlike the Step-2 closed form it degrades gracefully -- it
      is driven BY the observations, so it can't invent a 199 m arc. Step 2 stays as a
      per-arc VALIDATOR; Step 3 is the per-frame RECONSTRUCTOR.

  STEP 4  (later)  -- 3D trace overlay + output

This module is a PURE CONSUMER of what `--ball-eval` already writes
(output/ball_eval.jsonl). It does NOT touch the detector, tracker, or pipeline.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# Events that BREAK a free-flight arc: at each, the ball's velocity changes
# discontinuously, so a single parabola cannot span across them.
_BREAKING_EVENTS = ("floor_bounce", "wall_bounce", "hit")


@dataclass
class TrackPoint:
    """One frame of the ball track, as the physics layer needs it.

    `measured` is the key flag: True = a REAL detection this frame (a 2D observation the
    fit can trust); False = a tracker-coasted gap (no observation -- the physics fills
    it). Both the raw detection (u, v) and the smoothed Kalman position (track_x,
    track_y) are kept so later steps can choose which to trust.
    """
    frame: int
    u: Optional[float]            # chosen 2D detection (zero-lag); None if not found
    v: Optional[float]
    measured: bool                # True = real detection; False = coasting gap frame
    confidence: float
    track_x: Optional[float]      # Kalman position (smooth); present even while coasting
    track_y: Optional[float]
    status: str                   # "tracking" | "coasting" | "lost"


@dataclass
class BoundaryEvent:
    """An impact at one end of a segment (a bounce or a hit). A floor bounce also
    carries its court position in metres -- the Z=0 floor anchor for the later fit."""
    frame: int
    type: str                     # "floor_bounce" | "wall_bounce" | "hit"
    u: float
    v: float
    x_m: Optional[float]          # court metres via homography (reliable for FLOOR pts)
    y_m: Optional[float]
    in_court: Optional[bool]


@dataclass
class FlightSegment:
    """A contiguous span of frames during which the ball is in FREE FLIGHT (no impact),
    so a single ballistic arc applies. Bounded by an event at each end -- or by the
    track's start/end where it is open (and therefore weakly constrained)."""
    start_frame: int
    end_frame: int
    points: List[TrackPoint] = field(default_factory=list)
    start_event: Optional[BoundaryEvent] = None   # None => opens at track start
    end_event: Optional[BoundaryEvent] = None     # None => closes at track end

    @property
    def n_frames(self) -> int:
        return len(self.points)

    @property
    def n_measured(self) -> int:
        """Real detections in the span -- the points an arc fit can actually use."""
        return sum(1 for p in self.points if p.measured)

    @property
    def floor_anchors(self) -> int:
        """How many ends are Z=0 floor bounces: 2 = fully pinned parabola (best case),
        1 = one end on the floor, 0 = open arc (weak -> later flagged low-confidence)."""
        return sum(1 for e in (self.start_event, self.end_event)
                   if e is not None and e.type == "floor_bounce")

    @property
    def fittable(self) -> bool:
        """A parabola needs >= 3 real observations to fit."""
        return self.n_measured >= 3


def load_track(jsonl_path: str) -> Tuple[List[TrackPoint], List[BoundaryEvent]]:
    """Read ball_eval.jsonl -> (track points, breaking events).

    One JSON record per frame; a frame may additionally carry an 'event' (the impact
    detected on it). Returns the full per-frame track and the list of breaking events.
    """
    points: List[TrackPoint] = []
    events: List[BoundaryEvent] = []
    # Read raw and decode defensively: some logs come back peppered with stray NUL bytes
    # (a mixed-encoding artefact of how the file was written). Strip NULs and decode
    # tolerantly so one bad byte-run can't sink the whole parse; skip any line that still
    # won't parse and report the count rather than crashing.
    with open(jsonl_path, "rb") as fh:
        text = fh.read().replace(b"\x00", b"").decode("utf-8", errors="replace")
    n_bad = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            n_bad += 1
            continue
        tr = rec.get("track", {}) or {}
        points.append(TrackPoint(
            frame=int(rec["frame"]),
            u=rec.get("u"), v=rec.get("v"),
            measured=bool(tr.get("measured", False)),
            confidence=float(rec.get("confidence", 0.0)),
            track_x=tr.get("x"), track_y=tr.get("y"),
            status=str(tr.get("status", "lost")),
        ))
        ev = rec.get("event")
        if ev and ev.get("type") in _BREAKING_EVENTS:
            events.append(BoundaryEvent(
                frame=int(ev["frame"]), type=str(ev["type"]),
                u=float(ev["u"]), v=float(ev["v"]),
                x_m=ev.get("x_m"), y_m=ev.get("y_m"),
                in_court=ev.get("in_court"),
            ))
    if n_bad:
        print(f"[physics] note: skipped {n_bad} unparseable line(s) in {jsonl_path}")
    return points, events


def segment_track(points: List[TrackPoint],
                  events: List[BoundaryEvent]) -> List[FlightSegment]:
    """Split the track into free-flight segments at each breaking event.

    A bounce/hit frame is the SHARED endpoint of the segment that ends on it and the one
    that starts on it -- the ball is physically AT that point, so both arcs touch it.
    The track's first/last frame with a real position open/close the outer segments.
    Every segment is returned (even tiny ones) so the quality gate can see them all;
    `FlightSegment.fittable` marks which have enough points for a later fit.
    """
    if not points:
        return []
    # Frames with an actual track position (skips warmup / lost head & tail).
    tracked = [p for p in points if p.track_x is not None]
    if not tracked:
        return []
    first_f, last_f = tracked[0].frame, tracked[-1].frame

    # Boundaries = track start + each in-range event frame + track end (sorted, unique).
    ev_by_frame = {e.frame: e for e in events if first_f <= e.frame <= last_f}
    boundaries = sorted({first_f, last_f, *ev_by_frame.keys()})

    by_frame = {p.frame: p for p in points}
    segments: List[FlightSegment] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        seg_pts = [by_frame[f] for f in range(a, b + 1) if f in by_frame]
        segments.append(FlightSegment(
            start_frame=a, end_frame=b, points=seg_pts,
            start_event=ev_by_frame.get(a),
            end_event=ev_by_frame.get(b),
        ))
    return segments


def summarize(points: List[TrackPoint], events: List[BoundaryEvent],
              segments: List[FlightSegment]) -> None:
    """Human-readable gate report: did segmentation carve the track sensibly?"""
    n_meas = sum(1 for p in points if p.measured)
    ev_counts: dict = {}
    for e in events:
        ev_counts[e.type] = ev_counts.get(e.type, 0) + 1
    print(f"[physics] track: {len(points)} frames, {n_meas} measured "
          f"({100*n_meas/max(1,len(points)):.0f}%); "
          f"events: {ev_counts or 'none'}")
    print(f"[physics] -> {len(segments)} free-flight segments "
          f"({sum(1 for s in segments if s.fittable)} fittable, >=3 measured):")
    print(f"   {'frames':>13} {'len':>4} {'meas':>5} {'opens':>13} {'closes':>13} "
          f"{'Z0-anchors':>10} {'fit?':>5}")
    for s in segments:
        opens = s.start_event.type if s.start_event else "track-start"
        closes = s.end_event.type if s.end_event else "track-end"
        flag = "" if s.n_frames <= 60 else "  <- long (events likely under-detected)"
        print(f"   {s.start_frame:>5}-{s.end_frame:<7} {s.n_frames:>4} {s.n_measured:>5} "
              f"{opens:>13} {closes:>13} {s.floor_anchors:>10} "
              f"{'yes' if s.fittable else 'no':>5}{flag}")


# --------------------------------------------------------------------------- #
# STEP 2 -- ballistic reconstruction (recover Z)
# --------------------------------------------------------------------------- #
G = 9.81                     # gravity (m/s^2). Z is UP; it is the ball's ONLY
                             # free-flight acceleration (no horizontal force, ignoring
                             # air drag in v1 -- a small effect over a padel-length arc).
_MAX_PEAK_M = 8.0            # a recovered apex above this is physically implausible
_GOOD_REPROJ_PX = 50.0       # arc whose reprojection misses the 2D track by more than
                             # this probably hides an undetected mid-arc bounce


@dataclass
class BallisticArc:
    """One free-flight arc between two floor contacts, reconstructed in 3D.

    `kind`:
      "anchored" -- bounded by two floor points (Z=0 at both ends) -> the height is
                    fully determined by gravity + flight time. Z is RECOVERED.
      "open"     -- only one (or zero) floor anchor -> the arc is under-constrained for
                    a single camera; Z is NOT recovered in v1 (flagged, not faked).
    """
    start_frame: int
    end_frame: int
    anchor_start_xy: Optional[Tuple[float, float]]
    anchor_end_xy: Optional[Tuple[float, float]]
    peak_height_m: Optional[float]
    points_3d: List[Tuple[int, float, float, float]] = field(default_factory=list)
    reproj_px: Optional[float] = None
    kind: str = "anchored"


def _image_to_court(u: float, v: float, H: np.ndarray) -> Tuple[float, float]:
    """Map a FLOOR image point (px) to court coordinates (m) via the homography.
    Valid ONLY for points actually on the floor (e.g. a bounce) -- airborne points
    would be wrong, which is exactly why we anchor on bounces."""
    p = H @ np.array([float(u), float(v), 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def _floor_contact_frames(seg: FlightSegment, H: np.ndarray) -> List[int]:
    """Frames in a segment where the ball is KNOWN to be on the floor.

    v1: only the CONFIRMED bounce events at the segment's ends -- those have valid
    homography floor positions. We deliberately do NOT guess intermediate bounces from
    the 2D track: an image-v peak is not reliably a floor contact (it conflates height
    with depth), and anchoring an AIRBORNE point with the floor homography produces a
    garbage position. So a multi-arc span here stays one arc and will FAIL its
    reprojection check -- an honest signal that event detection (not this fit) is the
    bottleneck, rather than a fabricated wrong answer. Reliable sub-arc splitting is a
    later improvement (better Phase-3 events, or a height-aware bounce detector)."""
    contacts = set()
    if seg.start_event and seg.start_event.type == "floor_bounce":
        contacts.add(seg.start_frame)
    if seg.end_event and seg.end_event.type == "floor_bounce":
        contacts.add(seg.end_frame)
    return sorted(contacts)


def _build_arc(a_frame: int, b_frame: int, A_xy: Tuple[float, float],
               B_xy: Tuple[float, float], fps: float
               ) -> Optional[Tuple[List[Tuple[int, float, float, float]], float]]:
    """Closed-form 3D for ONE arc between floor points A (z=0) and B (z=0).

    Horizontal (x, y) moves at CONSTANT velocity -> linear from A to B (no horizontal
    force). The HEIGHT is the unique parabola that leaves z=0 and returns to z=0 over the
    flight time T: z(t) = vz0*t - 1/2 g t^2 with vz0 = 1/2 g T (so z(T)=0). The apex is
    g*T^2/8. This recovered z is exactly what the camera could not see."""
    T = (b_frame - a_frame) / fps
    if T <= 0:
        return None
    vz0 = 0.5 * G * T
    peak = vz0 * vz0 / (2.0 * G)
    pts: List[Tuple[int, float, float, float]] = []
    for f in range(a_frame, b_frame + 1):
        tau = (f - a_frame) / fps
        s = tau / T
        x = A_xy[0] + (B_xy[0] - A_xy[0]) * s
        y = A_xy[1] + (B_xy[1] - A_xy[1]) * s
        z = vz0 * tau - 0.5 * G * tau * tau
        pts.append((f, x, y, max(0.0, z)))
    return pts, peak


def _reproj_error(pts3d: List[Tuple[int, float, float, float]],
                  by_frame: Dict[int, TrackPoint], camera) -> Optional[float]:
    """Median pixel gap between the recovered 3D arc reprojected through the camera and
    the OBSERVED 2D detections -- the validation that the height we solved for is
    consistent with what the camera really saw. None if no camera / no observations."""
    if camera is None:
        return None
    errs = []
    for (f, x, y, z) in pts3d:
        p = by_frame.get(f)
        if p is None or not p.measured or p.u is None or p.v is None:
            continue
        uv = camera.project(np.array([x, y, z], dtype=float))
        errs.append(float(np.hypot(uv[0] - p.u, uv[1] - p.v)))
    return float(np.median(errs)) if errs else None


def reconstruct_segment(seg: FlightSegment, fps: float, H: np.ndarray,
                        camera) -> List[BallisticArc]:
    """Split a segment at its floor contacts and reconstruct each resulting arc."""
    by_frame = {p.frame: p for p in seg.points}
    contacts = _floor_contact_frames(seg, H)
    if len(contacts) < 2:
        # one or zero floor anchors -> can't pin the parabola for a single camera (v1).
        return [BallisticArc(seg.start_frame, seg.end_frame, None, None, None,
                             kind="open")]

    def contact_xy(fr: int) -> Tuple[float, float]:
        # Prefer a confirmed event's homography court position; else map the track point.
        for ev in (seg.start_event, seg.end_event):
            if ev and ev.frame == fr and ev.x_m is not None and ev.y_m is not None:
                return (ev.x_m, ev.y_m)
        p = by_frame[fr]
        return _image_to_court(p.track_x, p.track_y, H)

    arcs: List[BallisticArc] = []
    for a, b in zip(contacts[:-1], contacts[1:]):
        A_xy, B_xy = contact_xy(a), contact_xy(b)
        built = _build_arc(a, b, A_xy, B_xy, fps)
        if built is None:
            continue
        pts3d, peak = built
        arcs.append(BallisticArc(a, b, A_xy, B_xy, peak, pts3d,
                                 _reproj_error(pts3d, by_frame, camera), "anchored"))
    return arcs


def reconstruct_3d(jsonl_path: str, config: Dict[str, Any],
                   fps: Optional[float] = None) -> Tuple[List[BallisticArc],
                                                         List[FlightSegment]]:
    """Top-level Step 2: load track -> segment -> reconstruct every arc in 3D."""
    pts, evs = load_track(jsonl_path)
    segs = segment_track(pts, evs)
    H = np.array(config["homography"]["H"], dtype=float)
    if fps is None:
        fps = float(config.get("display", {}).get("playback_fps", 20.0)) or 20.0
    camera = None
    try:
        from core.camera_calib import build_camera
        camera, _ = build_camera(config)
    except Exception as exc:                     # noqa: BLE001 -- degrade, don't crash
        print(f"[physics] camera unavailable ({exc}); recovering Z from anchors only "
              f"(no reprojection validation).")
    arcs: List[BallisticArc] = []
    for s in segs:
        arcs.extend(reconstruct_segment(s, fps, H, camera))
    return arcs, segs


def write_trajectory(arcs: List[BallisticArc], path: str) -> int:
    """Write the recovered 3D trajectory (one row per frame). Returns rows written."""
    by_frame: Dict[int, Dict[str, Any]] = {}
    for a in arcs:
        for (f, x, y, z) in a.points_3d:
            by_frame[f] = {"frame": f, "x_m": round(x, 4), "y_m": round(y, 4),
                           "z_m": round(z, 4), "arc": f"{a.start_frame}-{a.end_frame}"}
    with open(path, "w", encoding="utf-8") as fh:
        for f in sorted(by_frame):
            fh.write(json.dumps(by_frame[f]) + "\n")
    return len(by_frame)


def report_arcs(arcs: List[BallisticArc]) -> None:
    """Step 2 gate report: did we recover plausible heights that match the 2D track?"""
    anchored = [a for a in arcs if a.kind == "anchored"]
    print(f"[physics] STEP 2: {len(arcs)} arcs "
          f"({len(anchored)} anchored -> Z recovered, "
          f"{len(arcs) - len(anchored)} open -> Z deferred):")
    print(f"   {'frames':>13} {'kind':>9} {'peakZ_m':>8} {'reproj_px':>10} {'verdict':>10}")
    zs = []
    for a in arcs:
        peak = f"{a.peak_height_m:.2f}" if a.peak_height_m is not None else "-"
        rp = f"{a.reproj_px:.0f}" if a.reproj_px is not None else "-"
        if a.kind == "open":
            verdict = "open"
        elif a.peak_height_m is not None and a.peak_height_m > _MAX_PEAK_M:
            verdict = "HIGH?"
        elif a.reproj_px is not None and a.reproj_px > _GOOD_REPROJ_PX:
            verdict = "mismatch"
        else:
            verdict = "ok"
            zs.append(a.peak_height_m)
        print(f"   {a.start_frame:>5}-{a.end_frame:<7} {a.kind:>9} {peak:>8} "
              f"{rp:>10} {verdict:>10}")
    if zs:
        print(f"[physics] recovered apex heights (ok arcs): min {min(zs):.2f}m  "
              f"median {float(np.median(zs)):.2f}m  max {max(zs):.2f}m")


# --------------------------------------------------------------------------- #
# STEP 3 -- projectile EKF (the robust per-frame reconstructor)
# --------------------------------------------------------------------------- #
def _cusp_frames(points: List[TrackPoint], prominence_px: float = 80.0,
                 min_gap: int = 5) -> set:
    """Lightweight bounce CUES: prominent local maxima of the ball's image-v (the ball
    sits lowest in the frame at a floor contact). These are only HINTS -- the EKF accepts
    one as a real floor bounce only when its own height estimate is already low, so a
    false cue (an airborne apex) is ignored."""
    from scipy.signal import find_peaks
    fr = [p.frame for p in points if p.track_y is not None]
    vy = np.array([p.track_y for p in points if p.track_y is not None], dtype=float)
    if len(vy) < 5:
        return set()
    peaks, _ = find_peaks(vy, prominence=prominence_px, distance=min_gap)
    return {fr[int(i)] for i in peaks}


class ProjectileEKF:
    """EKF on the 3D state s = [x, y, z, vx, vy, vz] (court metres).

    PREDICT: constant velocity in x, y; gravity on z (the model is linear -> a plain KF
    predict). UPDATE: the measurement is the 2D image point, a NONLINEAR projection of
    (x, y, z) -- hence 'extended': we linearise the projection with a numeric Jacobian.
    """

    def __init__(self, camera, H: np.ndarray, fps: float, meas_px: float = 15.0,
                 q_pos: float = 0.02, q_vel: float = 6.0, q_pos_z: float = 0.01,
                 q_vel_z: float = 0.5, restitution: float = 0.7,
                 bounce_z_thresh: float = 0.7, z_max: float = 8.0) -> None:
        self.cam = camera
        self.H = H
        self.dt = 1.0 / fps
        self.restitution = restitution
        self.bounce_z = bounce_z_thresh
        self.z_max = z_max
        dt = self.dt
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = dt                      # x += v*dt
        self.u = np.zeros(6)                           # gravity control input
        self.u[2] = -0.5 * G * dt * dt                 # z -= 1/2 g dt^2
        self.u[5] = -G * dt                            # vz -= g dt
        # ANISOTROPIC process noise: horizontal velocity may change (hits), but VERTICAL
        # velocity obeys gravity tightly between bounces (only a bounce changes it, and
        # those are handled explicitly) -- this is what stops z drifting up the ray.
        self.Q = np.diag([q_pos, q_pos, q_pos_z, q_vel, q_vel, q_vel_z])
        self.R = np.diag([meas_px ** 2, meas_px ** 2])
        self.s: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None

    def init(self, xy: Tuple[float, float], vxy: Tuple[float, float] = (0.0, 0.0)) -> None:
        self.s = np.array([xy[0], xy[1], 0.0, vxy[0], vxy[1], 0.0], dtype=float)
        # uncertain to start -- especially velocity and the unseen height
        self.P = np.diag([1.0, 1.0, 1.0, 36.0, 36.0, 36.0])

    def predict(self) -> None:
        self.s = self.F @ self.s + self.u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def _proj_jac(self, p: np.ndarray) -> np.ndarray:
        """Numeric 2x3 Jacobian of the camera projection at world point p."""
        J = np.zeros((2, 3))
        eps = 1e-3
        for i in range(3):
            dp = np.zeros(3)
            dp[i] = eps
            J[:, i] = (self.cam.project(p + dp) - self.cam.project(p - dp)) / (2 * eps)
        return J

    def update_2d(self, u: float, v: float, conf: float = 1.0) -> None:
        """EKF correction from a 2D detection."""
        p = self.s[:3]
        h = self.cam.project(p)
        if not np.all(np.isfinite(h)):
            return                                     # point behind camera -> skip
        Hj = np.zeros((2, 6))
        Hj[:, :3] = self._proj_jac(p)
        R = self.R / max(float(conf), 0.05)            # trust confident detections more
        y = np.array([u, v]) - h
        S = Hj @ self.P @ Hj.T + R
        K = self.P @ Hj.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (np.eye(6) - K @ Hj) @ self.P

    def anchor_floor(self, x_m: float, y_m: float) -> None:
        """Strong 3D correction at a CONFIRMED bounce: the ball is at (x, y, 0). This is
        the measurement that makes the unseen height observable."""
        Hj = np.zeros((3, 6))
        Hj[0, 0] = Hj[1, 1] = Hj[2, 2] = 1.0
        R = np.diag([0.05, 0.05, 0.02])                # tight, in metres
        y = np.array([x_m, y_m, 0.0]) - self.s[:3]
        S = Hj @ self.P @ Hj.T + R
        K = self.P @ Hj.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (np.eye(6) - K @ Hj) @ self.P

    def bounce_flip(self) -> None:
        """A floor bounce reverses vertical velocity (the ball must head back up)."""
        self.s[5] = abs(self.restitution * self.s[5])

    def constrain_floor(self) -> None:
        """Physical prior: a padel ball stays between the floor and ~8 m. When the state
        leaves that band (an unobserved-depth drift), pull it back with a soft pseudo-
        measurement. Cheap, principled, and keeps Z sane between sparse anchors."""
        z = self.s[2]
        target = 0.0 if z < 0.0 else (self.z_max if z > self.z_max else None)
        if target is None:
            return
        Hj = np.zeros((1, 6))
        Hj[0, 2] = 1.0
        R = np.array([[0.05]])
        y = np.array([target - z])
        S = Hj @ self.P @ Hj.T + R
        K = self.P @ Hj.T @ np.linalg.inv(S)
        self.s = self.s + (K @ y).ravel()
        self.P = (np.eye(6) - K @ Hj) @ self.P

    def update_3d(self, point3d: Tuple[float, float, float],
                  sigma3: Tuple[float, float, float]) -> None:
        """Strong correction from a TRIANGULATED 3D point (BOTH cameras saw the ball).
        This is what makes depth/height OBSERVABLE -- the second camera resolves the
        along-the-ray ambiguity a single camera cannot. Uses the triangulation's
        anisotropic std as measurement noise, so the weakest axis (Y) is trusted least."""
        Hj = np.zeros((3, 6))
        Hj[0, 0] = Hj[1, 1] = Hj[2, 2] = 1.0
        s = [max(float(v), 0.05) for v in sigma3]
        R = np.diag([s[0] ** 2, s[1] ** 2, s[2] ** 2])
        y = np.asarray(point3d, dtype=float) - self.s[:3]
        S = Hj @ self.P @ Hj.T + R
        K = self.P @ Hj.T @ np.linalg.inv(S)
        self.s = self.s + K @ y
        self.P = (np.eye(6) - K @ Hj) @ self.P

    def state(self) -> Tuple[float, float, float, float, float, float]:
        return tuple(float(x) for x in self.s)


def _load_fusion(path: str
                 ) -> Dict[int, Tuple[float, float, float, Tuple[float, float, float]]]:
    """Load Phase-4 fusion output -> {side-A frame: (x, y, z, std3)} for the frames where
    the ball was TRIANGULATED from both cameras. Robust to NUL-corrupted logs."""
    out: Dict[int, Tuple[float, float, float, Tuple[float, float, float]]] = {}
    try:
        with open(path, "rb") as fh:
            text = fh.read().replace(b"\x00", b"").decode("utf-8", errors="replace")
    except FileNotFoundError:
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        b = rec.get("ball3d")
        if b:
            std = b.get("std", [0.3, 0.3, 0.3])
            out[int(rec["frame"])] = (float(b["x"]), float(b["y"]), float(b["z"]),
                                      (float(std[0]), float(std[1]), float(std[2])))
    return out


def _rts_smooth(fwd: List[Dict[str, Any]], F: np.ndarray, z_max: float = 8.0
                ) -> List[Tuple[int, float, float, float]]:
    """Rauch-Tung-Striebel backward pass: re-estimate every frame using the FUTURE
    anchors too. This is what tames the height -- between two sparse anchors the forward
    filter EXTRAPOLATES (and drifts), but the smoother INTERPOLATES, pinning each frame
    from both sides. The chain is broken at bounce frames: a bounce is a velocity cusp,
    so we never smooth a falling arc into the rising one across it."""
    n = len(fwd)
    if n == 0:
        return []
    s_sm = [r["s_post"].copy() for r in fwd]
    P_sm = [r["P_post"].copy() for r in fwd]
    for k in range(n - 2, -1, -1):
        if fwd[k + 1]["brk"]:                              # bounce at k+1 -> don't cross it
            continue
        Pp = fwd[k + 1]["P_prior"]
        try:
            C = fwd[k]["P_post"] @ F.T @ np.linalg.inv(Pp)
        except np.linalg.LinAlgError:
            continue
        s_sm[k] = fwd[k]["s_post"] + C @ (s_sm[k + 1] - fwd[k + 1]["s_prior"])
        P_sm[k] = fwd[k]["P_post"] + C @ (P_sm[k + 1] - Pp) @ C.T
    # Final physical clamp: the ball lives in [floor, z_max]. The smoother can overshoot
    # the band in unanchored stretches (depth is unobserved there); bound it to plausible.
    return [(fwd[k]["frame"], float(s_sm[k][0]), float(s_sm[k][1]),
             float(min(z_max, max(0.0, s_sm[k][2])))) for k in range(n)]


def _height_stats(traj: List[Tuple[int, float, float, float]]) -> Tuple[float, ...]:
    z = np.array([t[3] for t in traj]) if traj else np.array([0.0])
    return (float(z.min()), float(np.median(z)), float(z.max()),
            100.0 * float((z < -0.05).mean()), 100.0 * float((z > 6.0).mean()))


def run_ekf(jsonl_path: str, config: Dict[str, Any], fps: Optional[float] = None,
            fusion_path: Optional[str] = None, smooth: bool = True,
            exclude: Optional[set] = None
            ) -> Tuple[List[Tuple[int, float, float, float]], Any]:
    """Step 3 / 3b / 3c: reconstruct the dense 3D trajectory with the projectile EKF, then
    (Step 3c) an RTS smoother that uses future anchors to interpolate the height.

    `fusion_path` (Phase-4 ball_fusion.jsonl) injects TRIANGULATED 3D points as depth
    anchors. `smooth` adds the backward smoothing pass. Returns (trajectory, camera)."""
    pts, evs = load_track(jsonl_path)
    H = np.array(config["homography"]["H"], dtype=float)
    if fps is None:
        fps = float(config.get("display", {}).get("playback_fps", 20.0)) or 20.0
    from core.camera_calib import build_camera
    cam, _ = build_camera(config)

    ekf = ProjectileEKF(cam, H, fps)
    ev_frames = {e.frame: e for e in evs if e.type == "floor_bounce"}
    cusps = _cusp_frames(pts)
    fusion = _load_fusion(fusion_path) if fusion_path else {}
    exclude = exclude or set()                             # anchors to hold out (validation)
    n_tri = 0

    fwd: List[Dict[str, Any]] = []                         # per-frame records for the RTS pass
    traj_fwd: List[Tuple[int, float, float, float]] = []
    inited = False
    for p in pts:
        if not inited:
            if p.frame in fusion and p.frame not in exclude:   # best: init from a 3D point
                x, y, z, _ = fusion[p.frame]
                ekf.init((x, y))
                ekf.s[2] = z
            elif p.measured and p.u is not None:
                ekf.init(_image_to_court(p.u, p.v, H))
            else:
                continue
            inited = True
            fwd.append({"frame": p.frame, "s_prior": ekf.s.copy(), "P_prior": ekf.P.copy(),
                        "s_post": ekf.s.copy(), "P_post": ekf.P.copy(), "brk": False})
            traj_fwd.append((p.frame, *ekf.state()[:3]))
            continue
        ekf.predict()
        s_prior, P_prior = ekf.s.copy(), ekf.P.copy()      # the F-prediction (for RTS gain)
        if p.measured and p.u is not None:
            ekf.update_2d(p.u, p.v, p.confidence)
        if p.frame in fusion and p.frame not in exclude:   # triangulated 3D -> depth anchor
            x, y, z, std = fusion[p.frame]
            ekf.update_3d((x, y, z), std)
            n_tri += 1
        is_bounce = False
        if p.frame in ev_frames:                           # confirmed bounce
            e = ev_frames[p.frame]
            if e.x_m is not None and e.y_m is not None:
                ekf.anchor_floor(e.x_m, e.y_m)
            ekf.bounce_flip()
            is_bounce = True
        elif p.frame in cusps and ekf.s[2] < ekf.bounce_z:   # soft cue, height-gated
            # A low cusp is almost certainly an undetected floor bounce -> anchor z=0 at
            # the floor position (homography is valid here, the ball is on the ground),
            # then flip vertical velocity. These extra z=0 anchors shorten the unanchored
            # spans and tighten the height between sparse triangulations.
            if p.track_x is not None and p.track_y is not None:
                cx, cy = _image_to_court(p.track_x, p.track_y, H)
                ekf.anchor_floor(cx, cy)
            ekf.bounce_flip()
            is_bounce = True
        ekf.constrain_floor()                              # ball never below the floor
        fwd.append({"frame": p.frame, "s_prior": s_prior, "P_prior": P_prior,
                    "s_post": ekf.s.copy(), "P_post": ekf.P.copy(), "brk": is_bounce})
        traj_fwd.append((p.frame, *ekf.state()[:3]))

    print(f"[physics]   depth anchors: {n_tri} triangulated 3D points"
          f"{'  (NONE found -> single-camera mode, height weak)' if not fusion else ''}")
    if not smooth:
        return traj_fwd, cam
    traj_sm = _rts_smooth(fwd, ekf.F, ekf.z_max)
    fz, sz = _height_stats(traj_fwd), _height_stats(traj_sm)
    print(f"[physics]   height Z  forward  : min {fz[0]:6.1f}  med {fz[1]:5.2f}  "
          f"max {fz[2]:6.1f}   (<0 {fz[3]:.0f}%, >6m {fz[4]:.0f}%)")
    print(f"[physics]   height Z  SMOOTHED : min {sz[0]:6.1f}  med {sz[1]:5.2f}  "
          f"max {sz[2]:6.1f}   (<0 {sz[3]:.0f}%, >6m {sz[4]:.0f}%)  <- RTS")
    return traj_sm, cam


def validate_height(jsonl_path: str, config: Dict[str, Any], fusion_path: str,
                    fps: Optional[float] = None, holdout_frac: float = 0.3,
                    seed: int = 0) -> None:
    """MEASURE height accuracy. The triangulated anchors are our only 3D ground truth, so:
    hold out a fraction of them, reconstruct the trajectory WITHOUT them, then compare the
    recovered height at those frames to their true triangulated height. This scores how
    well the filter+smoother INTERPOLATES height between anchors -- the thing we're tuning."""
    import random
    fusion = _load_fusion(fusion_path)
    frames = sorted(fusion.keys())
    if len(frames) < 6:
        print(f"[height-val] only {len(frames)} triangulated anchors -- too few to score.")
        return
    rng = random.Random(seed)
    holdout = set(rng.sample(frames, max(1, int(len(frames) * holdout_frac))))
    traj, _ = run_ekf(jsonl_path, config, fps=fps, fusion_path=fusion_path, exclude=holdout)
    zby = {f: z for (f, _x, _y, z) in traj}
    errs = [abs(zby[f] - fusion[f][2]) for f in holdout if f in zby]
    if errs:
        e = np.array(errs)
        print(f"[height-val] held out {len(holdout)}/{len(frames)} triangulated anchors -> "
              f"recovered-height error there: median {np.median(e):.2f}m  mean {e.mean():.2f}m"
              f"  p90 {np.percentile(e, 90):.2f}m")


def report_ekf(traj: List[Tuple[int, float, float, float]], pts: List[TrackPoint],
               cam) -> None:
    """Step 3 gate: does the EKF trajectory match the 2D track (reproject low) while
    producing a plausible, gap-filled HEIGHT the camera never saw?"""
    by_frame = {p.frame: p for p in pts}
    zs = [z for (_, _, _, z) in traj]
    errs = []
    for (f, x, y, z) in traj:
        p = by_frame.get(f)
        if p and p.measured and p.u is not None:
            uv = cam.project(np.array([x, y, z], dtype=float))
            if np.all(np.isfinite(uv)):
                errs.append(float(np.hypot(uv[0] - p.u, uv[1] - p.v)))
    measured = sum(1 for p in pts if p.measured)
    print(f"[physics] STEP 3 EKF: {len(traj)} frames reconstructed "
          f"({measured} driven by a detection, {len(traj) - measured} gap-filled).")
    if errs:
        e = np.array(errs)
        print(f"[physics]   reprojection vs 2D track: median {np.median(e):.0f}px  "
              f"mean {e.mean():.0f}px  p90 {np.percentile(e,90):.0f}px   "
              f"(Step-2 closed form was 100-13000px)")
    if zs:
        z = np.array(zs)
        print(f"[physics]   recovered height Z: min {z.min():.2f}m  median "
              f"{np.median(z):.2f}m  max {z.max():.2f}m   "
              f"(<0: {100*(z<-0.05).mean():.0f}%  >6m: {100*(z>6).mean():.0f}%)")


# --------------------------------------------------------------------------- #
# Gate / self-test:  python -m core.ball_physics [ball_eval.jsonl] [config-side1.json]
# Runs Step 1 (segmentation) + Step 3 (EKF reconstruction) on real footage. Step 2's
# closed form stays available as a per-arc validator (reconstruct_3d / report_arcs).
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    jsonl = sys.argv[1] if len(sys.argv) > 1 else "output/ball_eval.jsonl"
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config-side1.json"
    print(f"[physics] gate on {jsonl} (camera/homography from {cfg_path})")

    with open(cfg_path, "r", encoding="utf-8") as fh:
        config = json.load(fh)

    import os

    pts, evs = load_track(jsonl)
    segs = segment_track(pts, evs)
    summarize(pts, evs, segs)                          # Step 1
    print()
    fusion_path = "output/ball_fusion.jsonl"           # Phase-4 triangulated 3D anchors
    have_fusion = os.path.exists(fusion_path)
    print(f"[physics] {'Step 3b: dual-camera (fusion anchors ON)' if have_fusion else 'Step 3: single-camera (no fusion log found)'}")
    traj, cam = run_ekf(jsonl, config,
                        fusion_path=fusion_path if have_fusion else None)
    report_ekf(traj, pts, cam)
    n = write_trajectory(
        [BallisticArc(traj[0][0], traj[-1][0], None, None, None, points_3d=traj)],
        "output/ball_trajectory_3d.jsonl")
    print(f"[physics] wrote {n} frames -> output/ball_trajectory_3d.jsonl")
