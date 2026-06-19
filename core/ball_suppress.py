"""core/ball_suppress.py
STATIONARY-BALL SUPPRESSION -- stop the tracker from locking onto a PARKED ball.

The case audit showed the tracked "active ball" sits nearly still most of the time
(side-1 median speed 42 px/s): the Kalman tracker keeps getting captured by a stationary
yellow blob (an abandoned ball, a court mark, the yellow chairs) instead of following the
moving rally ball. This module sits BETWEEN the detector and the tracker and removes those
parked-blob candidates, so only real (moving / fresh) candidates reach the Kalman filter.

DESIGNED FOR A LIVE STREAM (Jetson), so everything is:
  * CAUSAL    -- past frames only; no look-ahead. A blob is judged "parked" by how long it
                 has ALREADY sat still.
  * BOUNDED   -- keeps only a small rolling set of blob-tracks + the active parked zones;
                 stale ones are retired every frame, so it runs for hours without growing.
  * CHEAP     -- a few distance checks per candidate per frame (negligible vs detection).

HOW IT STAYS CORRECT
  * Warm-up: a blob must sit still for `min_frames` (a streak of small steps) before it is
    latched as PARKED -- so a ball that briefly pauses is not suppressed.
  * Release-on-motion: once the parked blob drifts more than `release_disp_px` from where it
    parked (it was picked up / served), the zone is released immediately -- a served ball is
    never suppressed for more than the moment it starts moving.
  * Release-on-vanish: the internal blob-track is updated on the RAW candidates (even the
    ones we hide from the tracker), so if the parked ball disappears its track simply
    retires and the zone clears.
  * TIGHT suppress radius: only candidates within `suppress_radius_px` of a parked CENTRE
    are dropped, so a real ball flying past the parked spot loses at most a frame or two
    (the Kalman coasts through), not a whole region.

USE
    sup = StationaryBallSuppressor.from_config(config)      # once, per camera
    ...
    det.detect(frame)
    cands = sup.filter(frame_idx, det.last_candidates)      # drop parked-blob candidates
    track = trk.update_multi(cands)
"""
from __future__ import annotations

from collections import deque
from math import hypot
from typing import Any, Dict, List, Tuple


class _Blob:
    """One rolling candidate cluster, linked across frames. Tracks a 'still streak' so we
    can latch it PARKED after it has sat still long enough, and a latched centre so a small
    drift afterwards doesn't move the suppression spot."""

    __slots__ = ("u", "v", "last", "streak", "parked", "cx", "cy", "recent")

    def __init__(self, frame: int, u: float, v: float):
        self.u, self.v, self.last = u, v, frame
        self.streak = 0                 # consecutive "still" frames
        self.parked = False             # latched parked state
        self.cx = self.cy = 0.0         # latched parked centre (set on promotion)
        self.recent: deque = deque([(u, v)], maxlen=5)

    def add(self, frame: int, u: float, v: float, still_step: float) -> None:
        step = hypot(u - self.u, v - self.v)
        self.u, self.v, self.last = u, v, frame
        self.recent.append((u, v))
        self.streak = self.streak + 1 if step <= still_step else 0

    def drift(self) -> float:
        """Distance from the latched parked centre (how far it has moved since parking)."""
        return hypot(self.u - self.cx, self.v - self.cy)


class StationaryBallSuppressor:
    """Removes parked-ball candidates before they reach the Kalman tracker."""

    def __init__(self, enabled: bool = True, link_radius_px: float = 90.0,
                 drop_after: int = 10, min_frames: int = 40, still_step_px: float = 10.0,
                 release_disp_px: float = 60.0, suppress_radius_px: float = 14.0,
                 max_zones: int = 16) -> None:
        self.enabled = bool(enabled)
        self.link_radius = float(link_radius_px)
        self.drop_after = int(drop_after)
        self.min_frames = int(min_frames)         # still frames before a blob is "parked"
        self.still_step = float(still_step_px)     # per-frame move under this == "still"
        self.release_disp = float(release_disp_px)  # drift from centre that releases a zone
        self.suppress_radius = float(suppress_radius_px)  # tight kill radius around a centre
        self.max_zones = int(max_zones)
        self._blobs: List[_Blob] = []
        self.last_suppressed = 0                   # #candidates dropped on the last frame

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "StationaryBallSuppressor":
        s = config.get("ball", {}).get("suppress", {})
        return cls(
            enabled=s.get("enabled", True),
            link_radius_px=s.get("link_radius_px", 90.0),
            drop_after=s.get("drop_after", 10),
            min_frames=s.get("min_frames", 40),
            still_step_px=s.get("still_step_px", 10.0),
            release_disp_px=s.get("release_disp_px", 60.0),
            suppress_radius_px=s.get("suppress_radius_px", 14.0),
            max_zones=s.get("max_zones", 16),
        )

    def reset(self) -> None:
        """Drop all state (call between independent clips)."""
        self._blobs = []
        self.last_suppressed = 0

    @property
    def parked_centers(self) -> List[Tuple[float, float]]:
        """Active parked-zone centres (for drawing / debugging)."""
        return [(b.cx, b.cy) for b in self._blobs if b.parked]

    def filter(self, frame_idx: int, candidates: List[Any]) -> List[Any]:
        """Return the candidates with parked-blob ones removed.

        NOTE the internal blob bookkeeping runs on the RAW candidates (including the parked
        one) so we can keep watching the parked blob and release it when it moves/vanishes;
        only the RETURNED list is filtered."""
        if not self.enabled:
            return list(candidates)

        # 1) retire stale blob-tracks (bounded memory) ------------------------------------
        self._blobs = [b for b in self._blobs if frame_idx - b.last <= self.drop_after]

        # 2) link each raw candidate to its nearest blob-track (or start a new one) --------
        for c in candidates:
            best, bestd = None, self.link_radius ** 2
            for b in self._blobs:
                d = (b.u - c.u) ** 2 + (b.v - c.v) ** 2
                if d <= bestd:
                    best, bestd = b, d
            if best is None:
                self._blobs.append(_Blob(frame_idx, c.u, c.v))
            else:
                best.add(frame_idx, c.u, c.v, self.still_step)

        # 3) promote (still long enough) / release (moved away) ---------------------------
        for b in self._blobs:
            if not b.parked and b.streak >= self.min_frames:
                b.parked = True
                b.cx, b.cy = b.u, b.v                    # latch the centre
            elif b.parked and b.drift() >= self.release_disp:
                b.parked = False                         # it moved -> live again

        # bound the number of active zones (keep the most recently updated)
        parked = [b for b in self._blobs if b.parked]
        if len(parked) > self.max_zones:
            parked.sort(key=lambda b: b.last, reverse=True)
            for b in parked[self.max_zones:]:
                b.parked = False

        # 4) drop candidates sitting inside a parked zone (tight radius) -------------------
        centers = [(b.cx, b.cy) for b in self._blobs if b.parked]
        if not centers:
            self.last_suppressed = 0
            return list(candidates)

        r2 = self.suppress_radius ** 2
        out, dropped = [], 0
        for c in candidates:
            if any((cx - c.u) ** 2 + (cy - c.v) ** 2 <= r2 for cx, cy in centers):
                dropped += 1
            else:
                out.append(c)
        self.last_suppressed = dropped
        return out


# --------------------------------------------------------------------------- #
# Smoke test: a PARKED ball sitting at (500,500) + a real ball flying across.
#   python -m core.ball_suppress
# Expect: after warm-up the parked candidate is dropped every frame, the moving
# ball always passes, and when the parked ball is "served" it is released.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    class C:                                    # minimal stand-in for BallDetection
        def __init__(self, u, v):
            self.u, self.v, self.found, self.confidence = u, v, True, 0.9

    sup = StationaryBallSuppressor(min_frames=10, suppress_radius_px=14,
                                   release_disp_px=60, still_step_px=8)
    print("frame  in -> out   parked_zones   note")
    for f in range(40):
        cands = [C(500.0, 500.0)]               # the PARKED ball, always present
        if f >= 5:                              # a real ball flying left-to-right
            cands.append(C(100.0 + 25 * f, 300.0))
        if f >= 30:                             # the parked ball gets SERVED at f=30
            cands[0] = C(500.0 + 70 * (f - 29), 500.0 - 40 * (f - 29))
        out = sup.filter(f, cands)
        note = ""
        if f == 10:
            note = "<- parked ball now suppressed"
        if f >= 30 and not any(b.parked for b in sup._blobs):
            note = "<- served: zone released"
        print(f"{f:>4}   {len(cands)} -> {len(out)}    "
              f"{len(sup.parked_centers)}            {note}")
