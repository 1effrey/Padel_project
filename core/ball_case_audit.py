"""core/ball_case_audit.py
MEASUREMENT pass -- count how often the three "active-ball arbitration" failure cases
actually occur in real footage, BEFORE building any arbitration logic.
("Measure failures, don't hide them." -- architecture decision #5.)

It reuses the SAME detector + tracker + events the pipeline uses (no new model), then
links the detector's per-frame candidate blobs into short TRACKLETS so we can tell a
MOVING ball from a PARKED one:

  Case 1 -- SWAP risk        : >=2 ball-like tracklets are MOVING in the same frame
                               (the rally ball + a second moving ball).
  Case 2 -- PARKED ball      : a ball-like tracklet sits ~still for many frames (a
                               stationary abandoned ball) -- counted as episodes, and
                               separately the frames where it overlaps a MOVING active
                               track (the actually-dangerous overlap).
  Case 3 -- SLOW in-play ball: the tracked active ball is measured + "tracking" but its
                               speed dips low (lob apex / soft drop), away from a
                               hit/bounce. Reported as a speed distribution + counts
                               under several thresholds (so we don't hard-code "slow").

A candidate is optionally passed through the YELLOW gate (same as ball_dual) so we count
ball-LIKE blobs, not lights/limbs. Everything is a proxy -- the point is the ORDER OF
MAGNITUDE (rare vs common), to decide whether the arbitration layer is worth building.

RUN
    python -m core.ball_case_audit config-side1.json [max_frames]
"""
from __future__ import annotations

import json
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.ball_dual import _is_yellow
from core.ball_eval import _build_detector, _build_events, _build_tracker
from utils.homography import Homography
from utils.metrics import NumpyEncoder
from utils.video_io import ThreadedVideoReader

_FONT = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------- #
# Lightweight candidate tracklet: links a blob across frames so we can ask
# "is this thing MOVING, or PARKED?"  (separate from the Kalman ball tracker).
# --------------------------------------------------------------------------- #
class _Tracklet:
    __slots__ = ("u", "v", "first", "last", "umin", "umax", "vmin", "vmax", "recent")

    def __init__(self, frame: int, u: float, v: float):
        self.u, self.v = u, v
        self.first = self.last = frame
        self.umin = self.umax = u
        self.vmin = self.vmax = v
        self.recent: deque = deque([(u, v)], maxlen=5)   # last few positions

    def add(self, frame: int, u: float, v: float) -> None:
        self.u, self.v, self.last = u, v, frame
        self.umin, self.umax = min(self.umin, u), max(self.umax, u)
        self.vmin, self.vmax = min(self.vmin, v), max(self.vmax, v)
        self.recent.append((u, v))

    @property
    def life(self) -> int:
        return self.last - self.first + 1

    @property
    def spread(self) -> float:
        """Lifetime bounding-box size (px) -- small => parked."""
        return max(self.umax - self.umin, self.vmax - self.vmin)

    def moving(self, win_disp: float, min_life: int) -> bool:
        if self.life < min_life or len(self.recent) < 2:
            return False
        (u0, v0), (u1, v1) = self.recent[0], self.recent[-1]
        return float(np.hypot(u1 - u0, v1 - v0)) >= win_disp

    def is_static(self, min_frames: int, max_disp: float) -> bool:
        return self.life >= min_frames and self.spread <= max_disp


def _episodes(flags: List[bool]) -> Tuple[int, int]:
    """(#frames True, #contiguous runs) -- a run is one real-world occurrence."""
    total = sum(1 for f in flags if f)
    runs = sum(1 for i, f in enumerate(flags) if f and (i == 0 or not flags[i - 1]))
    return total, runs


def run_case_audit(config: Dict[str, Any], max_frames: Optional[int] = None,
                   yellow_gate: bool = True,
                   link_radius: float = 90.0, drop_after: int = 10,
                   moving_disp: float = 60.0, moving_min_life: int = 4,
                   static_min_frames: int = 40, static_max_disp: float = 28.0,
                   active_moving_px_s: float = 400.0, static_far_px: float = 150.0,
                   slow_thresholds_px_s: Tuple[float, ...] = (500.0, 1000.0, 2000.0),
                   apex_slow_px_s: float = 600.0, apex_fast_px_s: float = 1500.0,
                   apex_window: int = 8, suppress: bool = False,
                   event_guard: int = 4, save_examples: int = 6) -> Dict[str, Any]:
    """Run the detector+tracker over the clip and count the three cases. Returns the
    summary dict (also written to output/ball_case_audit.json)."""
    out_dir = config.get("output", {}).get("dir", "output")
    stem = os.path.splitext(os.path.basename(config["source"]))[0]   # per-camera namespace
    ex_dir = os.path.join(out_dir, f"case_audit_{stem}")
    os.makedirs(ex_dir, exist_ok=True)

    dec = config.get("decode", {})
    reader = ThreadedVideoReader(config["source"], hw_accel=dec.get("hw_accel", True))
    fps = reader.fps or 20.0
    det = _build_detector(config)
    trk, _ = _build_tracker(config, fps)
    hom = Homography.from_config(config)
    events = _build_events(config, hom)
    sup = None
    if suppress:
        from core.ball_suppress import StationaryBallSuppressor
        sup = StationaryBallSuppressor.from_config(config)
    n_suppressed = 0

    tracklets: List[_Tracklet] = []
    per_frame: List[Dict[str, Any]] = []          # one row/frame for offline counting
    static_seen: Dict[int, Dict[str, Any]] = {}   # id(tracklet) -> static episode info
    cand_hist = {0: 0, 1: 0, 2: 0, 3: 0}          # 3 == "3+"
    ex_saved = {1: 0, 2: 0, 3: 0}
    f = 0

    print(f"[case-audit] {config['source']}  fps={fps:.3f}  yellow_gate={yellow_gate}")
    while True:
        ok, frame = reader.read()
        if not ok:
            break

        det.detect(frame)
        cands = list(getattr(det, "last_candidates", []))
        if yellow_gate:
            cands = [c for c in cands if _is_yellow(frame, c.u, c.v)]

        # --- the tracker gets the SUPPRESSED candidates (parked blobs removed); the case
        # analysis below still uses the RAW `cands` so parked balls are still COUNTED ---
        cands_trk = cands
        if sup is not None:
            cands_trk = sup.filter(f, cands)
            n_suppressed += sup.last_suppressed
        track = trk.update_multi(cands_trk) if trk is not None else None
        ev = events.update(f, track) if (events is not None and track is not None) else None

        # --- link candidates to tracklets (greedy nearest within link_radius) ---
        alive = [t for t in tracklets if f - t.last <= drop_after]
        tracklets = alive
        for c in cands:
            best, bestd = None, link_radius ** 2
            for t in tracklets:
                d = (t.u - c.u) ** 2 + (t.v - c.v) ** 2
                if d <= bestd:
                    best, bestd = t, d
            if best is None:
                tracklets.append(_Tracklet(f, c.u, c.v))
            else:
                best.add(f, c.u, c.v)

        updated = [t for t in tracklets if t.last == f]
        moving = [t for t in updated if t.moving(moving_disp, moving_min_life)]
        static_now = [t for t in updated if t.is_static(static_min_frames, static_max_disp)]
        for t in static_now:                       # remember each parked-ball episode
            static_seen.setdefault(id(t), {"frame": t.first, "u": t.u, "v": t.v,
                                           "life": t.life})["life"] = t.life

        # candidate-count histogram (after the yellow gate)
        cand_hist[min(len(cands), 3)] += 1

        # active ball state
        sp = float(track.speed_px_s) if (track and track.x is not None) else 0.0
        a_meas = bool(track.measured) if track else False
        a_track = (track.status == "tracking") if track else False
        ax = (float(track.x), float(track.y)) if (track and track.x is not None) else None

        # Case 2 dangerous overlap: a parked ball present AND active ball moving & far
        c2_overlap = False
        if static_now and ax is not None and sp >= active_moving_px_s:
            c2_overlap = any(np.hypot(ax[0] - t.u, ax[1] - t.v) >= static_far_px
                             for t in static_now)

        per_frame.append({
            "f": f, "n_cand": len(cands), "n_moving": len(moving),
            "n_static": len(static_now), "speed": sp, "measured": a_meas,
            "tracking": a_track, "event": ev.type if ev else None,
            "c1": len(moving) >= 2, "c2_overlap": c2_overlap,
        })

        # --- save a few example frames per case so we can eyeball realness ---
        def _save(case: int, tag: str):
            if ex_saved[case] >= save_examples:
                return
            vis = frame.copy()
            for c in cands:
                cv2.circle(vis, (int(c.u), int(c.v)), 16, (0, 0, 255), 3)
            if ax is not None:
                cv2.circle(vis, (int(ax[0]), int(ax[1])), 22, (0, 255, 0), 2)
            cv2.putText(vis, f"frame {f}  {tag}", (40, 70), _FONT, 1.6, (0, 255, 255), 3)
            small = cv2.resize(vis, (vis.shape[1] // 3, vis.shape[0] // 3))
            cv2.imwrite(os.path.join(ex_dir, f"case{case}_{f:06d}.jpg"), small)
            ex_saved[case] += 1

        if len(moving) >= 2:
            _save(1, f"{len(moving)} moving balls")
        if static_now and ex_saved[2] < save_examples and f % 30 == 0:
            _save(2, "parked ball present")

        f += 1
        if f % 500 == 0:
            print(f"[case-audit]   {f} frames... (cands>=2 so far: "
                  f"{sum(1 for r in per_frame if r['n_cand'] >= 2)})")
        if max_frames is not None and f >= max_frames:
            break

    reader.stop()

    # ----------------------------- offline counting ----------------------------
    n = len(per_frame)
    ev_frames = [r["f"] for r in per_frame if r["event"]]
    ev_set = set()
    for ef in ev_frames:
        ev_set.update(range(ef - event_guard, ef + event_guard + 1))

    c1_flags = [r["c1"] for r in per_frame]
    c1_total, c1_runs = _episodes(c1_flags)

    c2_flags = [r["c2_overlap"] for r in per_frame]
    c2_total, c2_runs = _episodes(c2_flags)
    parked = sorted(static_seen.values(), key=lambda d: d["frame"])

    # Case 3: measured + tracking + slow + not near an event
    tracking_speeds = [r["speed"] for r in per_frame
                       if r["measured"] and r["tracking"] and r["f"] not in ev_set]
    pct = {}
    if tracking_speeds:
        arr = np.array(tracking_speeds)
        for p in (5, 10, 25, 50):
            pct[f"p{p}"] = round(float(np.percentile(arr, p)), 1)
    c3_counts = {}
    for thr in slow_thresholds_px_s:
        flags = [(r["measured"] and r["tracking"] and r["f"] not in ev_set
                  and 0 < r["speed"] < thr) for r in per_frame]
        t, runs = _episodes(flags)
        c3_counts[f"under_{int(thr)}px_s"] = {"frames": t, "episodes": runs,
                                              "pct_of_tracking": round(
                                                  100 * t / max(1, len(tracking_speeds)), 2)}

    # Case 3 (TRUSTWORTHY): a genuine APEX -- a slow frame flanked by FAST frames on the
    # same active track (fast -> slow -> fast). This excludes the tracker idling on a slow
    # blob (which stays slow throughout, never flanked by fast play).
    spd = [r["speed"] for r in per_frame]
    meas = [r["measured"] and r["tracking"] for r in per_frame]
    apex_flags = [False] * n
    for i in range(n):
        if not (meas[i] and per_frame[i]["f"] not in ev_set and 0 < spd[i] < apex_slow_px_s):
            continue
        lo, hi = max(0, i - apex_window), min(n, i + apex_window + 1)
        before = max((spd[j] for j in range(lo, i) if meas[j]), default=0.0)
        after = max((spd[j] for j in range(i + 1, hi) if meas[j]), default=0.0)
        if before >= apex_fast_px_s and after >= apex_fast_px_s:
            apex_flags[i] = True
    apex_total, apex_runs = _episodes(apex_flags)

    summary = {
        "source": config["source"],
        "frames_processed": n,
        "fps": round(fps, 4),
        "yellow_gate": yellow_gate,
        "suppression": {"enabled": suppress, "candidates_dropped": n_suppressed},
        "thresholds": {
            "link_radius_px": link_radius, "moving_disp_px": moving_disp,
            "static_min_frames": static_min_frames, "static_max_disp_px": static_max_disp,
            "active_moving_px_s": active_moving_px_s, "static_far_px": static_far_px,
            "event_guard_frames": event_guard,
        },
        "candidate_hist": {"0": cand_hist[0], "1": cand_hist[1], "2": cand_hist[2],
                           "3plus": cand_hist[3]},
        "frames_with_ball": sum(1 for r in per_frame if r["measured"]),
        "events_detected": len(ev_frames),
        "case1_two_moving_balls": {"frames": c1_total, "episodes": c1_runs,
                                   "pct_of_frames": round(100 * c1_total / max(1, n), 2)},
        "case2_parked_ball": {"distinct_parked": len(parked),
                              "danger_overlap_frames": c2_total,
                              "danger_overlap_episodes": c2_runs,
                              "examples": parked[:10]},
        "case3_slow_in_play": {"tracking_frames": len(tracking_speeds),
                               "speed_percentiles_px_s": pct, "by_threshold": c3_counts,
                               "genuine_apex": {"frames": apex_total, "episodes": apex_runs,
                                                "def": f"fast(>{int(apex_fast_px_s)})->"
                                                       f"slow(<{int(apex_slow_px_s)})->fast "
                                                       f"within +/-{apex_window}f"}},
        "example_frames_dir": ex_dir,
    }

    out_path = os.path.join(out_dir, f"ball_case_audit_{stem}.json")
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2, cls=NumpyEncoder)

    # ----------------------------- console report ------------------------------
    print("\n================ BALL CASE AUDIT ================")
    print(f"source            : {config['source']}")
    print(f"frames processed  : {n}   ({n / fps:.1f}s)   ball seen: "
          f"{summary['frames_with_ball']}   events: {len(ev_frames)}")
    print(f"candidates/frame  : 0:{cand_hist[0]}  1:{cand_hist[1]}  "
          f"2:{cand_hist[2]}  3+:{cand_hist[3]}   (yellow_gate={yellow_gate})")
    print("-------------------------------------------------")
    print(f"CASE 1  two moving balls : {c1_total} frames in {c1_runs} episodes "
          f"({summary['case1_two_moving_balls']['pct_of_frames']}% of frames)")
    print(f"CASE 2  parked ball(s)   : {len(parked)} distinct; dangerous overlap "
          f"{c2_total} frames in {c2_runs} episodes")
    print(f"CASE 3  slow in-play     : tracking frames {len(tracking_speeds)}; "
          f"speed pctiles {pct}")
    for k, v in c3_counts.items():
        print(f"          {k:>16} : {v['frames']} frames "
              f"({v['pct_of_tracking']}% of tracking) in {v['episodes']} episodes")
    print(f"          genuine apex   : {apex_total} frames in {apex_runs} episodes "
          f"(fast->slow->fast)  <-- the real Case-3 count")
    print(f"example frames -> {ex_dir}")
    print(f"summary        -> {out_path}")
    print("=================================================\n")
    return summary


if __name__ == "__main__":
    import sys

    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config-side1.json"
    mf = int(sys.argv[2]) if len(sys.argv) > 2 else None
    with open(cfg_path, "r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    run_case_audit(cfg, max_frames=mf)
