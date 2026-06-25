"""core/ball_eval.py
Phase-1 BALL quality gate -- a STANDALONE harness (it never touches the player
pipeline). It runs the TrackNetV2 ball detector over one clip and puts NUMBERS on
its quality, exactly the way utils/metrics.py does for the player pipeline:

    "Measure failures, don't hide them." (architecture decision #5)

WHAT IT MEASURES (none of it needs ground-truth labels -- these are proxies we can
compute on any clip):
  * detection_rate ......... fraction of (non-warmup) frames where a ball was found
  * confidence ............. mean / median / min / max of the heatmap peak on hits
  * longest_no_ball_run .... longest run of consecutive "no ball" frames. This is
                             the occlusion proxy: the ball IS routinely hidden (a
                             player's body, the net, motion blur), and the detector
                             MUST be allowed to say "no ball" on those frames rather
                             than hallucinate one. A sane run shows occasional gaps,
                             not one endless gap (dead detector) nor zero gaps
                             (over-firing on noise).
  * mode ................... "stub-no-weights" until trained weights exist, so the
                             report is honest about WHY the rate is what it is.

OUTPUTS (under config["output"]["dir"], default output/)
  * ball_eval.jsonl ........ one row per frame: {frame, found, u, v, confidence, reason}
  * ball_eval_metrics.json . the summary above (written via NumpyEncoder)
  * ball_eval_overlay.mp4 .. (only with --save-video) the clip with the ball drawn

RUN
    python main.py --ball-eval --source side-1-full-vid.mp4 --show
    python main.py --ball-eval --max-frames 500 --save-video
"""
from __future__ import annotations

import json
import os
import time
from collections import deque
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from core.ball_detector import BallDetector
from core.ball_events import BallEventDetector
from core.ball_tracker import BallTracker
from utils.display import PlaybackThrottle
from utils.homography import Homography
from utils.metrics import NumpyEncoder
from utils.video_io import ThreadedVideoReader


def _build_detector(config: Dict[str, Any]):
    """Create the configured ball detector purely from config (no hard-coded
    tunables). config["ball"]["method"] selects the backend:
        "tracknet" (default) -> TrackNetV2 (needs trained weights)
        "motion"             -> classical MOG2 baseline (works without weights)
    """
    from utils import roi as roi_utils
    ball = config.get("ball", {})
    method = ball.get("method", "tracknet")

    # COURT ROI shared by both backends: keep only ball detections inside the court
    # polygon, dilated by margin_px to allow airborne balls (kills background lights).
    roi_cfg = ball.get("roi", {})
    roi_on = roi_cfg.get("enabled", True)
    roi_margin = float(roi_cfg.get("margin_px", 200))
    court_poly = roi_utils.to_polygon(config.get("court", {}).get("polygon"))

    if method == "motion":
        from core.ball_detector_motion import MotionBallDetector
        m = ball.get("motion", {})
        mp = court_poly if (roi_on and m.get("use_court_roi", True)) else None
        return MotionBallDetector(
            history=m.get("history", 200),
            var_threshold=m.get("var_threshold", 25),
            min_area=m.get("min_area", 8),
            max_area=m.get("max_area", 1500),
            min_circularity=m.get("min_circularity", 0.35),
            morph_kernel=m.get("morph_kernel", 3),
            warmup_frames=m.get("warmup_frames", 15),
            court_polygon=mp,
            roi_margin_px=roi_margin,
        )

    return BallDetector(
        weights_path=ball.get("weights"),
        # ball block may set its own device; otherwise reuse the top-level one
        device=ball.get("device", config.get("device", "cuda")),
        input_width=ball.get("input_width", 512),
        input_height=ball.get("input_height", 288),
        in_frames=ball.get("in_frames", 3),
        heatmap_threshold=ball.get("heatmap_threshold", 0.5),
        min_blob_area=ball.get("min_blob_area", 2),
        max_blob_area=ball.get("max_blob_area", 0),
        court_polygon=(court_poly if roi_on else None),
        roi_margin_px=roi_margin,
        crop=ball.get("crop"),                 # court-crop box (or None) -> ball-only
    )


def _draw_ball(frame: np.ndarray, det) -> None:
    """Annotate the frame: a circle + confidence on a hit, a banner on a miss."""
    if det.found:
        u, v = int(det.u), int(det.v)
        cv2.circle(frame, (u, v), 10, (0, 255, 255), 2)
        cv2.circle(frame, (u, v), 2, (0, 255, 255), -1)
        cv2.putText(frame, f"ball {det.confidence:.2f}", (u + 12, v - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(frame, f"no ball ({det.reason})", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)


def _build_tracker(config: Dict[str, Any], fps_in: float):
    """Build the Phase-2 Kalman tracker from config['ball']['tracker'] (or None when
    disabled). dt defaults to the clip's real frame period (1/fps). Returns
    (tracker, trail_len)."""
    t = config.get("ball", {}).get("tracker", {})
    if not t.get("enabled", True):
        return None, 0
    dt = t.get("dt") or (1.0 / fps_in if fps_in else 0.05)
    tracker = BallTracker(
        dt=dt,
        process_noise=t.get("process_noise", 50000.0),
        meas_noise=t.get("meas_noise", 25.0),
        conf_ref=t.get("conf_ref", 0.7),
        conf_floor=t.get("conf_floor", 0.05),
        max_coast_frames=t.get("max_coast_frames", 15),
        gate=t.get("gate", 0.0),
        min_updates_before_gating=t.get("min_updates_before_gating", 3),
        assoc_radius=t.get("assoc_radius_px", 600.0),
    )
    return tracker, int(t.get("trail_len", 30))


def _draw_track(frame: np.ndarray, track, trail, disp) -> None:
    """Draw the smoothed trail and the ball marker at `disp`.

    `disp` is chosen ZERO-LAG: on a measured frame it is the RAW detection (so the
    marker sits exactly where TrackNet found the ball, no filter lag); during a gap
    it is the Kalman PREDICTION. GREEN = measured this frame, ORANGE = predicted."""
    for i in range(1, len(trail)):                  # the smoothed trail
        cv2.line(frame, trail[i - 1], trail[i], (255, 180, 0), 2, cv2.LINE_AA)

    if track is None or disp is None:
        if track is not None and track.status == "lost":
            cv2.putText(frame, "ball: lost", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return

    x, y = disp
    if track.status == "tracking":
        cv2.circle(frame, (x, y), 10, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
        cv2.putText(frame, f"ball  {track.speed_px_s:.0f} px/s", (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:                                           # coasting -> predicted in a gap
        cv2.circle(frame, (x, y), 10, (0, 165, 255), 2)
        cv2.putText(frame, f"PRED ({track.coast})", (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)


def _build_events(config: Dict[str, Any], homography):
    """Build the Phase-3 event detector from config['ball']['events'] (or None when
    disabled). Needs the homography for court metres / in-out."""
    e = config.get("ball", {}).get("events", {})
    if not e.get("enabled", True):
        return None
    return BallEventDetector(
        homography=homography,
        min_vy_px_s=e.get("min_vy_px_s", 500.0),
        min_vx_px_s=e.get("min_vx_px_s", 500.0),
        wall_margin_m=e.get("wall_margin_m", 0.6),
        hit_angle_deg=e.get("hit_angle_deg", 70.0),
        hit_min_speed_px_s=e.get("hit_min_speed_px_s", 1500.0),
        refractory_frames=e.get("refractory_frames", 3),
        in_out_margin_m=e.get("in_out_margin_m", 0.1),
        near_court_margin_m=e.get("near_court_margin_m", 1.5),
    )


def _draw_events(frame: np.ndarray, recent_events, cur_frame: int, ttl: int = 25) -> None:
    """Mark recent events: green diamond = floor bounce, orange = wall, red = hit.
    The label (with in/out + court metres) shows for the first few frames."""
    colors = {"floor_bounce": (0, 255, 0), "wall_bounce": (255, 128, 0), "hit": (0, 0, 255)}
    for fno, ev in recent_events:
        age = cur_frame - fno
        if age > ttl:
            continue
        col = colors.get(ev.type, (255, 255, 255))
        cv2.drawMarker(frame, (int(ev.u), int(ev.v)), col, cv2.MARKER_DIAMOND, 28, 3)
        if age <= 4:
            label = ev.type.replace("_", " ").upper()
            if ev.type == "floor_bounce" and ev.in_court is not None:
                label += " IN" if ev.in_court else " OUT"
            if ev.x_m is not None:
                label += f"  ({ev.x_m:.1f},{ev.y_m:.1f})m"
            cv2.putText(frame, label, (int(ev.u) + 16, int(ev.v) + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, col, 2)


def _summarize(confidences: List[float]) -> Dict[str, Any]:
    """mean/median/min/max of the hit confidences (or zeros when there were none)."""
    if not confidences:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(confidences, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def run_ball_eval(
    config: Dict[str, Any],
    show: bool = False,
    save_video: bool = False,
    max_frames: Optional[int] = None,
    profile: bool = False,
) -> Dict[str, Any]:
    """Run the ball detector over config["source"] and write the gate report.
    Returns the summary dict (also saved to output/ball_eval_metrics.json)."""
    out_dir = config.get("output", {}).get("dir", "output")
    os.makedirs(out_dir, exist_ok=True)

    detector = _build_detector(config)

    reader = ThreadedVideoReader(
        config["source"],
        hw_accel=config.get("decode", {}).get("hw_accel", True),
        queue_size=config.get("decode", {}).get("queue_size", 4),
    )
    fps_in = reader.fps or 20.0
    tracker, trail_len = _build_tracker(config, fps_in)
    trail: "deque" = deque(maxlen=max(1, trail_len))

    homog = Homography.from_config(config)
    events = _build_events(config, homog)
    recent_events: "deque" = deque(maxlen=90)   # for persistent on-frame markers
    event_counts = {"floor_bounce": 0, "wall_bounce": 0, "hit": 0}
    if events is not None and homog is None:
        print("[ball-eval] note: no homography in this config -> events fire but lack "
              "court metres / in-out. Use config-side1.json for full Phase-3 output.")

    # The overlay writer is created LAZILY on the first decoded frame, sized from
    # that frame's real shape. reader.width/height can read back 0 on some sources
    # (odd codecs / HW paths), and cv2.VideoWriter does NOT raise on a (0,0) size --
    # it just silently drops every frame into an empty file. Deferring guarantees a
    # correct size and lets us surface an open failure instead of hiding it.
    writer = None
    writer_path = os.path.join(out_dir, "ball_eval_overlay.mp4")

    throttle = PlaybackThrottle(config.get("display", {}).get("playback_fps", 0))
    if show:
        cv2.namedWindow("Ball Eval (Phase 1)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ball Eval (Phase 1)", 1280, 720)

    jsonl = open(os.path.join(out_dir, "ball_eval.jsonl"), "w")

    # --- accumulators for the gate numbers ---
    n = 0
    warmup_frames = 0
    frames_with_ball = 0
    confidences: List[float] = []
    cur_no_ball_run = 0
    longest_no_ball_run = 0
    # Phase-2 tracker accumulators
    frames_with_track = frames_tracking = frames_coasting = frames_gated = 0

    t0 = time.time()
    while True:
        loop_t0 = time.time()
        ok, frame = reader.read()
        if not ok:
            break

        det = detector.detect(frame)
        # feed ALL candidates to the tracker so it can pick the MOVING ball over a
        # static distractor (light / limb); det is still the raw "best" for metrics.
        track = (tracker.update_multi(getattr(detector, "last_candidates", []))
                 if tracker is not None else None)

        ev = events.update(n, track) if events is not None else None
        if ev is not None:
            event_counts[ev.type] += 1
            recent_events.append((n, ev))

        rec: Dict[str, Any] = {"frame": n, **det.to_dict()}
        if track is not None:
            rec["track"] = track.to_dict()
        if ev is not None:
            rec["event"] = ev.to_dict()
        jsonl.write(json.dumps(rec, cls=NumpyEncoder) + "\n")

        # Pick the ZERO-LAG display point: the raw detection on a measured frame
        # (the Kalman estimate itself trails a fast ball), the prediction in a gap.
        disp = None
        if track is not None:
            if track.gated:
                frames_gated += 1
            if track.measured and track.meas_u is not None:
                disp = (int(track.meas_u), int(track.meas_v))  # chosen candidate -> no lag
            elif track.x is not None:
                disp = (int(track.x), int(track.y))            # prediction bridges the gap
            if track.x is not None:
                frames_with_track += 1
                if track.status == "tracking":
                    frames_tracking += 1
                elif track.status == "coasting":
                    frames_coasting += 1
            if disp is not None:
                trail.append(disp)
            else:
                trail.clear()

        if det.reason == "warmup":
            warmup_frames += 1
        elif det.found:
            frames_with_ball += 1
            confidences.append(det.confidence)
            cur_no_ball_run = 0
        else:
            # A genuine "no ball" decision -> grows the occlusion-run proxy. Only
            # meaningful when the detector is actually operational: a stub TrackNet
            # returns "no ball" every frame, which would otherwise fake a full-clip
            # dead-detector run. So we only accumulate when it can really detect.
            if detector.operational:
                cur_no_ball_run += 1
                longest_no_ball_run = max(longest_no_ball_run, cur_no_ball_run)

        if save_video or show:
            if tracker is not None:
                _draw_track(frame, track, trail, disp)
            else:
                _draw_ball(frame, det)
            if events is not None:
                _draw_events(frame, recent_events, n)
            if save_video:
                if writer is None:                       # create on first frame
                    fh, fw = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        writer_path, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (fw, fh))
                    if not writer.isOpened():
                        print(f"[ball-eval] WARNING: could not open VideoWriter "
                              f"({fw}x{fh}) at {writer_path} -> overlay video disabled.")
                        writer = None
                        save_video = False               # stop retrying every frame
                if writer is not None:
                    writer.write(frame)
            if show:
                cv2.imshow("Ball Eval (Phase 1)", frame)
                work_ms = (time.time() - loop_t0) * 1000.0
                if throttle.wait(work_ms) == ord("q"):
                    break

        n += 1
        if max_frames is not None and n >= max_frames:
            break

    reader.stop()
    if writer is not None:
        writer.release()
    jsonl.close()
    if show:
        cv2.destroyAllWindows()

    elapsed = time.time() - t0
    proc_fps = round(n / elapsed, 2) if elapsed > 0 else 0.0
    scored = max(1, n - warmup_frames)   # frames that COULD have produced a ball

    summary = {
        "source": config["source"],
        "mode": detector.mode_label,
        "frames_processed": n,
        "warmup_frames": warmup_frames,
        "frames_with_ball": frames_with_ball,
        "detection_rate": round(frames_with_ball / scored, 4),
        "confidence": _summarize(confidences),
        # null when not operational (no real detector decisions to measure a run from)
        "longest_no_ball_run": (longest_no_ball_run if detector.operational else None),
        "source_fps": round(float(fps_in), 2),
        "processing_fps": proc_fps,
        "device": str(detector.device),
    }
    if tracker is not None:
        # track_coverage should be HIGHER than detection_rate: the Kalman filter
        # coasts through gaps, so we have a ball position on frames the detector
        # missed. frames_gated = bad detections the tracker rejected.
        summary["tracker"] = {
            "enabled": True,
            "frames_with_track": frames_with_track,
            "track_coverage": round(frames_with_track / scored, 4),
            "frames_tracking": frames_tracking,
            "frames_coasting": frames_coasting,
            "frames_gated": frames_gated,
        }
    if events is not None:
        summary["events"] = {**event_counts, "total": sum(event_counts.values())}
    metrics_path = os.path.join(out_dir, "ball_eval_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # --- human-readable report ---
    print(f"\n[ball-eval] processed {n} frames in {elapsed:.1f}s ({proc_fps} FPS)")
    if not detector.operational:
        print("[ball-eval] MODE = STUB (no weights): detection_rate is 0 BY DESIGN. "
              "This run proves the plumbing only.")
        print("[ball-eval] Provide trained weights at config['ball']['weights'], or "
              "set ball.method='motion' for the training-free baseline "
              "(see docs/ball_labeling.md).")
    else:
        print(f"[ball-eval] detection_rate = {summary['detection_rate']*100:.1f}% "
              f"of {scored} scored frames "
              f"(confidence mean={summary['confidence']['mean']}, "
              f"longest no-ball run={longest_no_ball_run} frames)")
        if tracker is not None:
            tr = summary["tracker"]
            print(f"[ball-eval] track_coverage = {tr['track_coverage']*100:.1f}% "
                  f"(measured={tr['frames_tracking']}, gap-filled={tr['frames_coasting']}, "
                  f"outliers_rejected={tr['frames_gated']})  <- Kalman vs "
                  f"{summary['detection_rate']*100:.1f}% raw")
        if events is not None:
            print(f"[ball-eval] events: {event_counts['floor_bounce']} floor bounces, "
                  f"{event_counts['wall_bounce']} wall bounces, {event_counts['hit']} hits")
    print(f"[ball-eval] metrics -> {metrics_path}")
    print(f"[ball-eval] per-frame log -> {os.path.join(out_dir, 'ball_eval.jsonl')}")
    return summary
