"""main.py -- Phase-1 entry point.

Modes:
  1) Normal run:             python main.py --source side-1-full-vid.mp4 --show
  2) Define the court ROI:   python main.py --calibrate-roi
  3) Define the homography:  python main.py --calibrate-homography

Everything tunable comes from config.json; CLI flags only OVERRIDE a few of
those values for convenience. We never hard-code thresholds in logic.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Dict

import cv2

from core.pipeline import run
from utils import homography as homography_utils


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def save_config(path: str, config: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def calibrate_roi(config: Dict[str, Any], config_path: str) -> None:
    """Click the court polygon on the first frame and save it to config.json.

    Controls:  Left-click = add point,  Right-click = undo,
               S = save & quit,         Q = quit without saving.

    We keep this human-in-the-loop on purpose: the court boundary is a judgment
    call best made by a person looking at the real court, not guessed by code.
    """
    cap = cv2.VideoCapture(config["source"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {config['source']}")

    # The frame is 4K. Downscale it to fit on screen for clicking, but remember
    # the scale so we can map every click back to FULL-resolution pixels (the
    # pipeline runs at full res, so the saved polygon must be in full-res coords).
    full_h, full_w = frame.shape[:2]
    max_w = 1280
    scale = min(1.0, max_w / full_w)
    disp_w, disp_h = int(full_w * scale), int(full_h * scale)
    base = cv2.resize(frame, (disp_w, disp_h))

    points: list[list[int]] = []  # stored in DISPLAY coords while clicking

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
        elif event == cv2.EVENT_RBUTTONDOWN and points:
            points.pop()

    cv2.namedWindow("calibrate-roi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("calibrate-roi", disp_w, disp_h)
    cv2.setMouseCallback("calibrate-roi", on_mouse)

    while True:
        disp = base.copy()
        for i, p in enumerate(points):
            cv2.circle(disp, (p[0], p[1]), 4, (0, 0, 255), -1)
            if i > 0:
                cv2.line(disp, tuple(points[i - 1]), tuple(p), (0, 255, 255), 2)
        if len(points) >= 3:  # show the closing edge as a hint
            cv2.line(disp, tuple(points[-1]), tuple(points[0]), (0, 255, 255), 1)
        cv2.putText(disp, "L:add  R:undo  S:save  Q:quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("calibrate-roi", disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("s") and len(points) >= 3:
            # scale DISPLAY clicks back up to full-resolution pixels before saving
            full_pts = [[int(round(px / scale)), int(round(py / scale))]
                        for px, py in points]
            config.setdefault("court", {})["polygon"] = full_pts
            save_config(config_path, config)
            print(f"[calibrate] saved {len(full_pts)} full-res points to {config_path}")
            break
        if key == ord("q"):
            print("[calibrate] cancelled, nothing saved.")
            break
    cv2.destroyAllWindows()


def calibrate_homography(config: Dict[str, Any], config_path: str) -> None:
    """Click known court landmarks on the first frame, fit the pixel<->meters
    homography, preview it, and save H + H_inv into config.json.

    Why human-in-the-loop: the court corners and line intersections are best
    identified by a person who knows the court. The tool walks you through the 8
    landmarks one at a time, FAR baseline first (the end facing the camera).

    This camera shoots across to the opposite side, so the NEAR baseline (the two
    LAST landmarks, under the camera) usually is not in frame -> press N to skip
    them. The other camera covers that end.

    Controls (COLLECT phase):
        Left-click .. set the currently-prompted landmark
        N ........... landmark NOT visible -> skip it (near baseline usually is)
        U ........... undo the last action
        C ........... compute the homography from what you've clicked (>=4 needed)
        Q ........... quit without saving
    Controls (PREVIEW phase):
        S ........... save to config.json
        R ........... go back and keep clicking
        Q ........... quit without saving
    """
    cap = cv2.VideoCapture(config["source"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {config['source']}")

    # 4K -> downscale for clicking, remember scale to map clicks back to full res
    # (the pipeline and the saved matrix work in FULL-resolution pixels).
    full_h, full_w = frame.shape[:2]
    scale = min(1.0, 1280 / full_w)
    disp_w, disp_h = int(full_w * scale), int(full_h * scale)
    base = cv2.resize(frame, (disp_w, disp_h))

    landmarks = homography_utils.PADEL_LANDMARKS
    records: list[dict] = []   # {"idx", "clicked": bool, "px_full", "world"}
    idx = 0                    # which landmark we are currently asking for
    pending_click: dict = {"pt": None}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click["pt"] = (x, y)

    win = "calibrate-homography"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, on_mouse)
    font = cv2.FONT_HERSHEY_SIMPLEX

    state = "collect"
    homog = None
    err_mean = err_max = 0.0
    H = H_inv = None
    img_pts: list = []
    wrld_pts: list = []

    def n_clicked() -> int:
        return sum(1 for r in records if r["clicked"])

    while True:
        if state == "collect":
            disp = base.copy()
            # show every landmark already placed
            for r in records:
                if r["clicked"]:
                    dx = int(r["px_full"][0] * scale)
                    dy = int(r["px_full"][1] * scale)
                    cv2.circle(disp, (dx, dy), 5, (0, 0, 255), -1)
                    cv2.putText(disp, str(r["idx"] + 1), (dx + 6, dy - 6),
                                font, 0.5, (0, 255, 255), 1)
            # consume a click for the current landmark
            if pending_click["pt"] is not None and idx < len(landmarks):
                cx, cy = pending_click["pt"]
                px_full = [cx / scale, cy / scale]
                records.append({"idx": idx, "clicked": True,
                                "px_full": px_full, "world": list(landmarks[idx][1])})
                idx += 1
            pending_click["pt"] = None

            if idx < len(landmarks):
                name, world = landmarks[idx]
                prompt = f"[{idx + 1}/{len(landmarks)}] CLICK: {name}  ({world[0]:.2f}, {world[1]:.2f} m)"
            else:
                prompt = "All landmarks visited. Press C to compute."
            cv2.putText(disp, prompt, (10, 28), font, 0.6, (255, 255, 255), 2)
            cv2.putText(disp, f"clicked={n_clicked()} (need >=4)  L:set  N:skip  U:undo  C:compute  Q:quit",
                        (10, disp_h - 14), font, 0.55, (200, 200, 200), 2)
            cv2.imshow(win, disp)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("n") and idx < len(landmarks):
                records.append({"idx": idx, "clicked": False,
                                "px_full": None, "world": list(landmarks[idx][1])})
                idx += 1
            elif key == ord("u") and records:
                last = records.pop()
                idx = last["idx"]
            elif key == ord("c"):
                if n_clicked() < 4:
                    print(f"[calibrate-h] need >=4 clicked points, have {n_clicked()}.")
                    continue
                img_pts = [r["px_full"] for r in records if r["clicked"]]
                wrld_pts = [r["world"] for r in records if r["clicked"]]
                H, H_inv, mask = homography_utils.compute_homography(img_pts, wrld_pts)
                err_mean, err_max = homography_utils.reprojection_error_m(H, img_pts, wrld_pts)
                homog = homography_utils.Homography(H, H_inv)
                n_in = int(mask.sum())
                print(f"[calibrate-h] {len(img_pts)} points, RANSAC inliers={n_in}, "
                      f"reprojection error mean={err_mean:.3f}m max={err_max:.3f}m")
                state = "preview"
            elif key == ord("q"):
                print("[calibrate-h] cancelled, nothing saved.")
                break

        else:  # preview
            canvas = frame.copy()
            homography_utils.draw_court_overlay(canvas, homog)
            disp = cv2.resize(canvas, (disp_w, disp_h))
            cv2.putText(disp, f"reprojection error: mean={err_mean:.3f}m  max={err_max:.3f}m",
                        (10, 28), font, 0.6, (0, 255, 0), 2)
            cv2.putText(disp, "magenta grid should hug the real court lines | S:save  R:redo  Q:quit",
                        (10, disp_h - 14), font, 0.55, (200, 200, 200), 2)
            cv2.imshow(win, disp)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("s"):
                config["homography"] = homography_utils.to_config_dict(
                    img_pts, wrld_pts, H, H_inv, err_mean, err_max)
                save_config(config_path, config)
                print(f"[calibrate-h] saved homography ({len(img_pts)} points) to {config_path}")
                break
            elif key == ord("r"):
                state = "collect"
            elif key == ord("q"):
                print("[calibrate-h] cancelled, nothing saved.")
                break

    cv2.destroyAllWindows()


def sync_offset(config_a: Dict[str, Any], config_b: Dict[str, Any],
                config_a_path: str, out_dir: str = "output",
                stride: int = 10, max_lag_frames: int = 600,
                max_frames: int | None = None) -> None:
    """Estimate the time offset (in frames) between the two camera clips and save
    it into camera-A's config, with a side-by-side montage for the quality gate.

    This is the FIRST piece of Phase-4 fusion: every later step needs to know that
    the same instant lives at  side-A frame f  and  side-B frame f+offset.
    """
    from utils import sync as sync_utils

    src_a, src_b = config_a["source"], config_b["source"]
    print(f"[sync] building motion-energy for A={src_a} ...")
    _, e_a = sync_utils.motion_energy(src_a, stride=stride, max_frames=max_frames)
    print(f"[sync] building motion-energy for B={src_b} ...")
    _, e_b = sync_utils.motion_energy(src_b, stride=stride, max_frames=max_frames)

    result = sync_utils.estimate_offset(e_a, e_b, stride=stride,
                                        max_lag_frames=max_lag_frames)
    off = result["offset_frames"]
    print(f"[sync] offset_frames={off}  (side-B frame = side-A frame + {off})")
    print(f"[sync] corr={result['corr']}  runner_up={result['corr_runner_up']}  "
          f"margin={result['margin']}")
    if result["corr"] < 0.3 or result["margin"] < 0.05:
        print("[sync] WARNING: weak / ambiguous correlation -- treat this offset "
              "with suspicion and check the montage carefully.")

    import os
    os.makedirs(out_dir, exist_ok=True)
    montage_path = os.path.join(out_dir, "sync_check.jpg")
    ok = sync_utils.save_alignment_montage(src_a, src_b, off, montage_path)
    if ok:
        print(f"[sync] wrote alignment montage -> {montage_path} "
              f"(each row: A | B should show the SAME instant)")

    # persist into camera-A's config so fusion can read it back
    config_a["sync"] = {
        "other_source": src_b,
        "offset_frames": off,
        "corr": result["corr"],
        "margin": result["margin"],
        "method": "motion-energy-xcorr",
        "stride": stride,
    }
    save_config(config_a_path, config_a)
    print(f"[sync] saved sync block to {config_a_path}")


def verify_homography(config: Dict[str, Any]) -> None:
    """Redraw the SAVED homography overlay on the first frame so you can confirm,
    any time, that the calibration still lines up with the court. Read-only.

    Controls:  Q / Esc = close.
    """
    homog = homography_utils.Homography.from_config(config)
    if homog is None:
        print("[verify-h] no homography in config -- run --calibrate-homography first.")
        return

    saved = config["homography"].get("reprojection_error_m", {})
    cap = cv2.VideoCapture(config["source"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {config['source']}")

    homography_utils.draw_court_overlay(frame, homog)
    if saved:
        cv2.putText(frame, f"saved reprojection error: mean={saved.get('mean')}m  max={saved.get('max')}m",
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    full_h, full_w = frame.shape[:2]
    scale = min(1.0, 1280 / full_w)
    disp = cv2.resize(frame, (int(full_w * scale), int(full_h * scale)))
    win = "verify-homography"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp.shape[1], disp.shape[0])
    print("[verify-h] showing saved overlay -- press Q or Esc to close.")
    while True:
        cv2.imshow(win, disp)
        key = cv2.waitKey(20) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Padel CV -- Phase 1")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--config2", help="second camera's config (for --sync / --fuse)")
    parser.add_argument("--sync", action="store_true",
                        help="estimate the time offset between --config and --config2 clips and exit")
    parser.add_argument("--fuse", action="store_true",
                        help="run cross-camera fusion (needs --config2, sync + both homographies)")
    parser.add_argument("--ball-fuse", action="store_true",
                        help="dual-camera 3D BALL triangulation (needs --config2, sync + both homographies)")
    parser.add_argument("--start-frame", type=int, default=0,
                        help="side-A frame to start fusion from (for quick spot-checks)")
    parser.add_argument("--source", help="override config source (video path / index)")
    parser.add_argument("--model", help="override config model path")
    parser.add_argument("--show", action="store_true", help="display the annotated window")
    parser.add_argument("--save-video", action="store_true", help="write output/phase1_annotated.mp4")
    parser.add_argument("--max-frames", type=int, default=None, help="stop after N frames (for quick measuring)")
    parser.add_argument("--profile", action="store_true", help="print a per-stage timing breakdown at the end")
    parser.add_argument("--ball-eval", action="store_true",
                        help="run the Phase-1 ball-detector quality gate on --source and exit")
    parser.add_argument("--ball-3d", action="store_true",
                        help="end-to-end 3D ball pipeline (dual-cam triangulate + physics) "
                             "-> output/ball_3d_trajectory.csv; needs --config2")
    parser.add_argument("--ball-dual", action="store_true",
                        help="live side-by-side dual-cam ball view + top-down court with one "
                             "cross-camera ball; needs --config2")
    parser.add_argument("--sync-manual", action="store_true",
                        help="interactive by-eye sync tuner (nudge the frame offset) -> saves "
                             "to config-A; needs --config2")
    parser.add_argument("--label-ball", action="store_true",
                        help="open the click-to-label ball tool on --source and exit")
    parser.add_argument("--label-from",
                        help="with --label-ball: label only the frames listed in this CSV")
    parser.add_argument("--mine-hard", action="store_true",
                        help="find frames worth labeling (model misses / low conf) and exit")
    parser.add_argument("--precision", action="store_true",
                        help="score the detector against the labeled frames (real precision) and exit")
    parser.add_argument("--calibrate-roi", action="store_true", help="define the court polygon and exit")
    parser.add_argument("--calibrate-homography", action="store_true", help="define the pixel<->meters homography and exit")
    parser.add_argument("--verify-homography", action="store_true", help="redraw the saved homography overlay and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.source:
        config["source"] = args.source
    if args.model:
        config["model"] = args.model

    if args.ball_eval:
        # Standalone Phase-1 ball gate. Imported here (not at top) so a normal
        # player run never imports torch-heavy ball code it doesn't use.
        from core.ball_eval import run_ball_eval
        run_ball_eval(config, show=args.show, save_video=args.save_video,
                      max_frames=args.max_frames, profile=args.profile)
        return

    if args.mine_hard:
        from core.ball_mine import run_mine_hard
        run_mine_hard(config, max_frames=args.max_frames)
        return

    if args.precision:
        from core.ball_precision import run_precision
        run_precision(config)
        return

    if args.label_ball:
        from core.ball_label import run_label_ball, read_frame_list
        frames = read_frame_list(args.label_from) if args.label_from else None
        run_label_ball(config, frames=frames)
        return

    if args.calibrate_roi:
        calibrate_roi(config, args.config)
        return

    if args.calibrate_homography:
        calibrate_homography(config, args.config)
        return

    if args.verify_homography:
        verify_homography(config)
        return

    if args.sync:
        if not args.config2:
            parser.error("--sync requires --config2 (the second camera's config)")
        config_b = load_config(args.config2)
        out_dir = config.get("output", {}).get("dir", "output")
        sync_offset(config, config_b, args.config, out_dir=out_dir,
                    max_frames=args.max_frames)
        return

    if args.fuse:
        if not args.config2:
            parser.error("--fuse requires --config2 (the second camera's config)")
        from core.fusion import FusionPipeline
        config_b = load_config(args.config2)
        FusionPipeline(config, config_b).run(
            show=args.show, save_video=args.save_video,
            max_frames=args.max_frames, start_frame=args.start_frame,
            profile=args.profile)
        return

    if args.ball_fuse:
        if not args.config2:
            parser.error("--ball-fuse requires --config2 (the second camera's config)")
        from core.ball_fusion import run_ball_fusion
        config_b = load_config(args.config2)
        run_ball_fusion(config, config_b, show=args.show, save_video=args.save_video,
                        max_frames=args.max_frames, start_frame=args.start_frame)
        return

    if args.ball_3d:
        # End-to-end 3D ball pipeline: Phase-4 dual-cam triangulation + side-1 track/events,
        # then Phase-5 projectile EKF + RTS smoother -> 3D trajectory CSV (+ overlay).
        if not args.config2:
            parser.error("--ball-3d requires --config2 (the second camera's config)")
        from core.ball_3d import run_ball_3d
        config_b = load_config(args.config2)
        run_ball_3d(config, config_b, max_frames=args.max_frames,
                    save_video=args.save_video, show=args.show)
        return

    if args.ball_dual:
        # Live side-by-side dual-camera ball view + top-down court with one cross-camera ball.
        if not args.config2:
            parser.error("--ball-dual requires --config2 (the second camera's config)")
        from core.ball_dual import run_dual_view
        config_b = load_config(args.config2)
        run_dual_view(config, config_b, max_frames=args.max_frames,
                      show=args.show, save_video=args.save_video)
        return

    if args.sync_manual:
        # Interactive by-eye sync tuner: nudge the frame offset, save to config-A.
        if not args.config2:
            parser.error("--sync-manual requires --config2 (the second camera's config)")
        from core.sync_manual import run_manual_sync
        config_b = load_config(args.config2)
        run_manual_sync(config, config_b, args.config)
        return

    run(config, show=args.show, save_video=args.save_video,
        max_frames=args.max_frames, profile=args.profile)


if __name__ == "__main__":
    main()
