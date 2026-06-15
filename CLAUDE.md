# Padel CV System — Project Brief

Real-time computer vision for a padel court at InMobiles Holding (Mkalles, Lebanon).
Detect players, track them with stable IDs, draw skeletons, recognize shots, generate
post-match reports. Runs on NVIDIA Jetson in production.

**AI Engineer Team:** Hussein Ibrahim, Jeffry Abou Zeidan, Kris Abi Daher (AI Engineer Interns).
All are junior and learning CV while building this. **Quality is the priority, not speed.**

## How to work on this project (READ FIRST)

- **One step at a time.** Do exactly the step requested, then STOP and report results.
  Do NOT build ahead into later phases or later steps without being asked.
- **Every step has a quality gate.** After building, we measure quality on REAL court
  footage before moving on. If you can't measure it, you're not done.
- **Explain the code.** The team is learning. Add clear comments and, when you write a
  new file, briefly explain in chat what each part does and why.
- **Ask, don't assume.** If a step is ambiguous, ask before writing code.
- **Correct first, then optimize at each phase boundary.** WITHIN a phase, get it
  correct and readable first — do not optimize mid-build. But optimization is NO LONGER
  deferred to Phase 6 alone: at the END of every phase, STOP, announce the phase is
  complete, and ASK whether to do an optimization pass before moving on. The user decides
  each time (yes -> optimize now; no -> move to the next phase). Speed work (TensorRT,
  threading, batching, FP16, GPU decode) is on the table at any phase boundary, not just
  Phase 6.

## Critical architecture decisions (do not violate)

1. **Two cameras is the product. Single stream is how we develop.**
   Each camera is mounted at one END of the court and shoots down its length — neither
   camera sees the full court well. The 1-camera fallback scenario is DROPPED.
   But cross-camera fusion (homography) is Phase 4. Until then, every component is built
   and validated on ONE feed at a time.
2. **Near-half ownership.** Each camera is authoritative ONLY for the half of the court
   nearest to it. Far-side players are small and their skeletons are unreliable — that is
   expected and NOT a bug to fix on this camera. The far half is the other camera's job.
3. **Jersey color is NOT a unique identity.** Real footage has two players in blue.
   Color is one weak anchor among several. Court position (via homography, Phase 4) is
   the strong ReID anchor. Never treat jersey color as a unique key.
4. **Court ROI filter is built early (Phase 1).** The camera sees spectators behind the
   glass, parked cars, chairs. Any detection whose box-center is outside the court polygon
   is discarded before tracking. This is geometric (cv2.pointPolygonTest), not AI.
5. **Measure failures, don't hide them.** Count ID swaps. Log bounding-box heights.
   Log per-frame confidence. The numbers tell us where to spend effort.

## Hardware & environment

- **Main dev machine: the GPU/CUDA laptop.** All detection, tracking, training happens
  here. A second CPU-only laptop exists — used later only to sanity-check worst-case FPS.
- **CUDA must be verified before any real work.** Install torch WITH the CUDA wheel, not
  the CPU build. `torch.cuda.is_available()` must print `True`. This is gate #1.
- Real court clips exist and are the ONLY footage we tune against. Never calibrate
  thresholds on generic YouTube padel videos.

## Tech stack (fixed for now)

| Component            | Choice                          | Notes                                   |
| -------------------- | ------------------------------- | --------------------------------------- |
| Detection + skeleton | YOLOv11n-pose (`yolo11n-pose.pt`) | nano model, COCO-17 keypoints, one pass |
| Tracking             | ByteTrack via `supervision`     | `sv.ByteTrack()`                        |
| Video / drawing      | OpenCV                          | `opencv-python`                         |
| Math                 | NumPy                           |                                         |
| Reports (later)      | `fpdf2` + json                  | Phase 2                                 |

Install (GPU laptop, CUDA wheel first — match the cuXXX tag to installed CUDA):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics supervision opencv-python numpy
```

## Target file structure

```
main.py            entry point, argparse (--source, --model, --show, --save-video)
config.json        camera sources, court polygon, net_x, thresholds
core/
  detector.py      YOLOv11-pose wrapper -> [{bbox, keypoints, conf}]
  tracker.py       ByteTrack wrapper -> detections with stable track_id
  pipeline.py      main loop: read -> detect -> ROI filter -> track -> draw -> show
utils/
  skeleton.py      COCO-17 skeleton drawing
  colors.py        per-player color palette
  roi.py           court polygon filter
output/            saved videos, reports, clips
```

Keep modules small and single-purpose. No god-files.

## Coding conventions

- Python 3, clear names, type hints on function signatures.
- Each `core/` and `utils/` module is independently importable and testable.
- All tunable values (confidence threshold, polygon, model path) come from `config.json`
  — never hard-code them in logic.
- Convert NumPy types to native Python before any JSON dump (NumpyEncoder later).

## Phase plan (high level — we go in order, gated)

- **Phase 1 (now):** detector -> ROI filter -> tracker -> skeleton, on one feed.
  Gate: clean detection of near players, spectators filtered out, ID-swap count measured.
- **Phase 2:** jersey color, enrollment, rule-based actions, form alerts, JSON/PDF report.
- **Phase 3:** ball (MOG2), clips, session comparison, timeline.
- **Phase 4:** camera calibration, homography, multi-camera threading + fusion.
- **Phase 5:** keypoint data collection, LSTM action model.
- **Phase 6:** TensorRT on Jetson, RTSP, GStreamer. ST-GCN optional.

**Current focus: Phase 1 only.**
