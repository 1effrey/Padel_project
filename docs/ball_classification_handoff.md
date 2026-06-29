# Ball Shot-Classification — Approach, Rationale & Model Comparison (Agent Handoff)

> Purpose: hand a complete picture to another agent. It explains **what we decided, why, and every
> fact needed to continue** — the two ball-tracking systems we have, how we reverse-engineered the
> manager's, the geometric reasoning that drove the decision, and the concrete plan + data schema.
> Read top-to-bottom once; the reference sections at the end are for lookup.

---

## 0. TL;DR — the decision

**Drop the 3D ball *mapping*. Do shot classification in 2D image space only.**

- The product goal was always **shot classification** (bounce / wall / net / fence-OUT / player-hit, in/out, who hit). The 3D ball mapping (top-down court position + height) was a *means*, not the goal.
- Single-camera 3D is **geometrically ill-posed** and was our biggest accuracy problem. Classification, however, reads **motion (pixel velocity)**, which a single camera gives perfectly well.
- So we remove the unreliable part (single-cam 3D projection / triangulation-for-position / physics) and keep the reliable part (2D motion → events). The manager's ("Fadi") system already proves this 2D approach works on our exact footage.
- We keep the homography for **one** valid job: in/out + landing position **at floor bounces** (z=0 is true there).

This converges our design with the manager's proven 2D approach while keeping our richer event taxonomy and pose-based player attribution.

---

## 1. Project context

- **Product:** real-time padel CV for InMobiles (Mkalles, Lebanon), runs on Jetson in prod.
- **Cameras:** TWO cameras, each mounted at one END of the court shooting down its length. Two cameras IS the product; single-stream is only for dev. Each camera is authoritative for its **near half** ("near-half ownership"); the far half is small/unreliable and is the other camera's job.
- **Repo:** branch `combobulator`. Pod path `/workspace/combined/Padel_project`; laptop `C:\Users\User\Desktop\Combined Version`. Code moves by git; data by rclone (`gdrive:Padel/`). Weights are NOT in git.
- **Stack:** YOLOv11-pose (players+skeleton), ByteTrack via `supervision` (tracking), **TrackNetV2** (`weights/ball_tracknet_v2.pt`) for the ball, OpenCV, NumPy. Configs drive all tunables.
- **Team:** three junior AI interns; quality > speed; gated phase-by-phase workflow (build → measure on real footage → ask before next phase).

---

## 2. The two ball systems we have

### System A — our pipeline (`core/ball_dual.py`, the `--ball-dual` mode): GEOMETRIC 3D
Per frame, with the two cameras synced by a fixed frame offset:
- Detect ball in each camera (TrackNet) → track (Kalman).
- **Placement:** if BOTH cameras see it → **triangulate** → real `(x,y,z)` in court metres (`source="both (3D)"`). Else **single-camera homography floor-projection** onto z=0, kept only if it lands on/near court (`source="side-1"/"side-2"`).
- Per-camera event detectors (`evA`,`evB`) emit events; near-half ownership + cross-camera dedup.
- Output: side-1 | side-2 | **top-down court map**, plus CSVs. Also has a richer path (`--ball-3d`) with physics/EKF trajectory (`core/ball_physics.py`, Phase 5).

**Strength:** true 3D — court position, height, trajectory, in/out in metres.
**Weakness:** the single-camera floor-projection is **wrong for an airborne ball** (z=0 is false in the air) → the top-down map is inaccurate whenever one camera only. In a representative 600-frame run, ~73% of placed frames were single-camera (see §4), so most map points were unreliable.

### System B — the manager's "Fadi" model (reverse-engineered, see §5): 2D OWNER-SELECTION
Pure 2D image-space, no triangulation, no projection:
- Detect per camera (TrackNet, 1080p, top-3 candidates).
- A **selection state machine** rejects false candidates ("ghosts"), recovers after misses, and picks an **owner camera** per frame by **near-half ownership done in IMAGE space** (a horizontal net-line Y threshold). The ball position = the owner camera's raw pixels, verbatim.
- **Inpaint** short detection gaps (≤10 frames) by interpolation; longer gaps left blank.
- Output: two **stacked 2D camera feeds** with the ball marked + event banners ("BOUNCE"). No court map, no metres, no height.

**Strength:** very robust 2D track + events; **no projection, so no airborne error**. Mature, tuned, works on the full video.
**Weakness:** gives NO court position / height / 3D — only per-camera pixels.

### Head-to-head

| Dimension | System A (ours, `ball_dual`) | System B (Fadi) |
|---|---|---|
| Output space | 3D court metres (+ height) | 2D image pixels, per camera |
| Both cams see ball | triangulate (correct) | pick owner, use its pixels |
| One cam sees ball | homography floor-project z=0 (**inaccurate airborne**) | owner = that camera's pixels (no projection) |
| Neither | drop | inpaint ≤10 frames, else blank |
| Ownership boundary | court metres (homography) | **image net-line Y threshold (no homography)** |
| Sync | fixed frame offset (config) | fixed 37-frame offset |
| False-positive handling | ROI + tracker | explicit **ghost** rejection (dropped ~12k on cam2) |
| Event taxonomy | RICH: `player_hit`(+hand via pose), `net_hit`, `wall_bounce`, `fence_hit`=OUT, `floor_bounce` | only "BOUNCE" demonstrated |
| 3D / court map / height | YES | NO |
| Maturity | mid-dev, branch in flux | finished, tuned, robust |

**Verdict:** B is the more robust, mature **2D** base; A is more ambitious (3D) but inherits the single-camera projection problem. Neither is "wrong" — different points on robustness-vs-richness.

---

## 3. Why this solution (the reasoning chain)

1. **Single-camera 3D is ill-posed, not buggy.** One camera + one pixel = a *ray*, not a point. Floor projection picks z=0 as the missing constraint; that is correct only for a grounded ball and systematically wrong for an airborne one. **No projection/homography trick fixes this** — the depth information isn't in a single frame. The only ways to recover it are *time + physics* (gravity model / EKF, anchored on bounces) or *learning* (a monocular predictor trained on stereo or synthetic labels). Both are real but heavy; physics is half-built (`ball_physics.py`), learning needs data and is camera-specific (external datasets do NOT transfer because the image→height mapping depends on this camera's exact geometry).
2. **Two cameras exist precisely because of (1).** The single-camera fallback is inherently limited; the product is dual-camera.
3. **Classification doesn't need 3D.** Events are velocity changes — a bounce is a vertical pixel-velocity sign flip, a hit is a trajectory kink + speed gain, wall/net/fence are deflections inside image-space regions. Our own `core/ball_events.py` already detects from the **2D track's velocity**; the homography is only used to *attach* court metres + in/out, not to detect. In/out at a bounce is the one place projection is valid (ball on floor).
4. **Therefore:** if the goal is classification, dropping the 3D mapping removes the exact part that's unreliable and keeps everything classification needs. It also deletes our hardest, most error-prone code and converges with the manager's proven approach.

---

## 4. Key finding A — the weights bug (root-cause story; important lesson)

Symptom: `--ball-dual` detected the ball in **0 of ~3196 frames** on the pod, while the same code worked on a laptop. We isolated it methodically:
- Ruled out FP16 vs FP32 (no difference).
- Ruled out torch version (the working laptop runs torch **2.12.0+cu130**, NEWER than the pod's 2.8.0 — so "new torch breaks it" was false).
- Diagnostic scripts (`diag_ball.py`, `diag2_ball.py`) printed the raw TrackNet heatmap peak per frame. Result: with the pod's weights, `global_max = 0.484` across the whole video — **never** reached the `heatmap_threshold = 0.5` → 0 detections.
- **Root cause:** the pod had a **different, undertrained checkpoint.** MD5s: GOOD (laptop) `28a9fd05abe24f24946480db27a43df7`; BAD (pod) `a0d9c3e019f1b641f802580662e219e3`. Both ~40.35 MiB.
- After copying the good weights: `global_max` 0.484→**0.826**, frames≥0.5 0→**2379/3196**, cross-cam connected 11%→**62%**, hits 0→**7 player / 4 net**, bounces 0→**4** (600-frame run: both-cam 102, side-1-only 206, side-2-only 67).

**Lesson for the next agent:** if the ball pipeline silently finds nothing, FIRST `md5sum weights/ball_tracknet_v2.pt` against the good hash above. Do NOT lower `heatmap_threshold` to compensate — that masks a bad checkpoint.

---

## 5. Key finding B — reverse-engineering the manager's model

Source files: `C:\Users\User\Desktop\Combined Version\Fadi Ball Tracking\` — a stacked-2D video + three CSV folders (`ball_before_filtering`, `ball_after_filtering`, `ball_after_inpaint`). No code, only outputs. All conclusions are evidence-backed from the data:

- **Detection:** TrackNet per camera at **1080p**, `top_k=3` candidates, `cand_min_score=0.3`, 512 heatmap, fps≈20. Candidate format `{rank,x,y,score,area}`. Multiplicity (side-1): 0 cands 20%, 1: 34%, 2: 25%, 3: 21% → the ball is genuinely ambiguous often, hence a selection stage.
- **Sync:** fixed offset, `cam2_frame = cam1_frame + 37`, constant for all 28,749 rows.
- **Selection state machine** (`ball_selected_state.csv`): columns include `cam{1,2}_rank/side/main/ghost/recovered`, `conf_side`, `owner_cam`, `ball_x/y`.
  - **owner_cam:** cam1 10,560 / cam2 7,540 / NONE 10,649; switches only **8.8%** of frames (stable).
  - **`ball_x/y` = owner camera's pixels, verbatim** (verified 100% match — no blending, no projection, no triangulation).
  - **Rank selection:** rank-1 chosen 94%, but overrides to rank-2/3 in **6%** → trajectory-aware, not blind trust.
  - **Ghost rejection:** cam2 dropped **12,361** ghosts (false positives).
  - **Owner switches are recovery-driven:** 39% of switches coincide with a `recovered` flag, 10% with `ghost`.
  - **★ Ownership is an IMAGE net-line, no homography.** `side` splits cleanly in **Y** (cam1: side-1 `y[4-915]` vs side-2 `y[919-1069]`; cam2 mirrored) and overlaps fully in X. So "which half" = which side of a horizontal net-line at ~**y917 (cam1) / y885 (cam2)**. Each camera owns the large near-half region; the compressed far band is the other camera's.
- **Inpaint** (`ball_inpainted.csv`, tags `detected`/`inpaint`/`none`): inpaint runs are **hard-capped at 10 frames**; longer gaps stay `none` (up to 327–359). So only ≤0.5 s dropouts are bridged; no hallucinating across long absences.
- **Output:** stacked 2D feeds + event banners. No top-down map.

**How they classify shots with only pixels:** they read **motion**. A bounce = vertical pixel-velocity flip; the owning (near) camera sees the ball large/clear, so its 2D velocity signal is clean — ideal for event detection. In/out can be done with an image-space court polygon. They simply never attempt single-camera 3D, which is why their tracking is artifact-free.

---

## 6. The recommended solution

Build **robust 2D shot classification**, leading with OUR architecture (keep `ball_events.py`, our taxonomy, our dual loop, pose-based player hits) and grafting only the manager's **track-cleaning** ideas where our pipeline measurably falls short. ~80% ours, ~20% his.

### Graft plan (phased, gated — see also the standalone prompt)
- **Phase 0 — baseline & instrumentation.** Log event counts, owner-switch false-event rate, gap stats. No behavior change.
- **Phase 1 — image-space ownership gate** (`ball_dual.py` arbitration ~L217–237): gate events by a per-camera **net-line Y threshold** (config-driven; ref y917/y885) instead of/in addition to the homography near-half test. Highest value, lowest risk.
- **Phase 2 — ghost rejection + rank-override** (`ball_tracker.py`/`_build_tracker`): reject trajectory-inconsistent candidates; override to lower-ranked when it fits; recover after misses.
- **Phase 3 — short-gap inpaint** (≤10 frames, tag inpainted, never inpaint across a suspected event): trickiest, do last, only if gaps still hurt recall.
- **Bounce enrichment:** for `floor_bounce` ONLY, project the landing pixel via homography → court `(x_m,y_m)` + in/out (z=0 valid). The sole surviving use of geometry.

**Hard rules:** keep one event detector per camera in its own pixel frame; **never merge cameras' pixels** (false events at handoffs); `ball_events.py` logic unchanged. **Validate** against `Fadi Ball Tracking/` (note: it validates BOUNCES only — our richer types are our own burden). Gate each phase on real footage; skip a phase that doesn't beat baseline.

### Data schema for model optimization (classification-only)
Two CSVs + a `meta.json` sidecar; header rows; units in names; empty-for-missing (never 0).
- **`ball_track.csv`** (per frame): `frame, t_s, owner_cam, ball_u, ball_v, src(detected/inpaint/none), heatmap_peak, conf, vx, vy, speed(px/s), gap_len`. Drives **detector hard-example mining** (sort by `heatmap_peak`↑, `gap_len`↓ → label/retrain those) and supplies motion features. NO `x_m/y_m/z_m`.
- **`ball_shots.csv`** (per event): identity (`event_id, frame, t_s, camera, rally_id, shot_index`); kinematics in **px** (`speed_in_pxs, speed_out_pxs, speed_change_pxs, turn_angle_deg, incoming/outgoing_angle_deg`); image pos (`u_px, v_px`); player context (`player_id, hand, player_u_px, player_v_px, player_dist_px`); **bounce-only** court (`bounce_x_m, bounce_y_m, in_court` — filled only for `floor_bounce`); quality (`ball_conf`); labels (`type` weak/auto, `shot_label` hand-filled GT, `label_correct` 0/1). Speeds are px/s (no metric scale without mapping).

Use: detector via `heatmap_peak`/`gap_len`; shot classifier via the feature table → `shot_label`, with `label_correct` tuning the rule thresholds in `config['ball']['events']` (`hit_min_speed_px_s`, `net_turn_deg`, `player_contact_px`, …).

---

## 7. What is expected to work (honest calibration)

- **Robust 2D shot classification: yes** — incremental hardening of a working base, proven in principle by Fadi on the same footage. High confidence on Phase 1, good on Phase 2, cautious on Phase 3.
- **It does NOT fix 3D mapping** — that's deliberately dropped, out of scope.
- **Risks / unknowns:** (a) our event RULES still need tuning to our footage — clean input + bad thresholds = wrong events; (b) Fadi's reference validates only bounces, not our full taxonomy (esp. pose-based `player_hit`); (c) the image net-line assumes a near-constant net Y — **must be checked on OUR camera frames**, the net may slant/curve and need a polygon; (d) possible redundancy — our tracker+ROI already suppress some ghosts, so Phase 2/3 may add little. The phase gates de-risk all of this: anything that doesn't beat baseline is skipped.

---

## 8. Reference — concrete facts the next agent needs

**Paths:** pod repo `/workspace/combined/Padel_project` (branch `combobulator`); laptop repo `C:\Users\User\Desktop\Combined Version`; manager files `C:\Users\User\Desktop\Combined Version\Fadi Ball Tracking`. Old abandoned pod path: `/workspace/3D_Kris` (do not use).

**Weights:** `weights/ball_tracknet_v2.pt`; GOOD md5 `28a9fd05abe24f24946480db27a43df7` (~40.35 MiB). Not in git — transfer via `gdrive:Padel/transfer/`.

**Configs:** `config-side1.json` (camera A — has the `sync` block + homography + `court.{polygon,net_polygon,walls,fence}`), `config-side2.json` (camera B — homography + court boundaries; net_polygon present). Plain `config.json` lacks homography/sync — don't use it for dual. `heatmap_threshold` should be `0.5`. `ball.fp16` true/false is irrelevant (no effect).

**Key commands (pod, bash):**
```
export RCLONE_CONFIG=/workspace/rclone.conf            # each new session
python main.py --config config-side1.json --config2 config-side2.json --ball-dual --save-video [--max-frames 600]
python main.py --config config-side1.json --ball-eval --max-frames 300      # detector quality gate
rclone copy output/ball_dual.mp4 gdrive:Padel/ --progress
```
Calibration (needs a DISPLAY — run on laptop, not headless pod): `--calibrate-roi`, `--calibrate-homography`, `--verify-homography`, `--calibrate-walls`, `--calibrate-fence`, `--calibrate-net`, `--sync-manual`.

**`ball_dual` outputs** (in `output/`, ALL three CSVs written unconditionally; only the video is gated by `--save-video`): `ball_dual.mp4`, `ball_dual_locations.csv` (`frame, cam1_detected, cam1_x_px, cam1_y_px, cam2_detected, cam2_x_px, cam2_y_px, court_x_m, court_y_m, court_z_m, source`), `ball_dual_hits.csv` (`frame, camera, type, u_px, v_px, hand`), `ball_dual_bounces.csv` (`frame, camera, court_x_m, court_y_m, in_court`).

**`core/ball_events.py`** — `BallEvent` fields: `frame, type, u, v, x_m, y_m, in_court, player_hand, speed_change`. Types: `floor_bounce | wall_bounce | fence_hit | hit | player_hit | net_hit`. Detection is from the smoothed 2D velocity; `x_m/y_m/in_court` filled only when a homography is available. Strict player-hit rule uses pose wrists (`player_contact_px`, `player_pass_frames`). Boundary polygons read from `court.net_polygon`, `court.walls`, `court.fence`.

**Codebase map:** `core/ball_detector.py` (TrackNetV2 wrapper, `_infer()` returns the heatmap, `last_candidates`), `core/ball_tracker.py` (Kalman), `core/ball_eval.py` (`_build_detector/_build_tracker/_build_events`, `--ball-eval` gate), `core/ball_dual.py` (the dual view), `core/ball_events.py` (classifier), `core/camera_calib.py` (`build_camera`, `triangulate_ball`), `core/ball_physics.py` (Phase 5 EKF, half-built), `core/ball_3d.py` (integrated 3D), `core/calibrate_boundaries.py` (wall/fence/net click tools), `utils/homography.py`, `utils/video_io.py` (threaded reader; GPU decode falls back to CPU — pip OpenCV has no NVDEC).

**Diagnostics that exist:** `diag_ball.py` (per-frame `heatmap_max`), `diag2_ball.py` (whole-video `global_max` + threshold histogram). Untracked; safe to keep.

---

## 9. Open decisions / next actions for the agent

1. **Product scope is decided: classification-only, drop 3D mapping.** Proceed on that basis. (If anyone later needs continuous 3D, the path is dual-cam coverage + `ball_physics.py` EKF, NOT single-cam projection.)
2. **First concrete experiment:** Phase 0 + Phase 1 on one clip — measure owner-switch false-event rate before/after the image net-line gate. Small, reversible, tells you within a day whether the core idea holds on our footage.
3. **Verify the net-line assumption** on our actual camera frames (is a flat Y threshold valid, or is a polygon needed?).
4. **Tune `config['ball']['events']` thresholds** against labeled shots using the `ball_shots.csv` `label_correct` column.
5. Keep `ball_events.py` logic intact; graft only track-cleaning where it beats baseline.

— End of handoff —
