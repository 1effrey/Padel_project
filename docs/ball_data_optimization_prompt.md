# Task prompt: enrich the ball CSV outputs for model optimization

You are working in the padel CV repo (branch `combobulator`). The dual-camera ball pipeline in
`core/ball_dual.py` (entry: `python main.py --ball-dual --config config-side1.json --config2
config-side2.json --save-video`) writes three CSVs into `output/`:
- `ball_dual_locations.csv` (per frame)
- `ball_dual_hits.csv` (per event)
- `ball_dual_bounces.csv` (per floor bounce)

These currently drop most of the signal. Enrich them so the data can be used to **optimize two
models**:
1. the **TrackNetV2 ball detector** (learned) — via hard-example mining;
2. the **shot classifier** (currently rule-based in `core/ball_events.py`) — via a feature+label table.

Do **not** change detection/classification behavior or the video output — only widen the CSV schema
and add a metadata sidecar. Keep the run reproducible and byte-identical aside from the new columns.

## Constraints
- Every CSV gets a header row; put units in column names (`_m`, `_mps`, `_px`, `_s`, `_deg`).
- One row per record, sorted by `frame`, with a stable unique id (`event_id` for hits/bounces).
- Missing value = empty string, never `0` (0 is a real coordinate/speed).
- Add a sidecar `output/ball_dual_meta.json`: `{video_a, video_b, fps, width, height, frame_count,
  weights_md5, heatmap_threshold, config_a, config_b}` so frames↔seconds convert and runs are traceable.

## Columns to add

### A. `ball_dual_locations.csv` → detector optimization (per frame, per camera)
Free/cheap — the values are already in the dual loop:
- `t_s` = frame / fps
- `heatmap_peak` (0–1): raw `detector._infer().max()` for that camera's frame — **the key hard-example
  mining signal**; low peak = a frame the model is unsure about.
- `conf` (0–1): detection/tracker confidence
- `x_m, y_m, z_m`: court metres; `z_m` only when `source == "both (3D)"` (triangulated)
- `n_cams` (1/2), `reproj_err_px`: triangulation residual (flag bad 3D points)
- `gap_len`: frames since the last detection on that camera (locate dropouts)

### B. `ball_dual_hits.csv` → shot-classifier optimization (per event)
Already computed on `BallEvent` (`core/ball_events.py`, fields `x_m, y_m, in_court, player_hand,
speed_change`) but dropped at the `writerow` — write them, plus the cheap derived ones:
- Identity/time: `event_id`, `t_s`, `rally_id` (segment between serves/dead-ball gaps), `shot_index`
- Kinematics: `speed_in_mps`, `speed_out_mps`, `speed_change`, `turn_angle_deg`,
  `incoming_angle_deg`, `outgoing_angle_deg`
- Spatial: `x_m`, `y_m`, `z_m` (contact height — separates smash/volley/groundstroke),
  `dist_to_net_m`
- Player context: `player_id` (link to the player tracker id), `hand`, `player_x_m`, `player_y_m`,
  `player_dist_to_ball_m`
- Quality: `n_cams`, `ball_conf`
- Labels: `type` (existing rule-based label = weak label), plus two empty columns for ground truth:
  `shot_label` (serve/forehand/backhand/volley/smash/lob/bandeja) and `label_correct` (0/1)

### C. `ball_dual_bounces.csv`
- Add `event_id`, `t_s`, `rally_id`, `z_m` (≈0 at floor), keep existing `court_x_m, court_y_m, in_court`.

## Effort tiers (do in this order)
1. **Free now** (already on `BallEvent` / in the loop, just write): `x_m, y_m, in_court, speed_change,
   t_s, source, heatmap_peak`.
2. **Cheap** (data nearby in `ball_dual.py`): `z_m, n_cams, reproj_err_px, gap_len, speed_in/out_mps`.
3. **Needs wiring** (player tracker link + arc segmentation): `player_id, player_x_m/y_m, rally_id,
   shot_index, incoming/outgoing_angle_deg`.

Ship tier 1+2 first (unlocks detector hard-example mining + a usable shot-feature table); add tier 3
when moving to per-player stats or a learned stroke classifier.

## How the data is used (so you keep the right fields)
- **Detector:** sort `ball_dual_locations.csv` by `heatmap_peak` ascending and `gap_len` descending →
  those frames are the labeling/retraining targets.
- **Shot classifier:** `ball_dual_hits.csv` features (kinematic + spatial + player) → `shot_label`;
  use `label_correct` to measure and tune the rule thresholds in `config['ball']['events']`
  (e.g. `hit_min_speed_px_s`, `net_turn_deg`, `player_contact_px`).

## Acceptance
- All three CSVs have headers + the new columns; `ball_dual_meta.json` is written.
- A run on `--max-frames 600` produces non-empty `x_m/y_m/z_m/speed_*/heatmap_peak` where applicable.
- No change to detection counts, classification, or the rendered video vs the pre-change run.
