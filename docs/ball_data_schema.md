# Ball data schema — for model optimization

Purpose: define the CSV columns the ball pipeline should emit so the data can be used to
**optimize two models**:

1. **Ball detector** (TrackNetV2, learned) — mine hard frames (misses / low confidence) to label and retrain.
2. **Shot classifier** (currently rule-based physics in `core/ball_events.py`) — build a feature+label
   table to train or validate a learned stroke classifier, or to tune the existing rules.

General rules for every CSV:
- Write a **header row**; put **units in the name** (`_m`, `_mps`, `_px`, `_s`, `_deg`).
- **One row per record**, sorted by `frame`, with a stable unique id.
- **Empty string for missing**, never `0` (0 is a real coordinate/speed).
- Ship a sidecar `meta.json` (video name, `fps`, width, height, frame_count, weights md5,
  heatmap_threshold, config files) so frames↔seconds convert cleanly and runs are reproducible.

---

## 1. `ball_track.csv` — per-frame (detector optimization + trajectory)

One row per frame per camera. Drives **hard-example mining** for TrackNet and feeds any
trajectory/physics model.

| Column | Unit | Source | Use |
|---|---|---|---|
| `frame` | int | loop counter | key |
| `t_s` | s | `frame / fps` | time join |
| `camera` | str | `"side-1"`/`"side-2"` | per-cam analysis |
| `detected` | 0/1 | `track.measured` | miss rate |
| `heatmap_peak` | 0–1 | `detector._infer().max()` | **hard-example mining**: low peak = weak frame to label/retrain |
| `conf` | 0–1 | tracker/detection conf | confidence calibration |
| `u_px`, `v_px` | px | detection pixel | re-projection, labeling overlay |
| `x_m`, `y_m`, `z_m` | m | homography / triangulation | court-space trajectory (z only when `source="both (3D)"`) |
| `source` | str | `both (3D)`/`side-1`/`side-2`/`lost` | which estimator placed the ball |
| `n_cams` | 1/2 | how many cams saw it | triangulation availability |
| `reproj_err_px` | px | triangulation residual | flag bad 3D points to drop or relabel |
| `gap_len` | int | frames since last detection | locate dropouts (occlusion vs true miss) |

**Detector workflow:** sort by `heatmap_peak` ascending and by `gap_len` descending → those are the
frames the model is unsure about or missing. Label those, add to the training set, retrain. This is
targeted labeling instead of labeling everything.

---

## 2. `ball_shots.csv` — per-event (shot-classifier optimization)

One row per detected event (`player_hit`, `net_hit`, `wall_bounce`, `fence_hit`, `floor_bounce`).
This is the **feature + label table** for a shot classifier.

### Identity / time
| Column | Unit | Notes |
|---|---|---|
| `event_id` | int | stable unique id per shot |
| `frame` | int | contact frame |
| `t_s` | s | `frame / fps` |
| `camera` | str | firing camera |
| `rally_id` | int | segment between serves / dead-ball gaps |
| `shot_index` | int | nth shot within the rally |

### Kinematic features (the strongest stroke cues)
| Column | Unit | Source | Discriminates |
|---|---|---|---|
| `speed_in_mps` | m/s | ball speed before contact | incoming pace |
| `speed_out_mps` | m/s | ball speed after contact | **power** |
| `speed_change` | px/s or m/s | `BallEvent.speed_change` (already computed) | smash/drive (adds energy) vs block |
| `turn_angle_deg` | deg | direction change at contact | hit hardness / deflection |
| `incoming_angle_deg` | deg | ball heading before | cross-court vs down-the-line |
| `outgoing_angle_deg` | deg | ball heading after | shot direction |

### Spatial features
| Column | Unit | Source | Discriminates |
|---|---|---|---|
| `x_m`, `y_m` | m | `BallEvent.x_m/y_m` (already computed) | court zone (service box / baseline / net) |
| `z_m` | m | triangulated height at contact | **overhead smash vs low volley vs groundstroke** |
| `dist_to_net_m` | m | derived from `y_m` | net play vs back court |

### Player context
| Column | Unit | Source | Discriminates |
|---|---|---|---|
| `player_id` | int | link to player tracker id | per-player stats, attribution |
| `hand` | str | `BallEvent.player_hand` (already computed) | forehand/backhand side hint |
| `player_x_m`, `player_y_m` | m | player court position at contact | server vs net player vs baseliner |
| `player_dist_to_ball_m` | m | derived | reach / contact quality |

### Detection quality (so you can trust/weight each row)
| Column | Unit | Notes |
|---|---|---|
| `n_cams` | 1/2 | 2 = triangulated (z trustworthy) |
| `ball_conf` | 0–1 | detector confidence around contact |

### Labels (targets)
| Column | Values | Notes |
|---|---|---|
| `type` | floor_bounce / wall_bounce / fence_hit / net_hit / player_hit | the **rule-based** label from `ball_events.py` — weak/auto label |
| `shot_label` | serve / forehand / backhand / volley / smash / lob / bandeja / "" | **ground truth — fill by hand** for a training set |
| `label_correct` | 0/1/"" | did the rule-based `type` match reality? → directly measures + tunes the rule classifier |

**Classifier workflow:**
- *Tune the existing rules:* fill `label_correct`, find where rules misfire, adjust the thresholds in
  `config['ball']['events']` (e.g. `hit_min_speed_px_s`, `net_turn_deg`) against this ground truth.
- *Train a learned classifier:* features above (kinematic + spatial + player) → `shot_label`. Even a
  small tree/boosting model separates smash/volley/groundstroke well once `z_m`, `speed_out_mps`,
  and `player_y_m` are present.

---

## Column availability (effort to produce)

- **Free now** (already on `BallEvent` / in the dual loop, just write them): `x_m, y_m, in_court,
  speed_change, t_s, source, heatmap_peak`.
- **Cheap** (data is nearby in `ball_dual.py`): `z_m, n_cams, reproj_err_px, gap_len, speed_in/out`.
- **Needs new wiring** (link to player tracker + arc segmentation): `player_id, player_x_m/y_m,
  rally_id, shot_index, incoming/outgoing_angle`.

Prioritize the **Free + Cheap** set first — it already unlocks detector hard-example mining and a
usable first-pass shot-feature table. Add the player/rally columns when you move to per-player stats
or a learned stroke classifier.
