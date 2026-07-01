# Hard-Negative Pass — near-player false positives (careful, hand-verified)

**Goal:** teach the ball detector to stop firing on a player's **hand / racket / the fence behind
them** (the "ball glued to a standing player" FP), **without** repeating the last retrain that
*dropped* precision (0.76→0.61 / 0.66→0.48).

**Why the last one failed:** the miner flagged *real* balls (slow-apex → "static", briefly-seen →
"isolated") and those got labelled not-a-ball, so the net learned to suppress real balls. This pass
fixes that with four rules.

## The four safeguards (do not skip)

1. **Better target.** We mine only **lingering-near-player** detections: a drawn ball that sits
   inside a player's box for ≥5 frames with little movement. A *real* ball only crosses a player
   box for 1–3 frames, so this rarely catches a real ball. (`mine_near_player_fp.py`)
2. **Hand-verify every frame, biased to KEEP.** The miner only *proposes*. In review the default
   is "it's a real ball" — press **B (not-visible)** ONLY when it's clearly not a ball. Use **A /
   D** to step to neighbouring frames and check it doesn't *move like a ball* before you reject it.
3. **Separate file — never touch the eval labels.** Verified negatives are written to
   `output/hardneg_ball_<clip>.csv`, NOT `output/ball_labels_<clip>.csv`. Training merges the two;
   the **eval always runs on the untouched clean labels**, so our precision number stays honest and
   a baseline `visible=1` always overrides a proposed negative.
4. **Small, side-2 first, reversible.** Do one small batch on the weak camera, keep the old weights
   (`weights/ball_tracknet_v2.prev.pt`), and only keep the retrain **if the eval improves**.

## Step by step (on the pod)

Prereq: a fresh `python render_selector.py` so `output/clean_side*.json` exists (the miner reuses
that ball track — it does **not** re-run the ball detector).

```bash
# 1) MINE candidates (one pose pass; writes fp_near_player_<clip>.csv)
python mine_near_player_fp.py config-side2.json

# 2) REVIEW / hand-verify -> writes ONLY to the separate hardneg file
python main.py --config config-side2.json --label-ball \
    --label-from fp_near_player_side-2-full-vid.csv \
    --label-out output/hardneg_ball_side-2-full-vid.csv
#   In the window: A/D = step frames (check motion), B = confirm NOT a ball (hard-negative),
#   L-click/ENTER = it IS a ball (keep as positive), S = save, Q = quit. When unsure -> skip.

# 3) BACK UP the current (good) weights before training
cp weights/ball_tracknet_v2.pt weights/ball_tracknet_v2.prev.pt

# 4) RETRAIN (train_ball auto-merges output/hardneg_ball_*.csv as negatives; eval labels untouched)
python train_ball.py config-side2.json      # small run; watch the loss/val print

# 5) MEASURE on the CLEAN held-out labels — keep the retrain ONLY if precision holds/improves
python eval_selector.py config-side2.json output/ball_labels_side-2-full-vid.csv
#   want: precision UP, recall ~flat. If recall dropped -> we poisoned again:
cp weights/ball_tracknet_v2.prev.pt weights/ball_tracknet_v2.pt   # ROLL BACK
```

## Accept / reject gate

- **Keep** the new weights iff: side-2 precision **≥** its current ~0.79 **and** recall did **not**
  fall more than ~0.02 on the clean labels. Then re-render and eyeball that the near-player FPs are
  gone.
- **Roll back** (step 5's `cp`) on any recall drop — a precision gain bought with recall loss is the
  failure mode we're avoiding.

## Dials (`mine_near_player_fp.py`)

| const | default | effect |
|---|---|---|
| `MIN_LINGER` | 5 | frames stuck in a box to flag (lower = more candidates, more real-ball risk) |
| `STUCK_PX` | 120 | max movement across the run (higher = also flag balls that drift near a player) |
| `BOX_MARGIN` | 0.15 | player-box expansion for racket/hand reach |

## Files

- `mine_near_player_fp.py` — miner (proposals only).
- `core/ball_label.py` `--label-out` / `main.py --label-out` — review into a separate file.
- `train_ball.py` — merges `output/hardneg_ball_<clip>.csv` as negatives (baseline-visible wins).
- eval stays on `output/ball_labels_<clip>.csv` (the clean 2591-label baseline).
