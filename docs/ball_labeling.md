# Ball detection — labeling & training requirement (Phase 1)

**This is the gate-blocker for Phase 1.** `core/ball_detector.py` is a *supervised*
TrackNetV2 network. The architecture and interface are complete and runnable, but
the model is useless without **trained weights**, and **no padel-ball weights exist
yet**. Until a weights file is present, the detector runs in **stub mode** (returns
"no ball" on every frame, loudly flagged). This document is how we get from stub to
a real detector.

---

## 0. The tools (all built — no external setup)

| Goal | Command | File |
| --- | --- | --- |
| See a ball **now**, no data | set `ball.method:"motion"`, then `python main.py --ball-eval --source <clip> --save-video` | `core/ball_detector_motion.py` |
| Label frames | `python main.py --config <cfg> --label-ball` | `core/ball_label.py` |
| Verify the trainer first | `python train_ball.py --smoke` (random data), then `--dry-run` (your data) | `train_ball.py` |
| Train real weights | `python train_ball.py --config config-side1.json --config2 config-side2.json` | `train_ball.py` |
| Measure the result | `python main.py --ball-eval --source <clip> --save-video` | `core/ball_eval.py` |

The **motion baseline** (`ball.method:"motion"`) needs no labels or weights — use it for
immediate feedback and to bootstrap labels while you train the real TrackNetV2 detector
(`ball.method:"tracknet"`, the default). It is noisier (picks up limbs/shadows), so it is
a stopgap, not the final detector.

---

## 1. Why we can't just download weights

Public TrackNet weights exist for **tennis** and **badminton**, not padel. They will
detect *something* on our footage but poorly: the padel ball is a different size on
screen, a different colour, on a different court, under our specific lighting and our
**20 fps** motion blur. The honest path to a working detector is:

1. **Label** padel-ball frames from *our two cameras*.
2. **Fine-tune** TrackNetV2 on them (optionally starting from public tennis weights
   as initialization to converge faster).

We tune on **our** clips only — same rule as the rest of the project (never calibrate
on generic YouTube padel videos).

---

## 2. What to label

For each frame, the ball is **one point** `(u, v)` in full-resolution pixels, or
**absent** (occluded / out of frame). That's it — no boxes, no class.

**Label schema (one CSV per clip):**

| column    | meaning                                                    |
| --------- | ---------------------------------------------------------- |
| `frame`   | 0-based frame index in the clip                            |
| `visible` | `1` if the ball is visible this frame, else `0`            |
| `u`       | ball x in full-res pixels (blank when `visible=0`)         |
| `v`       | ball y in full-res pixels (blank when `visible=0`)         |

This matches the public TrackNet "Label_csv" format, so existing tooling/scripts and
the public datasets line up with ours.

**Label the hard cases on purpose** — these are what make or break the detector:
- serves and **smashes** (the fastest, most blurred balls),
- the ball **near vs far** from each camera (size changes a lot),
- **occlusion** (behind a player, crossing the net) — label these `visible=0` so the
  network learns that "no ball" is a valid, correct answer,
- **glass-wall** situations (the ball near/against the back and side glass).

## 3. How much to label

| Split | Frames (rough) | Notes                                                |
| ----- | -------------- | ---------------------------------------------------- |
| Train | 3,000–8,000    | across BOTH cameras, multiple rallies, varied speed  |
| Val   | 800–1,500      | held-out rallies (not just held-out frames)          |

You do not have to label every frame of a clip — sample rallies. Balance "ball
visible" and "ball absent" frames so the network doesn't learn to always fire.

**Suggested tooling:** a simple click-to-label viewer (one click = ball point, a key
= "not visible", auto-advance). The repo already has the click-handling pattern in
`main.py` (`calibrate_roi` / `calibrate_homography`) — a `--label-ball` tool can be
built the same way in a later step. CVAT or the public TrackNet labeling tool also
export the CSV above.

---

## 4. How the labels become training targets

TrackNetV2 predicts a **heatmap**, not a coordinate. Convert each labeled `(u, v)`
into a target heatmap the same size as the model output (default 288×512, so scale
the point down from full-res first):

- a 2D **Gaussian** blob (σ ≈ 2–3 px at network resolution) centered on `(u, v)`,
  values in `[0, 1]`;
- an **all-zeros** heatmap for `visible=0` frames.

The model input is the **stack of `in_frames` consecutive frames** (default 3, RGB,
resized to 512×288, `/255`), oldest first, the labeled frame last — identical to what
`BallDetector` feeds at inference, so train/infer preprocessing must stay in sync.

## 5. Training recipe (starting point)

| Item          | Suggested setting                                                 |
| ------------- | ----------------------------------------------------------------- |
| Loss          | weighted **BCE** on the heatmap (or focal loss; ball pixels are rare) |
| Optimizer     | Adam (Adadelta also works for TrackNet), LR ~1e-3                  |
| Init          | random, **or** public tennis TrackNetV2 weights for warm start    |
| Augmentation  | small brightness/contrast jitter; avoid geometric warps that move the ball off its label |
| Input size    | 512×288 (`config["ball"]["input_width"/"input_height"]`)          |
| Epochs        | until val detection-rate / localization error plateaus            |

Save the trained weights as a plain `state_dict` (or a checkpoint dict with a
`state_dict` key — the loader accepts both) and point `config["ball"]["weights"]`
at it (e.g. `weights/ball_tracknet_v2.pt`).

## 6. How we measure success (the gate)

Once weights exist, run the harness on real footage — no code changes:

```
python main.py --ball-eval --source side-1-full-vid.mp4 --save-video
```

It writes `output/ball_eval_metrics.json` with:
- **detection_rate** (fraction of non-warmup frames with a ball),
- **confidence** (mean/median/min/max of heatmap peaks on hits),
- **longest_no_ball_run** (occlusion behavior — should show occasional gaps, not one
  endless gap and not zero gaps).

That is the Phase-1 acceptance evidence: a measurable detection rate, reported
confidence, and correct "no ball" behavior under occlusion.

---

## 7. Data realities to keep in mind (they constrain later phases, not labeling)

Labeling is pure 2D `(u, v)` per camera — straightforward. The *hard* part lives
downstream and is why later phases use physics, not just the cameras:
- the cameras measure ball **X** (court width) well, **Y** (length) poorly, and
  height **Z** essentially not at all (Phases 4–5 infer Y and Z from a projectile
  model anchored on floor bounces);
- 20 fps **undersamples** fast shots; **spin** is unobservable; **glass-wall**
  bounces are legal and must be handled as events (Phase 3).

None of that changes what you label here — it just sets expectations for what the
full 3D system can and cannot promise.
