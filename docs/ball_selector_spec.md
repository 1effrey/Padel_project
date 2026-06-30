# Fixed-Lag Buffered Ball Selector — Spec

**Purpose:** turn the noisy per-frame ball *detections* into an **accurate, clean per-frame 2D
ball track** suitable for shot classification — **without** losing recall and **without** breaking
the real-time requirement. This is the component that closes the gap between our causal tracker and
Fadi's clean output.

**One-line idea:** decide each frame's ball position with a **small lookahead** (a few frames),
using *support* (does a detection continue across neighbouring frames?) to reject ghosts. It runs in
real time with a small fixed latency (~0.25 s), not by processing the whole video.

---

## 1. Why this, and why it isn't what we have

- We already do **2D pixel tracking** (`core/ball_tracker.py`, a causal Kalman filter). The problem
  is not 2D vs 3D — it's that the tracker is **causal**: it must commit a position *this instant*,
  so an isolated false positive (a fence/limb blip) is indistinguishable from a real ball, and
  aggressive rejection (Phase 2) dropped real post-hit balls.
- Fadi's output is clean because his selection is **buffered**: it looks a few frames ahead to
  confirm a detection has **support** before committing. That rejects ghosts *without* losing real
  balls.
- **Real-time is satisfied** by a *fixed-lag* window (commit frame N−k once frame N is seen).
  k≈5 @ 20 fps = **0.25 s** constant latency. This is standard real-time fixed-lag smoothing; it works
  live on Jetson and on recorded clips identically. No whole-video processing.

## 2. Where it sits (pipeline)

```
detector (per cam, per frame)             [REUSE: core/ball_detector.py -> last_candidates]
        |  list of candidates {u,v,conf,area}
        v
FIXED-LAG BALL SELECTOR  (NEW: core/ball_selector.py)   <-- this spec
        |  ONE clean ball point per frame (committed k frames late) or "no ball"
        v
events / classification                   [REUSE: core/ball_events.py runs on the clean track]
```

The selector **replaces the candidate-selection responsibility** currently inside
`BallTracker.update_multi`. A light Kalman/EMA may still smooth the *accepted* points (optional), but
the *which-candidate-is-the-ball* decision moves here, where it has lookahead.

## 3. Interface

```python
class FixedLagBallSelector:
    def __init__(self, lag=5, max_step_px=350.0, min_support=2,
                 static_radius_px=20.0, gap_fill_max=5): ...

    def push(self, frame_idx: int, candidates: list[Cand]) -> Optional[BallPoint]:
        """Feed this frame's detector candidates. Returns the COMMITTED ball point for
        frame (frame_idx - lag), or None while warming up. Real-time safe: O(cand^2 * window)."""

    def flush(self) -> list[BallPoint]:
        """End-of-stream: emit the remaining buffered decisions (the last `lag` frames)."""
```
- `Cand`  = `{u, v, conf, area}`  (already produced by the detector)
- `BallPoint` = `{frame, u, v, conf, source}` where `source ∈ {detected, interpolated, none}`
- Per camera: one selector instance each (side-1, side-2). Owner-selection / near-half logic stays
  where it is and consumes these clean tracks.

## 4. Algorithm (v1 — support-counted; ship this first)

State: a rolling buffer of the last `2*lag+1` frames of candidates, plus the recently committed
points.

On `push(N, cands)`:
1. Store `cands` for frame `N`.
2. If the buffer doesn't yet span `[N-2*lag … N]`, return `None` (warming up).
3. Decide frame **t = N − lag** (it now has `lag` frames of *future* context):
   a. For each candidate `c` at `t`, compute **support**: the number of frames `f` in
      `[t-lag, t+lag]`, `f≠t`, that contain at least one candidate within a **motion tube** of `c`
      — i.e. within `max_step_px * |f-t|` px of `c` (the ball can move, bounded per frame).
   b. Pick the candidate with the **highest support** (tie-break by confidence, then proximity to the
      last committed point).
   c. **Accept** it iff `support ≥ min_support`. Else → frame `t` has **no ball** (it was an isolated
      blip → a ghost, correctly dropped).
   d. **Static reject:** if the matched candidates across the window all lie within
      `static_radius_px` (the "path" doesn't move) → reject (a held ball / light / logo) → no ball.
4. **Gap fill:** a run of `none` frames bounded on both sides by `detected` points and no longer than
   `gap_fill_max` → linearly interpolate, tag `source=interpolated`. Longer gaps stay `none` (break).
5. Return the committed `BallPoint` for `t`.

This is exactly the **isolated / static / support** logic Fadi uses (`fp_isolated`, `fp_static`,
`recovered`, `inpaint`), but as a real-time fixed-lag pass. It is cheap (a handful of candidates, a
~11-frame window).

**v2 (later, only if v1 isn't clean enough):** replace step 3 with proper **tracklet linking** —
build smooth multi-frame paths through the window (greedy or Hungarian on a motion-cost), pick the
candidate on the strongest path through `t`. More robust through crossings; more code.

## 5. Parameters (the dials)

| Param | Start | Trades |
|---|---|---|
| `lag` | 5 frames (0.25 s) | latency ↔ cleanliness/recovery |
| `max_step_px` | 350 (4K) | reject teleports ↔ keep fast smashes |
| `min_support` | 2 | reject ghosts ↔ keep briefly-seen real balls |
| `static_radius_px` | 20 | reject held/static balls ↔ keep slow apex balls |
| `gap_fill_max` | 5 | bridge dropouts ↔ don't invent long arcs |

All come from `config['ball']['selector']` — never hard-coded.

## 6. What to REUSE vs BUILD

- **Reuse:** the detector + weights (0e82cbf), the 2,591 labels, the precision harness, the event
  detector (`ball_events.py`), the configs/calibration, the per-run output + scoring scripts.
- **Build:** `core/ball_selector.py` (this spec) + wire it into the dual loop in place of
  `update_multi`'s candidate picking.
- **Strongly consider:** porting Fadi's `ball_selected_state` selection logic instead of
  re-deriving — it is this component, already tuned. He's on the team.
- **Drop from the classification path:** triangulation / EKF / homography court map (not needed for a
  2D track; keep only as a separate optional enrichment).

## 7. Acceptance gates (the finish line — define "done")

Measure on the held-out labels with the existing harness; **A/B against the current causal tracker:**
1. **Detection quality:** precision **↑** and recall **held** (target: ≥ current 0.76 / 0.66 F1, with
   FP-rate down). The whole point is precision-up-without-recall-loss.
2. **Track cleanliness:** count of large frame-to-frame jumps (`>max_step_px`) **drops sharply** vs
   the causal track (this is the zigzag/teleport metric).
3. **Visual:** on the spot-check clips (frames ~2000, ~2600, the zigzag frame) the trail follows the
   real ball — no fence teleports, no zigzags, breaks cleanly on true gaps.
4. **Latency:** confirmed fixed at `lag` frames; pipeline still streams (no whole-video dependency).

When 1–3 pass and 4 holds → the **ball track is done**; move to the classifier on top of it.

## 8. Phased build (gated, finishable)

- **P0 — Lock foundation:** good weights restored (0e82cbf) + bad FP labels reverted → clean
  0.76/0.66 baseline. *(in progress)*
- **P1 — Selector v1:** implement `FixedLagBallSelector` (support-counted) + unit tests (synthetic:
  smooth ball passes, isolated blip rejected, static blob rejected, post-hit reversal kept, short gap
  filled). Gate: unit tests pass.
- **P2 — Wire into dual loop** behind a config flag (`ball.selector.enabled`), default off. A/B run
  baseline vs selector; score with the harness + visual. Gate: acceptance gates §7.
- **P3 — Tune the dials** on real footage to the gates; then enable by default.
- **P4 (optional) — v2 tracklet linking** only if v1 misses the bar on crossings.

## 9. Risks / open questions

- **Crossings / two near balls** (rare in singles): v1 support-counting may pick the wrong one; v2
  tracklet linking handles it. Acceptable for v1.
- **Very fast smashes** could exceed `max_step_px` for one frame → tune up, or scale the tube by the
  recent speed.
- **`lag` vs product latency:** confirm 0.25 s is acceptable to stakeholders (it is for shot/bounce
  analytics; confirm there's no sub-100 ms requirement).
- **Decide reuse-Fadi vs build-new** before P1 — porting his tuned selection is likely faster than
  re-deriving.

---

**Bottom line:** this is a small, real-time-safe, finishable component — a fixed-lag buffered selector
between the detector and events — that delivers an *accurate* 2D ball track by checking support over a
~5-frame window. It reuses everything that already works, drops the 3D complexity, and has concrete
accept gates so it actually ends.
