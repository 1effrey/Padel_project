# weights/

Drop trained model weights here. They are **not** checked into git (binary, large).

## Ball detector (Phase 1)

`core/ball_detector.py` looks for the file named in `config["ball"]["weights"]`,
which defaults to:

```
weights/ball_tracknet_v2.pt
```

Until that file exists, the ball detector runs in **stub mode** (returns "no ball"
on every frame, with a loud warning) — the player pipeline is unaffected either way.

The file must be a TrackNetV2 `state_dict` (or a checkpoint dict containing a
`state_dict` / `model` key) matching the architecture in `core/ball_detector.py`
(default `in_frames=3` → 9 input channels, 288×512 input). See
[`docs/ball_labeling.md`](../docs/ball_labeling.md) for how to produce it.
