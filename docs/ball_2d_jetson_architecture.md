# 2D Padel Ball Tracking — Live Edge Architecture (Jetson Orin Nano 8GB)

**Status:** design specification (Phase-6 territory). Nothing here is built yet — this
is the blueprint we build against, gated step by step, measured on real footage.

**Target:** ingest a live stream, detect the 6.5 cm padel ball at **60+ FPS**, track it
with physics-informed logic, project it to court coordinates, and emit the result with
**< 50 ms capture-to-result latency** — on an **Orin Nano 8GB** with no monitor (output
goes to disk / downstream, mirroring the current NVR workflow).

---

## 0. Hardware reality (Orin Nano 8GB) — what it forces

| Capability | Orin Nano 8GB | Consequence for this spec |
| --- | --- | --- |
| GPU | 1024-core Ampere, ~40-67 INT8 TOPS | All inference on GPU. Model must stay tiny. |
| **DLA** | **none** | Cannot offload detection to a DLA. No "GPU + DLA" split. |
| **NVENC (encoder)** | **none** | **No hardware H.264/H.265 encode.** RTSP/MP4 output would need CPU encode (`x264enc`) - expensive. -> output **data, not video**. |
| NVDEC (decoder) | yes (1x) | RTSP/file H.264/H.265 decode is hardware-accelerated (`nvv4l2decoder`). |
| Memory | 8 GB **unified** LPDDR5, ~68 GB/s | CPU and GPU share one pool. Every host<->device copy costs twice. Zero-copy NVMM is mandatory. |
| JetPack | 6.x (TensorRT 8.6/10, GStreamer 1.20, CUDA 12) | Element/API names below assume JetPack 6. |

**Three facts dominate every decision below:** no DLA (keep the net small), no NVENC
(emit data, render overlays offline), unified memory (never leave NVMM).

---

## 1. Hardware-Accelerated Live Ingestion Pipeline

### 1.1 The source is the only pluggable part

You asked whether the camera type has to be decided now. **It does not.** The pipeline
is built as a **source bin** that, whatever the input, hands the rest of the system the
*same* thing: `video/x-raw(memory:NVMM), format=NV12` at full resolution. Everything
downstream (preprocess -> TensorRT -> tracker -> output) is identical. Swapping cameras =
swapping one bin, selected by `config["ball"]["stream"]["source_profile"]`.

```
+-- source bin (pluggable) --------------+
|  file  | rtsp | csi | gige             |
+----------------------------------------+
              |  video/x-raw(memory:NVMM),NV12   (ZERO-COPY boundary)
              v
        nvvidconv (NVMM, NV12 FORMAT only — NO resize; native res to the model)
              v
        appsink (max-buffers=1, drop=true)  -- latest-frame-wins ring
              v
   [Python] capture thread -> bounded queue(1) -> inference thread
```

### 1.2 The three source profiles you will actually hit

**(a) NOW — footage pulled from the NVR (file on disk).** Decode-only, no network:

```
filesrc location=clip.mp4
  ! qtdemux ! h264parse ! nvv4l2decoder
  ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12
  ! appsink name=sink sync=false max-buffers=1 drop=true emit-signals=true
```
Use `sync=false` so we process as fast as the GPU allows (offline footage has no live clock).

**(b) SOON — RTSP IP camera (H.264/H.265).** The latency-sensitive case:

```
rtspsrc location=rtsp://CAM/stream latency=0 drop-on-latency=true protocols=tcp
  ! rtph264depay ! h264parse ! nvv4l2decoder
  ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12
  ! appsink name=sink sync=false max-buffers=1 drop=true emit-signals=true
```
- `latency=0 drop-on-latency=true` -> the jitter buffer is the #1 hidden latency source on
  RTSP; minimize it and let it drop rather than accumulate (prevents drift over a long match).
- H.265 camera -> `rtph265depay ! h265parse` (NVDEC handles both).
- `protocols=tcp` only if UDP is unreliable on the network; UDP is lower latency.

**(c) LATER — GigE Vision cameras into the Jetson.** WARNING: **GigE Vision != RTSP.** These
are machine-vision cameras: **uncompressed** GenICam frames, no `rtspsrc`, no decoder.
Source = `aravissrc` (Aravis/GenICam) or the vendor's GStreamer element; output is raw
Bayer/Mono:

```
aravissrc camera-name="Vendor-SN" features="ExposureTime=2000,AcquisitionFrameRate=60"
  ! bayer2rgb            # or CUDA debayer; Mono cameras skip this
  ! nvvidconv ! video/x-raw(memory:NVMM),format=NV12
  ! appsink name=sink sync=false max-buffers=1 drop=true emit-signals=true
```
**Bandwidth is the new bottleneck, not decode:** 1 GigE ~= 125 MB/s. 1920x1080 Mono8 @60fps
~= 124 MB/s — that *saturates a single 1 GigE link*. Color or higher fps needs 5/10 GigE,
multiple NICs, or the camera's packed/compressed mode. Flag this when sizing the cameras.

### 1.3 Non-blocking, latest-frame-wins buffer

The hard rule: **a slow inference frame must never stall capture.** Two layers enforce it:

1. **In GStreamer:** `appsink ... max-buffers=1 drop=true`. The sink holds at most one
   frame; a newer frame overwrites the unconsumed one. The decoder never blocks on us.
2. **In Python:** a depth-1 drop-on-full handoff between the capture thread and the
   inference thread, so even the appsink->Python pull stays latest-only:

```python
# core/ball_stream.py  (skeleton)
class LatestFrame:
    """Single-slot, lock-protected, latest-wins handoff. No queue growth, ever."""
    def __init__(self):
        self._buf = None
        self._lock = threading.Lock()
        self._evt = threading.Event()
    def put(self, frame, pts):                 # capture thread
        with self._lock:
            self._buf = (frame, pts)           # overwrite — drop the stale one
        self._evt.set()
    def get(self, timeout=0.1):                # inference thread
        if not self._evt.wait(timeout): return None
        with self._lock:
            self._evt.clear()
            return self._buf
```

Per-frame we record `pts` (the GStreamer buffer PTS) so latency is measurable end-to-end
(section 4). Dropped frames are *counted*, not hidden — that count is a benchmark metric, and
"measure failures, don't hide them" is project rule #5.

> DeepStream alternative: if you later move the hot path to native DeepStream, the same
> behavior comes from `nvstreammux live-source=1 batched-push-timeout=<~1 frame>` plus a
> drop policy, instead of the appsink/ring above. For one camera and a Python team,
> GStreamer + appsink is simpler and just as fast; revisit DeepStream when you fan out to
> N cameras on one Jetson.

---

## 2. Edge-Optimized Model Architecture & Quantization

### 2.1 Why a heatmap, not boxes (and why it fits *this* repo)

The existing labels are **point labels** — `frame, visible, u, v` (TrackNet CSV format,
`docs/ball_labeling.md`). An **anchor-free heatmap** detector trains directly on those
points; a YOLO box model would require re-labeling every frame as a box and tends to miss
the tiny far ball. So the model is a **CenterNet/TrackNet-lite heatmap regressor**, which
also reuses the entire existing labeling + eval toolchain (`ball_label.py`,
`ball_eval.py`, `ball_precision.py`) unchanged.

### 2.2 Architecture — slim encoder, high-res head

```
Input:   3 stacked grayscale frames (t-2,t-1,t), /255   [temporal motion cue].
         Source is 4K @ ~20 fps. Feed the model COARSE-TO-FINE (see 2.3): a downscaled
         full frame (~1280 wide) for whole-court coverage + a NATIVE 4K crop refine around
         the detection. We do NOT run the net on full 4K (too heavy, x2 cams) NOR only on a
         blurry full downscale (erases the ball) — coarse for coverage, native crop for px.
         (single-frame RGB is the fallback if 3-frame inference busts the latency budget)

Backbone: MobileNetV3-Small (width 0.75), strides P1..P5
Neck:     Slim-BiFPN fusing P2(stride4)+P3+P4 ONLY — keep the high-res P2 path.
          P2 is non-negotiable: even at native 1280x720 the 6.5 cm ball is ~2-4 px;
          downsampling it past stride-4 erases it. We pay for one high-res fusion level.
Head:     1x heatmap (sigmoid, output stride 4)         -> where the ball is
          + 2x sub-pixel offset (CenterNet local offset) -> precise (u,v) below grid res
Params:   ~1.5-3 M.  Output heatmap = input / 4 (e.g. 320x180 at 1280x720).
```

Decode at inference: `argmax` of the heatmap -> grid cell; add the regressed offset ->
sub-pixel `(u,v)`; the peak value is the confidence (feeds the tracker's `R`, section 3). All of
this runs on GPU (CUDA/NPP) — no host copy.

### 2.3 Track-guided ROI — the key edge trick

Full-frame inference every frame at a resolution high enough for a tiny ball is the
expensive path. Instead, **let the tracker steer the detector**:

At 20 fps the ball jumps far between frames (a smash moves ~1 m -> ~1000 px at 4K), so a
small "track-guided" crop alone would lose it. So we run COARSE-TO-FINE EVERY frame:
- **Coarse (coverage):** downscale the 4K to ~1280 wide and run the heatmap on the full
  frame to locate the ball anywhere on court. Cheap; robust to the big inter-frame jumps.
- **Fine (precision):** crop a NATIVE-res window (e.g. 512x512) from the 4K around the
  coarse hit (or Kalman prediction) and re-run the heatmap there — the ball sits at full
  pixel density, recovering the sub-pixel precision the downscale lost.

This keeps native 4K detail in the loop without ever running the net on the full 4K (too
heavy, x2 cameras, no DLA). Whether the coarse pass alone already suffices, or the fine
pass earns its cost, is something we MEASURE — not assume.

### 2.4 TensorRT deployment — INT8 without losing the ball

Export ONNX -> build a TensorRT engine. The danger with INT8 is specific and real: the
ball's heatmap response is a **small, rare, low-amplitude** activation; naive calibration
treats it as an outlier and clips it -> the quantized net goes blind to faint/far balls.

**Ship FP16 first (the baseline).** FP16 carries no quantization risk to the faint-ball
signal and, for a ~2-3 M-param model with track-guided ROI cropping, still fits the latency
budget. Treat INT8 as an **optional** optimization to reclaim speed/VRAM — adopt it only if
it passes the precision gate below; otherwise FP16 ships.

**INT8 mitigations (if/when we pursue it):**

1. **Mixed precision, not pure INT8.** Backbone INT8 (where the throughput is); keep the
   **BiFPN fusion + heatmap/offset head in FP16**. In TensorRT set the builder flags
   `INT8 | FP16`, then pin the head layers:
   ```python
   for layer in head_layers:           # last fusion + heatmap + offset
       layer.precision = trt.float16
       layer.set_output_type(0, trt.float16)
   config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
   ```
2. **Ball-rich calibration set.** Calibrate on frames where the ball is *present and
   varied* (near/far, fast/blurred, against glass) — NOT a random sample dominated by
   empty frames. Entropy calibrator:
   ```python
   class BallCalibrator(trt.IInt8EntropyCalibrator2):
       # feeds preprocessed batches of LABELED ball-present frames; caches to disk
       def get_batch(self, names): ...
       def read_calibration_cache(self): ...
       def write_calibration_cache(self, cache): ...
   ```
   Use ~500-1000 calibration frames, balanced near/far. Cache the table so rebuilds are cheap.
3. **Quantization acceptance gate = `ball_precision.py`.** The repo already scores the
   detector against held-out labels (recall / precision / localization error in px and
   meters). Run it on the **FP16** engine and the **INT8** engine and compare. **Accept
   INT8 only if** recall@tight-tol and localization error are within an agreed margin of
   FP16 (e.g. <=1-2% recall drop, <=0.5 px localization drift). If INT8 fails the gate,
   ship FP16 — it still fits the latency budget on this model size.

**Build settings:** cap `config.max_workspace_size` (e.g. 512 MB — unified memory is
scarce, section 2.5); `builder.max_batch_size = 1` (live, one frame); serialize the engine to
disk and load it (never build on the device at startup).

### 2.5 Memory budget (8 GB unified — write it down, defend it)

| Consumer | Tactic | Rough budget |
| --- | --- | --- |
| OS + JetPack + Python | headless, no desktop | ~1.5 GB |
| GStreamer NVMM decode pool | **bound** the pool (few surfaces); `nvv4l2decoder` extra surfaces low | ~150-400 MB |
| TensorRT engine (INT8) + workspace | INT8 shrinks weights; cap workspace | ~150-400 MB |
| CUDA context + NPP preprocess | one context, reuse buffers | ~600 MB-1 GB |
| Tracker / events / app | NumPy, negligible | < 50 MB |
| **Headroom kept free** | guard against fragmentation/spikes | **>= 2 GB** |

Rules: keep frames in **NVMM** end-to-end (no `nvvidconv` to CPU memory unless an output
needs it); do the coarse downscale + native crop on GPU; reuse fixed device buffers (no
per-frame `cudaMalloc`); monitor with `tegrastats`/`jtop` (section 4). A 4K NV12 frame is
~12 MB; with 2 cameras that is the main NVMM pressure, so bound the decode pool tightly.
The NET still only ever sees a ~1280 coarse frame + a small native crop, so INFERENCE
memory stays small regardless of the 4K source (see sections 2.3 / 6).

---

## 3. Lightweight Streaming Post-Processing & Physics

This is where the repo is already strongest — port and reuse, don't reinvent.

### 3.1 Single-target tracker (reuse `core/ball_tracker.py`)

There is **one** ball. Full ByteTrack/Hungarian is overkill — it solves multi-object
assignment we don't have. Use the existing **constant-velocity Kalman**
(`BallTracker`, state `[px,py,vx,vy]`), which already:
- smooths noisy `(u,v)`,
- **coasts through occlusions** (player/net) on last velocity, growing covariance — never
  resetting on a gap,
- **scales measurement noise `R` by detection confidence** (the heatmap peak from section 2.2),
- optionally gates outliers (off by default — a CV model mispredicts through bounces).

Where >1 candidate blob appears (a second ball on court, a reflection), assignment is a
trivial **1xN nearest-under-gate**, not a Hungarian matrix. Parked/abandoned balls are
already handled by `core/ball_suppress.py`; the failure rates that justify it were
measured by `core/ball_case_audit.py`. Keep all three.

### 3.2 Bounce / hit detection (reuse `core/ball_events.py`) — the cheap heuristic

You asked for "a low-overhead math heuristic, not a neural net, watching sudden changes in
the velocity vector." **That module already exists** and does exactly this from the
*smoothed Kalman velocity* (never across an occlusion gap — a held prediction would fire
false events), with a refractory window so one bounce fires once:

- **floor_bounce** — vertical image velocity flips **down->up** while the ball maps inside
  the court footprint (a Z=0 instant -> also a valid in/out call).
- **wall_bounce** — horizontal velocity reverses at the side-glass boundary.
- **hit** — sharp impulsive direction change at speed (serve/smash/volley), not a bounce.

**One upgrade for live use:** feed events *back into* the tracker. At a detected `hit`,
**inflate process noise `Q` for one step** so the CV filter accepts the post-hit detection
instead of gating it (the tracker's own docstring flags this as the bounce/hit weakness).
Cheap, and it fixes the most important frames (the ones right after impact).

### 3.3 Pixel -> court via homography (reuse `utils/homography.py`)

The calibrated `H`/`H_inv` already live in `config-side1.json` (mean reprojection error
0.04 m). Use `Homography.pixel_to_meters((u,v))` to put the ball on the 10x20 m court model
(net at y=10). **Honest limitation, already documented in the module:** the floor
homography is metric **only when the ball is on/near the floor** (i.e. at bounces); an
airborne ball maps with a parallax error that grows with height. So:
- court coordinates are **authoritative at floor-contact events** (in/out, bounce
  position), and **approximate while airborne** — label the emitted record accordingly.

**CuPy vs NumPy — don't cargo-cult the GPU.** Per frame we map **one** point. A 3x3*3x1
matmul is sub-microsecond in NumPy; shipping it to CuPy costs more in host<->device latency
than it saves. -> **NumPy for the per-frame point.** Reserve CuPy/vectorization for *batch*
work (reprojecting an entire trajectory offline, or many candidates at once). Stating this
prevents a pointless GPU round-trip in the hot loop.

---

## 4. Real-Time Output & Benchmarking

### 4.1 Output: data first (you have no monitor, and no NVENC)

Given the NVR/disk workflow and the missing encoder, the production output is the
**tracking data stream**, not a rendered video:

- **Primary sink:** one record per frame ->
  `{frame, pts_ns, u, v, conf, court_x, court_y, vx, vy, status, event}` appended to a
  rotating **JSONL** (and/or **Parquet** — `polars` is already a dependency) on disk, and
  optionally pushed over a lightweight IPC (UDP/ZeroMQ/shared-memory) to any live consumer
  (scoreboard, API). This is microseconds of work and never touches the encoder.
- **QA overlay (dev toggle only):** drawing boxes/trails and writing an MP4 needs CPU
  encode (`x264enc ultrafast`) on Orin Nano — costly. So: **don't render in production.**
  Render overlays **offline** from `data + the original NVR footage` (which you keep
  anyway), or record only **short clips around events** at low cadence. Make it a flag,
  default off.

> If a remote live video feed becomes a hard requirement later, that is the argument for
> an **Orin NX 8GB** (it has NVENC + DLA) — call it out at the hardware-selection boundary
> rather than burning the Nano's CPU on software H.264.

### 4.2 Benchmark suite (`edge/benchmark.py`)

"If you can't measure it, you're not done." Four metrics, each with a concrete method:

| Metric | How | Target |
| --- | --- | --- |
| **Capture-to-result latency** | wall-clock at result-emit minus frame **PTS** at capture; report **p50 / p95 / p99**. (True "glass-to-glass" needs a display; without one, photon->record via the camera/NVR PTS is the honest proxy.) | p95 **< 50 ms** |
| **Pipeline frame drops** | count appsink drops + Python depth-1 drops vs frames produced; report **drop %** over a full match | low & **non-drifting** over time |
| **VRAM / unified-mem use** | parse `tegrastats` (or `jtop`) for peak RAM used during a sustained run; log per-run | **peak fits with >=2 GB free** (section 2.5) |
| **Tracking precision** | reuse `ball_precision.py`: recall / precision / **localization error** at tight pixel tolerances (e.g. **3 / 5 / 8 px**) on held-out labels, plus error in **meters** via homography; plus `ball_eval.py` detection-rate & longest-no-ball-run | recall@5px & error agreed per gate |

Run every benchmark **on real court footage only** (project rule), and run the
**FP16-vs-INT8** comparison through the precision gate (section 2.4) so quantization is a
*measured* decision, not a hope.

---

## 5. Proposed module layout (build order, gated)

```
core/
  ball_detector_trt.py   NEW    TensorRT heatmap detector: engine load, GPU pre/post, ROI crop
  ball_tracker.py        reuse  Kalman (add: Q-inflation on event)
  ball_events.py         reuse  velocity bounce/hit (add: emit -> tracker hook)
  ball_suppress.py       reuse  parked-ball suppression
  ball_stream.py         NEW    GStreamer source bin + appsink + LatestFrame ring
  ball_live.py           NEW    async orchestrator: capture thread -> infer -> track -> emit
edge/
  build_engine.py        NEW    ONNX export + TensorRT INT8/FP16 builder
  calibrator.py          NEW    IInt8EntropyCalibrator2 over ball-rich labeled frames
  benchmark.py           NEW    latency / drops / VRAM / precision harness
gst/
  source_file.txt  source_rtsp.txt  source_gige.txt   pluggable source-bin strings
```

Config additions (all tunable, never hard-coded):
```jsonc
"ball": {
  "trt":    { "engine": "weights/ball_heatmap_int8.engine",
              "precision": "int8", "input_wh": [512,288], "heatmap_stride": 2,
              "roi_crop": [384,384], "in_frames": 3 },
  "stream": { "source_profile": "file",   // file | rtsp | csi | gige
              "uri": "...", "appsink_drop": true }
}
```

**Suggested gated order (each measured before the next):**
1. `ball_stream.py` ingest + drop-on-full ring -> prove decode FPS & zero growth on file source.
2. Train the heatmap detector on existing point labels (GPU laptop) -> pass `ball_precision.py`.
3. ONNX -> TensorRT FP16 on Jetson -> benchmark latency/VRAM.
4. INT8 calibration -> pass the FP16-vs-INT8 gate.
5. Wire tracker + events + homography -> end-to-end on file source.
6. Swap source bin to RTSP -> re-run the live benchmark suite.

---

## 6. Honest risks / open tensions (don't let these surprise us)

1. **The cameras are 4K @ ~20 fps — so 60 FPS is OFF the table.** 20 fps undersamples fast
   shots and adds motion blur — the DOMINANT accuracy limiter for a 6.5 cm ball, and a
   *camera* limit software cannot fix. Target is real-time at ~20 fps (< 50 ms/frame, which
   conveniently equals one 20-fps frame period). If the cams expose any higher-fps mode
   (even at lower res, e.g. 1080p@60), it would likely beat 4K@20 for the fast ball and is
   worth testing.
2. **No NVENC** -> no cheap live video out. Production = data; overlays offline. (Orin NX if
   that changes.)
3. **No DLA** -> no inference offload; the small-model discipline in section 2 is mandatory,
   not optional.
4. **Single-camera homography is metric only at the floor.** Airborne court coordinates are
   approximate; treat bounce-time positions as the trustworthy ones.
5. **GigE bandwidth** can saturate a 1 GigE link before compute is the issue — size the NIC
   and camera mode together.
6. **Sub-50 ms over RTSP is the hard case** (camera encode + jitter buffer dominate);
   file (now) and GigE (later, uncompressed) are easier. Validate the budget per source.
7. **2x 4K@20 decode + coarse-to-fine model input.** Decode is fixed by the camera at 4K;
   2x 4K@20 fits the Orin Nano's single NVDEC (it does 2x 4K@30). The model never runs on
   the full 4K (too heavy x2 cams, no DLA) and never on a blurry full downscale alone
   (erases the ball) — instead a **coarse ~1280 downscale for coverage + a native 4K crop
   for ball pixels** (section 2.3). "Decode at 720p for a lighter decode" is NOT possible
   with a 4K-only camera: you decode whatever the camera ENCODED (4K), then crop/resize;
   to decode less the camera must send a smaller stream (some IP cams have a 1080p
   sub-stream — worth checking). Training-data resolution changes are an offline prep step
   (downscale the 4K clips once, scale the (u,v) labels by the same factor). **Precision
   knob is independent:** ship **FP16** first; INT8 only if it passes the gate (section 2.4).
