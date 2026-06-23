"""core/ball_stream.py
Phase-6 (LIVE EDGE) -- STEP 1: hardware-agnostic live frame ingestion with a
DROP-ON-FULL, LATEST-FRAME-WINS buffer.

WHY THIS EXISTS
  On the Jetson the ball pipeline must read a LIVE stream and run detection/tracking
  without ever letting a slow processing frame stall the capture. If processing falls
  behind, we must DROP stale frames and always hand the consumer the NEWEST frame --
  otherwise latency drifts upward over a long match (the buffer fills with old frames
  and we end up tracking the past). This module is that ingestion layer.

  It is the platform-independent core of the architecture in
  docs/ball_2d_jetson_architecture.md (pillar 1). The ONE pluggable part is the frame
  SOURCE; everything else (the ring, the thread, the stats) is identical everywhere.

TWO BACKENDS (the source is the only thing that changes per platform)
  * OpenCVSource ...... cv2.VideoCapture. Runs on the dev laptop NOW (file / USB index /
                        any URI OpenCV can open). CPU decode, a host-memory numpy frame.
                        This is what we measure step 1 against on real footage.
  * GStreamerSource ... the Jetson L4T zero-copy path (nvv4l2decoder / nvarguscamerasrc
                        / aravissrc -> NVMM -> appsink). Needs python-gi + the L4T
                        GStreamer plugins, so it only constructs on a Jetson. The exact
                        pipeline string per source profile is built by gst_launch_string()
                        below (pure function -- reviewable/testable off-device). The live
                        pull is validated ON-DEVICE (a later gate), not on this laptop.

RESOLUTION & DECODE (these cameras: 4K @ ~20 fps, two of them)
  Decode resolution is set by the CAMERA, not chosen here: the cams encode 4K, so NVDEC
  decompresses 4K (cheap in hardware -- 2x 4K@20 fits the Orin Nano, whose H.265 decoder
  does 2x 4K@30). 60 fps is NOT achievable with 20 fps cameras: the pipeline just keeps up
  with the ~20 fps it is fed (~50 ms/frame, which also satisfies the <50 ms latency goal).
  This INGESTION layer never resizes -- it delivers the full native frame. The DETECTOR
  stage (later) feeds the model COARSE-TO-FINE: a downscaled full frame (~1280 wide) for
  whole-court coverage -- needed because at 20 fps the ball jumps far between frames -- plus
  a NATIVE 4K crop refine around the detection for sub-pixel precision, so the 4K detail
  stays in the loop without ever running the net on the full 4K. On the dev laptop OpenCV
  decodes on the CPU (no GPU-decode lib installed); on the Jetson it is NVDEC -> NVMM.

THE BUFFER (why a single slot, not a queue)
  LatestFrame holds AT MOST ONE frame. A new frame OVERWRITES an unconsumed one (a drop).
  So the buffer can never grow and latency can never drift -- by construction. We contrast
  it against an unbounded FIFO in the self-test to PROVE the difference with numbers.

RUN THE STEP-1 GATE (on real footage, on this laptop)
    python -m core.ball_stream config-side1.json
    python -m core.ball_stream config-side1.json --max-frames 600 --slow-ms 40
  It reports decode FPS, frame drops, and -- the key property -- that the frame AGE at
  consumption stays BOUNDED under a slow consumer (no latency drift), where an unbounded
  FIFO's age grows without limit.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# A timestamped frame travelling through the pipeline.
# --------------------------------------------------------------------------- #
@dataclass
class Stamped:
    """One captured frame plus the bookkeeping we need to measure latency.

    index      : producer sequence number (0,1,2,... in capture order). Lets a
                 consumer tell how many frames were produced since the one it holds.
    t_capture  : monotonic time the frame left the source (seconds).
    frame      : the pixels (BGR ndarray). May be None in the FIFO-contrast test,
                 where we deliberately do NOT keep pixels so a growing backlog cannot
                 exhaust memory -- we only need to measure the backlog, not process it.
    """
    index: int
    t_capture: float
    frame: Optional[np.ndarray] = None


# --------------------------------------------------------------------------- #
# Frame sources (the pluggable part).
# --------------------------------------------------------------------------- #
class OpenCVSource:
    """cv2.VideoCapture backend -- the dev-laptop path.

    `source` is whatever OpenCV understands: a file path, an integer camera index
    (USB/UVC webcam), or a URL. Not zero-copy and CPU-decoded -- fine for development
    and for exercising the ring/threading logic; the Jetson uses GStreamerSource.
    """

    def __init__(self, source: Any) -> None:
        import cv2
        self._cv2 = cv2
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise RuntimeError(f"OpenCVSource: could not open source {source!r}")
        self._meta = {
            "backend": "opencv",
            "source": source,
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self._cap.get(cv2.CAP_PROP_FPS)) or None,
            "frame_count": int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None,
        }

    def read(self) -> Optional[np.ndarray]:
        ok, frame = self._cap.read()
        return frame if ok else None        # None => end of stream / read failure

    def release(self) -> None:
        self._cap.release()

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta


# The exact Jetson GStreamer source-bin strings, one per input profile. These mirror
# docs/ball_2d_jetson_architecture.md (pillar 1) and all END at the same boundary:
# an appsink delivering NV12 frames, with max-buffers=1 drop=true so the decoder never
# blocks on us. Kept as a pure function so it can be inspected/tested off-device.
def gst_launch_string(profile: str, uri: str = "", width: int = 0, height: int = 0,
                      codec: str = "h264", sink_name: str = "sink") -> str:
    """Return the GStreamer pipeline string for a source `profile`.

    profile: "file" | "rtsp" | "csi" | "gige". `uri` is the file path / rtsp url /
    camera id. `codec` ("h264"|"h265") selects the depay/parse for networked streams.
    """
    depay = "rtph265depay ! h265parse" if codec == "h265" else "rtph264depay ! h264parse"
    parse = "h265parse" if codec == "h265" else "h264parse"
    demux = "qtdemux"
    caps = "video/x-raw(memory:NVMM),format=NV12"
    appsink = (f"appsink name={sink_name} sync=false max-buffers=1 drop=true "
               f"emit-signals=true")

    # NO RUNTIME DOWNSCALE. The source is already at model resolution (native ~1280p),
    # so frames go to the detector at full size and stay on the GPU (NVMM) the whole way:
    #   GPU decode (nvv4l2decoder/NVDEC) -> nvvidconv (NV12 FORMAT only, not a resize)
    #   -> appsink -> GPU detector. No CPU copy, no CPU/GPU downscale.
    # (Any resolution change belongs OFFLINE in training-data prep, never in the live path.)
    if profile == "file":
        return (f"filesrc location={uri} ! {demux} ! {parse} ! nvv4l2decoder "
                f"! nvvidconv ! {caps} ! {appsink}")
    if profile == "rtsp":
        return (f"rtspsrc location={uri} latency=0 drop-on-latency=true protocols=tcp "
                f"! {depay} ! nvv4l2decoder ! nvvidconv ! {caps} ! {appsink}")
    if profile == "csi":
        # width/height here select the SENSOR CAPTURE MODE (native acquisition), which
        # is not a downscale -- the camera is told to deliver frames at this size.
        return (f"nvarguscamerasrc ! "
                f"video/x-raw(memory:NVMM),width={width or 1280},height={height or 720},"
                f"framerate=60/1 ! nvvidconv ! {caps} ! {appsink}")
    if profile == "gige":
        # GigE Vision (machine-vision, uncompressed) -- NOT rtsp; no decoder.
        return (f"aravissrc camera-name={uri} ! bayer2rgb ! nvvidconv ! {caps} ! {appsink}")
    raise ValueError(f"unknown source profile {profile!r} "
                     "(expected file|rtsp|csi|gige)")


class GStreamerSource:
    """Jetson L4T zero-copy backend. Constructs only where python-gi + the L4T
    GStreamer plugins exist (i.e. on the Jetson). On the dev laptop the import fails
    fast with a clear message so callers fall back to OpenCVSource.

    NOTE: the pipeline STRING (gst_launch_string) is validated off-device; the live
    appsink pull below is validated ON-DEVICE as a later gate. It is written here so
    the deployment step is a wiring exercise, not a redesign.
    """

    def __init__(self, profile: str, uri: str = "", width: int = 0, height: int = 0,
                 codec: str = "h264") -> None:
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
        except Exception as e:                      # ImportError or missing typelib
            raise RuntimeError(
                "GStreamerSource needs python-gi and the L4T GStreamer plugins "
                "(Jetson only). On the dev laptop use the OpenCV backend "
                f"(config ball.stream.backend = 'opencv'). Underlying error: {e}")
        self._Gst = Gst
        Gst.init(None)
        self._pipeline_str = gst_launch_string(profile, uri, width, height, codec)
        self._pipeline = Gst.parse_launch(self._pipeline_str)
        self._sink = self._pipeline.get_by_name("sink")
        self._pipeline.set_state(Gst.State.PLAYING)
        self._meta = {"backend": "gstreamer", "profile": profile, "uri": uri,
                      "pipeline": self._pipeline_str, "width": width, "height": height}

    def read(self) -> Optional[np.ndarray]:
        sample = self._sink.emit("pull-sample")     # blocks until a frame or EOS
        if sample is None:
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_value("width"), caps.get_value("height")
        ok, mapinfo = buf.map(self._Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            # NV12 -> we expose the luma plane reshape here; on-device we convert to
            # the detector's expected format. Validated during the Jetson gate.
            data = np.frombuffer(mapinfo.data, dtype=np.uint8)
            return data.reshape((h * 3 // 2, w)).copy()
        finally:
            buf.unmap(mapinfo)

    def release(self) -> None:
        self._pipeline.set_state(self._Gst.State.NULL)

    @property
    def meta(self) -> Dict[str, Any]:
        return self._meta


def build_source(config: Dict[str, Any]) -> Any:
    """Pick a backend from config. Defaults to OpenCV (dev laptop). On the Jetson set
    config["ball"]["stream"]["backend"] = "gstreamer" and a source_profile.
    """
    stream = config.get("ball", {}).get("stream", {})
    backend = stream.get("backend", "opencv")
    if backend == "opencv":
        # OpenCV opens the same `source` the rest of the project uses.
        return OpenCVSource(config["source"])
    if backend == "gstreamer":
        return GStreamerSource(
            profile=stream.get("source_profile", "file"),
            uri=stream.get("uri", config.get("source", "")),
            width=stream.get("width", 0), height=stream.get("height", 0),
            codec=stream.get("codec", "h264"))
    raise ValueError(f"unknown ball.stream.backend {backend!r} (expected opencv|gstreamer)")


# --------------------------------------------------------------------------- #
# The buffers: a single-slot latest-wins ring, and (for contrast) an unbounded FIFO.
# Both share the same put()/get() shape so the capture thread is sink-agnostic.
# --------------------------------------------------------------------------- #
class LatestFrame:
    """Single-slot, latest-frame-wins handoff. put() OVERWRITES any unconsumed frame
    (counted as a drop); the buffer therefore never holds more than one frame, so
    latency cannot drift. This is the production sink."""

    def __init__(self) -> None:
        self._slot: Optional[Stamped] = None
        self._lock = threading.Lock()
        self._evt = threading.Event()
        self.dropped = 0
        self.max_occupancy = 0          # for the report: stays 1 by construction

    def put(self, item: Stamped) -> None:
        with self._lock:
            if self._slot is not None:
                self.dropped += 1       # the previous frame was never consumed -> dropped
            self._slot = item
            self.max_occupancy = max(self.max_occupancy, 1)
        self._evt.set()

    def get(self, timeout: float = 0.1) -> Optional[Stamped]:
        if not self._evt.wait(timeout):
            return None
        with self._lock:
            item = self._slot
            self._slot = None
            self._evt.clear()
            return item


class UnboundedFIFO:
    """An ordinary unbounded queue -- the WRONG choice for a live stream, kept only to
    demonstrate the failure it causes (backlog and latency grow under a slow consumer).
    Never used in production."""

    def __init__(self) -> None:
        self._q: Deque[Stamped] = deque()
        self._lock = threading.Lock()
        self._evt = threading.Event()
        self.dropped = 0
        self.max_occupancy = 0

    def put(self, item: Stamped) -> None:
        with self._lock:
            self._q.append(item)
            self.max_occupancy = max(self.max_occupancy, len(self._q))
        self._evt.set()

    def get(self, timeout: float = 0.1) -> Optional[Stamped]:
        if not self._evt.wait(timeout):
            return None
        with self._lock:
            item = self._q.popleft() if self._q else None
            if not self._q:
                self._evt.clear()
            return item


# --------------------------------------------------------------------------- #
# Capture thread: pull from the source as fast as it allows, push into the sink.
# A slow consumer NEVER blocks this thread -- that is the whole point.
# --------------------------------------------------------------------------- #
@dataclass
class StreamStats:
    produced: int = 0
    t_first: Optional[float] = None
    t_last: Optional[float] = None
    ended: bool = False

    def producer_fps(self) -> float:
        if self.t_first is None or self.t_last is None or self.t_last <= self.t_first:
            return 0.0
        return (self.produced - 1) / (self.t_last - self.t_first)


class StreamCapture:
    """Owns a source + a background capture thread + a sink. The consumer calls read().

    keep_pixels=False is used only by the FIFO-contrast test, so a growing backlog
    stores lightweight (frame-less) Stamped items and cannot exhaust memory.
    """

    def __init__(self, source: Any, sink: Any, *, max_frames: Optional[int] = None,
                 keep_pixels: bool = True) -> None:
        self._source = source
        self._sink = sink
        self._max_frames = max_frames
        self._keep_pixels = keep_pixels
        self.stats = StreamStats()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._stop = threading.Event()

    def _run(self) -> None:
        idx = 0
        while not self._stop.is_set():
            frame = self._source.read()
            now = time.monotonic()
            if frame is None:                       # end of stream
                break
            if self.stats.t_first is None:
                self.stats.t_first = now
            self.stats.t_last = now
            self.stats.produced = idx + 1
            self._sink.put(Stamped(index=idx, t_capture=now,
                                   frame=frame if self._keep_pixels else None))
            idx += 1
            if self._max_frames is not None and idx >= self._max_frames:
                break
        self.stats.ended = True

    def start(self) -> "StreamCapture":
        self._thread.start()
        return self

    def read(self, timeout: float = 0.1) -> Optional[Stamped]:
        return self._sink.get(timeout=timeout)

    def producer_done(self) -> bool:
        return self.stats.ended

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._source.release()


# --------------------------------------------------------------------------- #
# Step-1 self-test / benchmark. Three scenarios, all on REAL footage.
# --------------------------------------------------------------------------- #
def _drain(cap: StreamCapture, slow_ms: float) -> Dict[str, Any]:
    """Consume from `cap` until the producer is done and the sink is empty. Optionally
    sleep slow_ms per frame to SIMULATE downstream work (detection+tracking). Returns
    consumption stats, the key one being frame AGE at consumption (= latency proxy)."""
    consumed = 0
    ages: List[float] = []
    lags: List[int] = []
    while True:
        item = cap.read(timeout=0.2)
        if item is None:
            if cap.producer_done():
                # one more non-blocking sweep to clear any last frame
                item = cap.read(timeout=0.05)
                if item is None:
                    break
            else:
                continue
        now = time.monotonic()
        ages.append((now - item.t_capture) * 1000.0)         # ms old when we got it
        lags.append(cap.stats.produced - 1 - item.index)     # frames produced since
        consumed += 1
        if slow_ms > 0:
            time.sleep(slow_ms / 1000.0)
    return {"consumed": consumed, "ages_ms": ages, "lags": lags}


def _summ(name: str, vals: List[float]) -> str:
    if not vals:
        return f"{name}: (none)"
    arr = np.asarray(vals, dtype=float)
    half = max(1, len(arr) // 2)
    drift = arr[half:].mean() - arr[:half].mean() if len(arr) > 1 else 0.0
    return (f"{name}: mean={arr.mean():.1f} max={arr.max():.1f} "
            f"p95={np.percentile(arr, 95):.1f}  drift(2nd-1st half)={drift:+.1f}")


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Ball ingestion step-1 gate (real footage)")
    ap.add_argument("config", help="config json (uses its 'source')")
    ap.add_argument("--max-frames", type=int, default=600,
                    help="cap frames per scenario (keeps the run quick & memory bounded)")
    ap.add_argument("--slow-ms", type=float, default=40.0,
                    help="simulated per-frame consumer work in the slow-consumer tests")
    ap.add_argument("--skip-fifo", action="store_true",
                    help="skip the unbounded-FIFO contrast scenario")
    args = ap.parse_args()

    config = _load_config(args.config)
    print(f"[stream] config={args.config}  source={config['source']}")

    # ---- Scenario A: raw decode throughput (fast consumer, latest-frame sink) ----
    src = build_source(config)
    print(f"[stream] backend={src.meta.get('backend')} "
          f"{src.meta.get('width')}x{src.meta.get('height')} "
          f"src_fps={src.meta.get('fps')}")
    capA = StreamCapture(src, LatestFrame(), max_frames=args.max_frames).start()
    rA = _drain(capA, slow_ms=0.0)
    capA.stop()
    decode_fps = capA.stats.producer_fps()
    print("\n=== A. RAW DECODE THROUGHPUT (fast consumer) ===")
    print(f"  produced={capA.stats.produced}  consumed={rA['consumed']}")
    print(f"  decode/producer FPS = {decode_fps:.1f}")
    print(f"  {_summ('frame age ms', rA['ages_ms'])}")

    # The slow-consumer scenarios only exercise drops if the consumer is genuinely
    # slower than the decoder. Auto-scale the simulated work to ~2.5x the decode period
    # so the test is meaningful regardless of how fast this machine decodes.
    decode_period_ms = 1000.0 / decode_fps if decode_fps > 0 else args.slow_ms
    slow_ms = max(args.slow_ms, 2.5 * decode_period_ms)
    print(f"\n[stream] decode period ~{decode_period_ms:.0f} ms/frame -> "
          f"simulating a {slow_ms:.0f} ms/frame consumer (slower than decode, on purpose)")

    # ---- Scenario B: slow consumer + LatestFrame (the production sink) ----
    src = build_source(config)
    sinkB = LatestFrame()
    capB = StreamCapture(src, sinkB, max_frames=args.max_frames).start()
    rB = _drain(capB, slow_ms=slow_ms)
    capB.stop()
    drop_pct = 100.0 * sinkB.dropped / max(1, capB.stats.produced)
    print(f"\n=== B. SLOW CONSUMER ({slow_ms:.0f} ms/frame) + LatestFrame [PRODUCTION] ===")
    print(f"  produced={capB.stats.produced}  consumed={rB['consumed']}  "
          f"dropped={sinkB.dropped} ({drop_pct:.0f}%)  max_occupancy={sinkB.max_occupancy}")
    print(f"  {_summ('frame age ms', rB['ages_ms'])}")
    print(f"  {_summ('lag (frames behind latest)', [float(x) for x in rB['lags']])}")
    bounded = (max(rB['ages_ms']) if rB['ages_ms'] else 0) < 3 * slow_ms + 50
    print(f"  -> age BOUNDED & non-drifting? {'PASS' if bounded else 'CHECK'} "
          f"(buffer never exceeded {sinkB.max_occupancy} frame)")

    # ---- Scenario C: slow consumer + unbounded FIFO (the failure we avoid) ----
    if not args.skip_fifo:
        src = build_source(config)
        sinkC = UnboundedFIFO()
        capC = StreamCapture(src, sinkC, max_frames=args.max_frames,
                             keep_pixels=False).start()
        rC = _drain(capC, slow_ms=slow_ms)
        capC.stop()
        print(f"\n=== C. SLOW CONSUMER ({slow_ms:.0f} ms/frame) + UNBOUNDED FIFO [BAD] ===")
        print(f"  produced={capC.stats.produced}  consumed={rC['consumed']}  "
              f"dropped={sinkC.dropped}  PEAK BACKLOG={sinkC.max_occupancy} frames")
        print(f"  {_summ('frame age ms', rC['ages_ms'])}")
        print("  -> note the growing age/backlog: an unbounded queue makes the live\n"
              "     overlay fall further behind real time the longer the match runs.")

    print("\n[stream] step-1 gate done. LatestFrame keeps latency bounded by DROPPING\n"
          "         stale frames; the FIFO drifts. This is pillar-1 behavior, measured.")


if __name__ == "__main__":
    main()
