"""core/ball_detector.py
Phase-1 BALL detector: a TrackNetV2 heatmap network that finds the padel ball in
each frame and returns (u, v, confidence) or "no ball".

WHY TRACKNET AND NOT YOLO
  The ball is tiny, fast (a smash crosses ~1.4 m per frame at 20 fps) and badly
  motion-blurred. A box detector like YOLO is built for objects that occupy many
  pixels and have stable appearance; it does poorly on a 5-10 px smear. TrackNet
  instead consumes a SHORT STACK OF CONSECUTIVE FRAMES (default 3) and predicts a
  HEATMAP of where the ball is in the most recent frame. Feeding it several frames
  lets the network use MOTION (the ball moves, the court does not) to pick the ball
  out of the blur -- exactly the cue a single-frame detector throws away.

WHAT THIS FILE GIVES THE REST OF THE PIPELINE
  Same idea as core/detector.py: a thin wrapper around the model that returns a
  PLAIN, model-agnostic result, so nothing upstream imports torch or knows what a
  heatmap is. If we ever swap TrackNetV2 for WASB or a TensorRT engine, only this
  file changes.

      BallDetector.detect(frame) -> BallDetection
          .found      : bool                  was a ball located this frame?
          .u, .v      : float | None          ball pixel in FULL-frame coords
          .confidence : float                 heatmap peak value (0..1)
          .reason     : str                   "ok" | "no-ball" | "warmup" | "stub-no-weights"

  Coordinates are FULL-resolution image pixels (the model runs on a downscaled copy
  internally and we scale the point back up), so they line up with everything else
  the pipeline draws.

WEIGHTS (READ THIS -- it is the Phase-1 gate-blocker)
  TrackNet is SUPERVISED: it only works with trained weights, and there are NO
  padel-ball weights yet. Until a weights file exists at config["ball"]["weights"],
  this detector runs in a clearly-flagged STUB MODE that always returns "no ball"
  (found=False, reason="stub-no-weights") and prints one loud warning. The model
  architecture and the whole interface are still real and runnable -- drop a trained
  .pt in and detection turns on with no other code change. See docs/ball_labeling.md
  for the labeling format and the fine-tune recipe that produces those weights.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# The plain result object the pipeline consumes (no torch / heatmap leakage)
# --------------------------------------------------------------------------- #
@dataclass
class BallDetection:
    """One frame's ball result. `found` is the single source of truth: when it is
    False, (u, v) are None and the ball is absent/occluded/not-yet-warmed-up."""

    found: bool
    u: Optional[float] = None
    v: Optional[float] = None
    confidence: float = 0.0
    reason: str = "no-ball"

    def to_dict(self) -> Dict[str, Any]:
        """JSON/JSONL-ready (native python types only, per the project convention)."""
        return {
            "found": bool(self.found),
            "u": None if self.u is None else float(self.u),
            "v": None if self.v is None else float(self.v),
            "confidence": float(self.confidence),
            "reason": self.reason,
        }


# --------------------------------------------------------------------------- #
# TrackNetV2 model (VGG-style encoder/decoder, no skip connections)
# --------------------------------------------------------------------------- #
def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """The repeating unit of TrackNetV2: 3x3 conv -> batch-norm -> ReLU.
    Padding 1 keeps the spatial size, so only the maxpools/upsamples change it."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TrackNetV2(nn.Module):
    """TrackNetV2 heatmap network.

    INPUT  : (N, 3*in_frames, H, W)  -- `in_frames` consecutive RGB frames stacked
             on the channel axis (oldest first, current frame last). Default 3 -> 9
             channels. H, W default to 288 x 512 (both divisible by 8 so the three
             /2 pools and three x2 upsamples line up exactly).
    OUTPUT : (N, 1, H, W) in [0, 1]  -- per-pixel probability that the ball center
             is there IN THE MOST RECENT input frame.

    It is a straight encoder -> decoder (NOT a U-Net): the encoder halves the
    resolution three times while growing channels 64->128->256->512, then the
    decoder mirrors that back up to full resolution and a 1x1 conv + sigmoid
    squeezes it to one heatmap.
    """

    def __init__(self, in_frames: int = 3) -> None:
        super().__init__()
        in_ch = in_frames * 3  # RGB per frame

        # ---- encoder (downsampling path) ----
        self.enc1 = nn.Sequential(_conv_block(in_ch, 64), _conv_block(64, 64))
        self.enc2 = nn.Sequential(_conv_block(64, 128), _conv_block(128, 128))
        self.enc3 = nn.Sequential(_conv_block(128, 256), _conv_block(256, 256),
                                  _conv_block(256, 256))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- bottleneck (deepest features, no further downsample) ----
        self.bottleneck = nn.Sequential(_conv_block(256, 512), _conv_block(512, 512),
                                        _conv_block(512, 512))

        # ---- decoder (upsampling path, mirror of the encoder) ----
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec3 = nn.Sequential(_conv_block(512, 256), _conv_block(256, 256),
                                  _conv_block(256, 256))
        self.dec2 = nn.Sequential(_conv_block(256, 128), _conv_block(128, 128))
        self.dec1 = nn.Sequential(_conv_block(128, 64), _conv_block(64, 64))

        # ---- head: 1x1 conv to a single channel + sigmoid -> probability map ----
        self.head = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)              # (N, 64,  H,   W)
        x = self.pool(x)              # (N, 64,  H/2, W/2)
        x = self.enc2(x)              # (N, 128, H/2, W/2)
        x = self.pool(x)              # (N, 128, H/4, W/4)
        x = self.enc3(x)              # (N, 256, H/4, W/4)
        x = self.pool(x)              # (N, 256, H/8, W/8)

        x = self.bottleneck(x)        # (N, 512, H/8, W/8)

        x = self.up(x)                # (N, 512, H/4, W/4)
        x = self.dec3(x)              # (N, 256, H/4, W/4)
        x = self.up(x)                # (N, 256, H/2, W/2)
        x = self.dec2(x)              # (N, 128, H/2, W/2)
        x = self.up(x)                # (N, 128, H,   W)
        x = self.dec1(x)              # (N, 64,  H,   W)

        # sigmoid here so .detect() gets probabilities directly. For TRAINING you
        # would typically use BCELoss on this output (or move the sigmoid out and
        # use BCEWithLogitsLoss); keep the choice consistent with the saved weights.
        return torch.sigmoid(self.head(x))   # (N, 1, H, W) in [0, 1]


# --------------------------------------------------------------------------- #
# Shared preprocessing / target helpers
#   These are module-level so the DETECTOR (inference) and the TRAINER
#   (train_ball.py) use the EXACT same frame preprocessing and heatmap geometry.
#   Keeping them in one place prevents train/infer skew -- a classic, silent cause
#   of "the model trained fine but detects nothing live".
# --------------------------------------------------------------------------- #
def preprocess_frame(frame: np.ndarray, net_w: int, net_h: int) -> np.ndarray:
    """BGR full-frame -> (3, net_h, net_w) RGB float32 in [0, 1] for TrackNetV2."""
    img = cv2.resize(frame, (net_w, net_h))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(img, (2, 0, 1))           # HWC -> CHW


def make_gaussian_heatmap(net_w: int, net_h: int,
                          u_net: Optional[float], v_net: Optional[float],
                          sigma: float = 3.0) -> np.ndarray:
    """Training target: a (net_h, net_w) heatmap with a Gaussian bump (peak 1.0) at
    (u_net, v_net) given in NETWORK pixels. All-zeros when the point is None (the
    ball was labeled 'not visible') -- that teaches the net that 'no ball' is a
    valid answer, which is what makes occlusion handling work."""
    if u_net is None or v_net is None:
        return np.zeros((net_h, net_w), dtype=np.float32)
    ys = np.arange(net_h, dtype=np.float32)[:, None]
    xs = np.arange(net_w, dtype=np.float32)[None, :]
    hm = np.exp(-((xs - u_net) ** 2 + (ys - v_net) ** 2) / (2.0 * sigma * sigma))
    return hm.astype(np.float32)


# --------------------------------------------------------------------------- #
# The wrapper the pipeline actually calls
# --------------------------------------------------------------------------- #
class BallDetector:
    """Stateful TrackNetV2 wrapper.

    STATEFUL on purpose: TrackNet needs the last `in_frames` frames, so the wrapper
    keeps a small rolling buffer and you simply hand it ONE frame at a time in order
    (exactly like core/detector.py). For the first `in_frames`-1 frames the buffer
    is not full yet, so detect() returns found=False with reason="warmup".
    """

    def __init__(
        self,
        weights_path: Optional[str] = None,
        device: str = "cuda",
        input_width: int = 512,
        input_height: int = 288,
        in_frames: int = 3,
        heatmap_threshold: float = 0.5,
        min_blob_area: int = 2,
        court_polygon: Optional[Any] = None,
        roi_margin_px: float = 0.0,
        fp16: bool = False,
    ) -> None:
        self.net_w = int(input_width)
        self.net_h = int(input_height)
        self.in_frames = int(in_frames)
        # Clamp strictly positive: a 0 threshold would select the WHOLE frame as one
        # blob, average it, and risk a 0/0 (NaN) centroid. A tiny floor rejects an
        # empty heatmap cleanly while leaving any sane configured value untouched.
        self.heatmap_threshold = max(float(heatmap_threshold), 1e-6)
        self.min_blob_area = int(min_blob_area)
        # Optional COURT ROI: reject ball detections outside the court polygon
        # (dilated by roi_margin_px) -> kills background lights / out-of-court blobs.
        # The ball flies above the floor and off the glass, so the margin gives the
        # airspace headroom; None = no spatial filter.
        self.court_polygon = court_polygon
        self.roi_margin_px = float(roi_margin_px)

        # --- pick the device, falling back to CPU if CUDA was asked for but is not
        #     actually available (the config sometimes says "CUDA"/"cuda") ---
        want = str(device).lower()
        if "cuda" in want and not torch.cuda.is_available():
            print("[ball] WARNING: device='cuda' requested but CUDA is not "
                  "available -> falling back to CPU.")
            want = "cpu"
        self.device = torch.device("cuda" if "cuda" in want else "cpu")

        # --- half-precision (FP16) is INFERENCE-ONLY and CUDA-ONLY. On a 6 GB GPU it
        #     roughly halves VRAM and speeds the forward pass ~2x with no change to
        #     the detection output. On CPU, FP16 is unsupported/slow, so we silently
        #     stay in FP32 (and say so once if it was asked for). ---
        self.use_fp16 = bool(fp16) and self.device.type == "cuda"
        if fp16 and not self.use_fp16:
            print("[ball] note: fp16 requested but device is CPU -> staying FP32 "
                  "(FP16 is CUDA-only).")

        # --- build the model and (try to) load weights ---
        self.weights_path = weights_path
        self.model = TrackNetV2(in_frames=self.in_frames).to(self.device).eval()
        self.has_weights = self._load_weights(weights_path)
        # Weights load as FP32 (above); cast the whole model to half AFTER loading so
        # the on-disk checkpoint stays FP32 and only the in-memory model runs FP16.
        if self.use_fp16:
            self.model = self.model.half()
        # `operational` = "can this detector actually produce detections right now?"
        # The eval harness keys off THIS (not has_weights) so it works for any
        # backend -- the motion baseline sets operational=True with no weights.
        self.operational = self.has_weights
        if not self.has_weights:
            print("[ball] " + "=" * 64)
            print("[ball] STUB MODE: no ball weights loaded "
                  f"(looked for: {weights_path!r}).")
            print("[ball] The detector will report 'no ball' on EVERY frame until a")
            print("[ball] trained TrackNetV2 weights file is provided. The interface")
            print("[ball] and model are real -- only the weights are missing.")
            print("[ball] See docs/ball_labeling.md to produce padel-ball weights.")
            print("[ball] " + "=" * 64)
        else:
            print(f"[ball] TrackNetV2 weights loaded from {weights_path} "
                  f"(device={self.device}, precision="
                  f"{'fp16' if self.use_fp16 else 'fp32'}, "
                  f"in_frames={self.in_frames}, "
                  f"input={self.net_w}x{self.net_h}).")

        # rolling buffer of the last `in_frames` preprocessed frames (CHW, RGB, 0..1)
        self._buffer: Deque[np.ndarray] = deque(maxlen=self.in_frames)
        # candidates from the LAST detect() (in-ROI blobs, confidence-sorted). The
        # tracker reads these to pick the moving ball over a static distractor.
        self.last_candidates: List[BallDetection] = []
        # raw heatmap peak (0..1) of the LAST detect(), or None on warmup/stub frames.
        # This is the hard-example-mining signal: a LOW peak = a frame the model is unsure
        # about. Captured here (no extra inference) purely for logging -- it does NOT affect
        # detection. (For a found ball it equals the global max; for a miss it's the residual.)
        self.last_heatmap_peak: Optional[float] = None

    # ------------------------------------------------------------------ public
    def detect(self, frame: np.ndarray) -> BallDetection:
        """Push one BGR frame and return this frame's ball result.

        Always returns a BallDetection (never None) so callers have a single,
        explicit object to log -- `found` tells them whether a ball was located.
        """
        self._buffer.append(self._preprocess(frame))

        # stub mode: honest "no ball" everywhere, clearly tagged
        if not self.has_weights:
            self.last_candidates = []
            self.last_heatmap_peak = None
            return BallDetection(found=False, reason="stub-no-weights")

        # not enough frames yet to form the input stack
        if len(self._buffer) < self.in_frames:
            self.last_candidates = []
            self.last_heatmap_peak = None
            return BallDetection(found=False, reason="warmup")

        heatmap = self._infer()                       # (net_h, net_w) in [0, 1]
        self.last_heatmap_peak = float(heatmap.max())  # logging signal only (no behavior change)
        orig_h, orig_w = frame.shape[:2]
        return self._heatmap_to_detection(heatmap, orig_w, orig_h)

    def reset(self) -> None:
        """Clear the frame buffer (call between independent clips so the first
        frames of a new clip do not mix with the tail of the previous one)."""
        self._buffer.clear()

    @property
    def mode_label(self) -> str:
        """Short string describing the detector mode for the eval report."""
        return ("stub-no-weights" if not self.has_weights
                else f"tracknet:{self.weights_path}")

    # ----------------------------------------------------------------- private
    def _load_weights(self, weights_path: Optional[str]) -> bool:
        """Load a state_dict if the file exists. Accepts either a raw state_dict or
        a checkpoint dict that nests it under 'state_dict' / 'model'. Returns True on
        success, False (-> stub mode) if the file is missing or incompatible."""
        if not weights_path or not os.path.isfile(weights_path):
            return False
        try:
            ckpt = torch.load(weights_path, map_location=self.device)
            if isinstance(ckpt, dict):
                state = ckpt.get("state_dict", ckpt.get("model", ckpt))
            else:
                state = ckpt
            # Checkpoints saved under DataParallel/DDP prefix every key with
            # "module." -- strip it so they still line up with this plain model.
            state = {(k[len("module."):] if k.startswith("module.") else k): v
                     for k, v in state.items()}
            # strict=False so a NEAR match still activates detection (we don't want a
            # valid checkpoint to fall back to stub over one renamed buffer). We then
            # inspect what matched and refuse to pretend a totally-wrong file loaded.
            result = self.model.load_state_dict(state, strict=False)
            n_total = len(self.model.state_dict())
            n_loaded = n_total - len(result.missing_keys)
            if n_loaded == 0:
                print(f"[ball] WARNING: weights at {weights_path} matched 0 of "
                      f"{n_total} model tensors -> wrong file? Treating as no "
                      f"weights (stub mode).")
                return False
            if result.missing_keys or result.unexpected_keys:
                print(f"[ball] note: loaded {n_loaded}/{n_total} tensors from "
                      f"{weights_path} (missing={len(result.missing_keys)}, "
                      f"unexpected={len(result.unexpected_keys)}). Detection is ON; "
                      f"if quality is poor, check this is the right checkpoint.")
            self.model.eval()
            return True
        except Exception as exc:  # noqa: BLE001  -- want to degrade to stub, not crash
            print(f"[ball] WARNING: failed to load weights from {weights_path}: "
                  f"{exc} -> running in stub mode.")
            return False

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """BGR full-frame -> (3, net_h, net_w) RGB float32 in [0, 1] for the net.
        Delegates to the shared module helper so training matches inference."""
        return preprocess_frame(frame, self.net_w, self.net_h)

    def _infer(self) -> np.ndarray:
        """Run the model on the current buffer and return the heatmap as numpy.

        The buffer is oldest-first, current-frame-last; we concatenate on the
        channel axis to make the (3*in_frames, H, W) stack TrackNet expects."""
        stack = np.concatenate(list(self._buffer), axis=0)        # (3*in_frames,H,W)
        tensor = torch.from_numpy(stack).unsqueeze(0).to(self.device)  # (1,C,H,W)
        # match the input dtype to the model: FP16 model needs an FP16 input tensor.
        if self.use_fp16:
            tensor = tensor.half()
        with torch.no_grad():
            out = self.model(tensor)                              # (1,1,H,W)
        # cast back to float32 before numpy: cv2/numpy postprocessing expects float32
        # (an FP16 array would break connectedComponents / weighted-centroid maths).
        return out[0, 0].float().detach().cpu().numpy()

    def _heatmap_to_detection(
        self, heatmap: np.ndarray, orig_w: int, orig_h: int
    ) -> BallDetection:
        """Turn a probability heatmap into a single ball point (or 'no ball').

        We scan EVERY blob above threshold (not just the global peak): a bright
        background light can outscore the real ball, so we keep the strongest blob
        whose centre is INSIDE the court (the ROI). That both rejects out-of-court
        lights and recovers the ball when a distractor scored higher. The reported
        point is the chosen blob's INTENSITY-WEIGHTED centroid (sub-pixel), scaled
        from net resolution back to full-frame pixels.
        """
        peak_all = float(heatmap.max())
        if peak_all < self.heatmap_threshold:
            self.last_candidates = []
            return BallDetection(found=False, confidence=peak_all, reason="no-ball")

        mask = (heatmap >= self.heatmap_threshold).astype(np.uint8)
        n_labels, labels, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)

        sx, sy = orig_w / self.net_w, orig_h / self.net_h
        cands: List[Tuple[float, float, float]] = []   # (blob_peak, u, v), in-ROI only
        rejected_by_roi = False
        for lab in range(1, n_labels):
            if stats[lab, cv2.CC_STAT_AREA] < self.min_blob_area:
                continue
            ys, xs = np.where(labels == lab)
            w = heatmap[ys, xs]
            denom = float(np.sum(w))
            if denom <= 0.0:                 # defensive: never divide by zero -> NaN
                continue
            u = float(np.sum(xs * w) / denom) * sx
            v = float(np.sum(ys * w) / denom) * sy
            if not self._in_roi(u, v):
                rejected_by_roi = True       # out-of-court (e.g. a ceiling light)
                continue
            cands.append((float(w.max()), u, v))

        cands.sort(key=lambda c: c[0], reverse=True)     # strongest first
        self.last_candidates = [BallDetection(found=True, u=u, v=v, confidence=p,
                                              reason="ok") for (p, u, v) in cands]
        if not cands:
            reason = "out-of-court" if rejected_by_roi else "no-ball"
            return BallDetection(found=False, confidence=peak_all, reason=reason)
        p, u, v = cands[0]
        return BallDetection(found=True, u=u, v=v, confidence=p, reason="ok")

    def _in_roi(self, u: float, v: float) -> bool:
        """True if (u, v) is inside the court polygon dilated by roi_margin_px.
        No polygon configured -> always True (no spatial filter)."""
        if self.court_polygon is None:
            return True
        dist = cv2.pointPolygonTest(self.court_polygon, (float(u), float(v)), True)
        return dist >= -self.roi_margin_px


# --------------------------------------------------------------------------- #
# Tiny self-test: proves the architecture + postprocessing run WITHOUT weights.
#   python -m core.ball_detector
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("[ball] smoke test: forward pass + postprocessing on random data")
    m = TrackNetV2(in_frames=3).eval()
    dummy = torch.rand(1, 9, 288, 512)
    with torch.no_grad():
        y = m(dummy)
    print(f"  model output shape = {tuple(y.shape)} (expected (1, 1, 288, 512)), "
          f"range [{float(y.min()):.3f}, {float(y.max()):.3f}]")

    # postprocessing on a synthetic heatmap with one bright spot at (200, 100)
    det = BallDetector(weights_path=None, device="cpu")  # stub
    hm = np.zeros((288, 512), dtype=np.float32)
    cv2.circle(hm, (200, 100), 4, 1.0, -1)
    out = det._heatmap_to_detection(hm, orig_w=1024, orig_h=576)
    print(f"  synthetic ball at net(200,100) -> full-frame "
          f"(u={out.u:.1f}, v={out.v:.1f}), conf={out.confidence:.2f}, "
          f"found={out.found}  [expected ~ (400, 200)]")
    print("[ball] smoke test done.")
