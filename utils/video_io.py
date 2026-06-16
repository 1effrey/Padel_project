"""utils/video_io.py
Open a cv2.VideoCapture with GPU hardware-accelerated decoding (NVDEC) when it is
available, falling back cleanly to CPU decode.

WHY THIS EXISTS
  Profiling showed the pipeline is decode-bound: all video decompression ran on
  the CPU (Task Manager's "Video Decode" engine sat at 0%), which starved the GPU.
  Opening the capture with FFmpeg hardware acceleration routes decode through the
  GPU's dedicated decoder -- on Windows that is D3D11VA/DXVA2, which uses NVDEC
  under the hood -- so the CPU is freed for everything else.

WHAT STAYS THE SAME (important)
  This changes ONLY how the capture is opened. The decoded frame is still handed
  back as a normal BGR numpy array of the same shape/dtype, so `.read()` and every
  downstream stage (detector, tracker, ROI, drawing, minimap) receive byte-
  identical frames and behave exactly as before. A fully zero-copy GPU-resident
  reader (cv2.cudacodec) is intentionally OUT OF SCOPE.
"""
from __future__ import annotations

import queue
import threading
from typing import Any, Optional, Tuple

import cv2


def open_capture(source: Any, hw_accel: bool = True) -> Tuple[cv2.VideoCapture, bool]:
    """Open `source` for decoding, preferring GPU hardware decode.

    Returns (capture, hw_active):
      * hw_active is True ONLY when the FFmpeg backend actually negotiated a
        hardware acceleration mode (the property reads back > NONE).
      * If hardware decode is unavailable (older OpenCV, no HW path, or a source
        the FFmpeg backend can't open such as a webcam index), we fall back to the
        plain default-backend open -- byte-for-byte the original behaviour -- so
        the program still runs. A log line states which path is active.
    """
    # CAP_PROP_HW_ACCELERATION / VIDEO_ACCELERATION_ANY exist on OpenCV >= 4.5.2.
    # Guard so an older build still works (it simply uses the CPU fallback).
    have_consts = (hasattr(cv2, "CAP_PROP_HW_ACCELERATION")
                   and hasattr(cv2, "VIDEO_ACCELERATION_ANY"))

    if hw_accel and have_consts:
        cap = cv2.VideoCapture(
            source, cv2.CAP_FFMPEG,
            [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY],
        )
        # The property reads back the NEGOTIATED mode: NONE (0) means decode stayed
        # on the CPU; any value > NONE (e.g. 2 = D3D11VA -> NVDEC) means the GPU's
        # hardware decoder engaged.
        mode = (cap.get(cv2.CAP_PROP_HW_ACCELERATION)
                if cap.isOpened() else cv2.VIDEO_ACCELERATION_NONE)
        if cap.isOpened() and mode and mode != cv2.VIDEO_ACCELERATION_NONE:
            print(f"[decode] hardware (NVDEC/D3D11VA) acceleration: ON "
                  f"(mode={int(mode)}) -> {source}")
            return cap, True
        # opened on CPU, or failed to open -> drop it and retry the plain path
        cap.release()

    # Plain default-backend open == the original behaviour (works for files,
    # streams, and webcam indices alike).
    cap = cv2.VideoCapture(source)
    print(f"[decode] hardware acceleration not active -> CPU decode -> {source}")
    return cap, False


class ThreadedVideoReader:
    """Decode frames on a BACKGROUND thread so decoding overlaps inference.

    WHY (proven by the profiler, not guessed)
      `cap.read()` is synchronous and costs real wall-clock time even with NVDEC
      (it also copies the 4K frame off the GPU and converts colour) -- ~33 ms/frame
      single, ~52 ms dual. Run sequentially, the GPU sits idle during that read.
      A reader thread fills a small queue while the main loop runs inference on the
      previous frame, so the two overlap. Same frames, same order -> byte-identical
      output; this is pure scheduling.

    SYNC SAFETY
      The queue is bounded and NEVER drops frames (a full queue makes the reader
      wait). Order is FIFO, so the fusion pipeline's frame-exact A<->B alignment is
      preserved -- dropping frames would desync the fixed offset. (For a future
      LIVE-stream "always latest" mode, add a drop-oldest flag; not needed offline.)
    """

    def __init__(self, source: Any, hw_accel: bool = True,
                 queue_size: int = 4, start_frame: int = 0) -> None:
        self.cap, self.hw_active = open_capture(source, hw_accel)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")
        if start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Cache properties NOW (before the thread starts touching the capture), so
        # the main thread never races the reader thread on the same cv2 object.
        self.fps: float = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        self.width: int = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height: int = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self._queue: "queue.Queue" = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _put(self, item: Optional[Any]) -> bool:
        """Block until there is room, but wake every 100 ms to honour stop()."""
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _reader_loop(self) -> None:
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self._put(None)   # end-of-stream sentinel
                break
            if not self._put(frame):
                break

    def read(self) -> Tuple[bool, Optional[Any]]:
        """Pull the next frame. Returns (ok, frame); (False, None) at end-of-stream
        -- same contract as cv2.VideoCapture.read(), so call sites barely change."""
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            return (item is not None), item
        return False, None

    def stop(self) -> None:
        """Signal the thread, drain so a blocked put() can exit, join, release."""
        self._stop.set()
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=2.0)
        self.cap.release()
