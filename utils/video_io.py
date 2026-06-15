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

from typing import Any, Tuple

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
