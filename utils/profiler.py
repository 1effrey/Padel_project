"""utils/profiler.py
Optional per-stage timing for the processing loop -- OFF by default.

WHY (architecture decision #5: measure failures, don't hide them)
  After NVDEC freed the CPU, single-camera FPS is still ~13.7. The bottleneck has
  moved off decode, but we don't yet KNOW where the per-frame time goes. This
  timer breaks one frame into named stages (decode / detect / track / reid / draw
  / ...) and at the end prints the average milliseconds and % of frame time spent
  in each, so we optimise the REAL bottleneck instead of guessing.

USAGE
    timer = StageTimer(enabled)        # enabled comes from the --profile flag
    while ...:
        timer.start_frame()
        ... ; timer.lap("decode")
        ... ; timer.lap("detect")
        timer.end_frame()
    timer.report("single")

WHEN DISABLED
  Every method is a one-line bool check / no-op, so leaving the lap() calls in the
  hot loop costs nothing measurable and behaviour is unchanged.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from typing import Optional


class StageTimer:
    """Accumulates wall-clock time per named stage across many frames."""

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = enabled
        # OrderedDict so the report prints stages in the order they first appear.
        self.totals: "OrderedDict[str, float]" = OrderedDict()
        self.frames = 0
        self._last: Optional[float] = None

    def start_frame(self) -> None:
        """Call once at the very top of each loop iteration."""
        if self.enabled:
            self._last = time.perf_counter()

    def lap(self, name: str) -> None:
        """Record time elapsed since the previous lap()/start_frame() under `name`."""
        if not self.enabled:
            return
        now = time.perf_counter()
        self.totals[name] = self.totals.get(name, 0.0) + (now - self._last)
        self._last = now

    def end_frame(self) -> None:
        """Call once at the end of each fully-processed frame."""
        if self.enabled:
            self.frames += 1

    def report(self, tag: str = "") -> None:
        """Print the per-stage breakdown: avg ms/frame and % of measured time."""
        if not self.enabled or self.frames == 0:
            return
        total = sum(self.totals.values())
        label = f" {tag}" if tag else ""
        print(f"[profile{label}] {self.frames} frames | "
              f"{1000.0 * total / self.frames:.1f} ms/frame measured "
              f"(~{self.frames / total:.1f} FPS if nothing else):")
        for name, secs in self.totals.items():
            ms = 1000.0 * secs / self.frames
            pct = 100.0 * secs / total if total > 0 else 0.0
            print(f"    {name:<16} {ms:7.2f} ms/frame   {pct:5.1f}%")
