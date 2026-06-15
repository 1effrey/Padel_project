"""utils/display.py
Playback-speed control for the live preview window (the `--show` view).

WHY THIS EXISTS
---------------
`cv2.imshow` only paints a frame. The thing that decides how FAST the window
plays is the call right after it: `cv2.waitKey(n)`. `waitKey(n)` shows the frame
and then pauses up to `n` milliseconds waiting for a keypress before the loop
continues. That pause is the ONLY refresh-rate knob OpenCV gives us:

    waitKey(1)   ~1 ms pause   -> effectively NO cap; the window runs as fast as
                                  the detector + tracker + drawing allow.
    waitKey(33)  ~33 ms pause  -> about 30 frames/sec   (1000 / 33 ~= 30)
    waitKey(0)   wait forever  -> step mode: it freezes until you press a key.

THE CATCH (why we don't just hard-code waitKey(33))
---------------------------------------------------
Each frame ALREADY costs time to process (detection is the big one). If that
takes 25 ms and we then waitKey(33), the real per-frame time is 25 + 33 = 58 ms
-> only ~17 fps, not the 30 we asked for. To actually HIT a target fps we must
pause only the *leftover* of the frame's time budget, i.e.

    pause = (1000 / target_fps)  -  time_already_spent_this_frame

This class does exactly that, and clamps the pause to a minimum of 1 ms (0 would
mean "wait forever" to OpenCV).
"""
from __future__ import annotations

from typing import Optional

import cv2


class PlaybackThrottle:
    """Turns a desired on-screen FPS into the correct per-frame `waitKey` pause.

    config["display"]["playback_fps"]:
        0 / null / missing -> no cap: waitKey(1), runs as fast as it can (the
                              original behaviour, so existing runs are unchanged).
        > 0                -> aim for this many frames per second on screen.
    """

    def __init__(self, playback_fps: Optional[float]) -> None:
        # The full time budget for ONE frame, in milliseconds. 30 fps -> 33.3 ms.
        # None means "no throttle".
        if playback_fps and playback_fps > 0:
            self.budget_ms: Optional[float] = 1000.0 / float(playback_fps)
        else:
            self.budget_ms = None

    def wait(self, work_ms: float) -> int:
        """Pause the right amount for this frame and return the key pressed.

        `work_ms` is how long this frame's processing + drawing already took
        (measure it in the loop). Return value is the `cv2.waitKey` key code
        already masked with 0xFF, so callers can compare it to `ord("q")`.
        """
        if self.budget_ms is None:
            delay_ms = 1                      # no cap -> minimal 1 ms pause
        else:
            # spend only what's LEFT of the budget after processing; never < 1 ms
            delay_ms = max(1, int(round(self.budget_ms - work_ms)))
        return cv2.waitKey(delay_ms) & 0xFF
