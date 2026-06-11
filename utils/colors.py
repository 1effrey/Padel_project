"""utils/colors.py
Deterministic, visually-distinct color per track_id (BGR, OpenCV's order).

Same id -> same color on every frame, so a player keeps one consistent color
on screen. This is purely for VISUALISATION; color here has nothing to do with
jersey color or identity (architecture decision #3: color is never an id key).
"""
from __future__ import annotations

from typing import Tuple

# A small palette of high-contrast BGR colors.
_PALETTE: list[Tuple[int, int, int]] = [
    (0, 0, 255),     # red
    (0, 255, 0),     # green
    (255, 0, 0),     # blue
    (0, 255, 255),   # yellow
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (0, 128, 255),   # orange
    (128, 0, 255),   # pink/purple
]


def color_for_id(track_id: int) -> Tuple[int, int, int]:
    """Map a track id to a stable color (cycles if there are many ids)."""
    return _PALETTE[int(track_id) % len(_PALETTE)]
