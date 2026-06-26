"""core/calibrate_boundaries.py
Phase-3/4 BOUNDARY MARKING TOOLS -- draw, per camera, the image-space regions the
ball-event detector needs to tell one kind of contact from another.

The FLOOR is NOT marked here -- floor in/out comes from the HOMOGRAPHY
(is_inside_court on the mapped metres); we only DISPLAY the homography court as context.

  --calibrate-walls  -> court.walls   GLASS wall regions (ball deflects -> stays IN). Each
                                      panel is a QUAD you mark by CLICKING ITS 4 CORNERS
                                      (so it hugs the perspective-angled glass). Draw THREE
                                      (BACK + LEFT + RIGHT); on save they are JOINED into one
                                      perimeter -- where panels touch, the seam is removed.
  --calibrate-fence  -> court.fence   METAL fence regions (ball hits -> OUT). Two quads
                                      (LEFT + RIGHT), 4 corners each. They don't touch, so
                                      they stay two regions.
  --calibrate-net    -> court.net_polygon   the WHOLE net perimeter, a freeform polygon.

Marking by the 4 real corners is precise on a perspective view (an axis-aligned box can't
hug the slanted side glass). Every region is stored in FULL-RESOLUTION image pixels (the
space the ball track lives in), so the detector classifies a contact with a plain point-in-
polygon test. These tools ONLY mark + save; wiring the events is a separate, later step.

CONTROLS  (quad tools: walls, fence)
  Left-click ... place a corner; after the 4th the panel auto-completes and the next begins
  Right-click .. undo the last corner (empty panel -> step back to the previous one)
  S ............ SAVE (join/union touching panels) and quit       Q/Esc .. quit, no save
CONTROLS  (net: freeform polygon)
  Left-click ... add a point     Right-click .. undo last point     S .. save     Q/Esc .. quit
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from utils.homography import Homography, draw_court_lines

_FONT = cv2.FONT_HERSHEY_SIMPLEX

# dim colours used to show OTHER, already-saved regions as context while you edit one.
_CONTEXT = {
    "walls": (120, 90, 0),
    "fence": (0, 0, 120),
    "net_polygon": (120, 0, 120),
}


def _read_first_frame(config: Dict[str, Any]) -> np.ndarray:
    cap = cv2.VideoCapture(config["source"])
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {config['source']}")
    return frame


def _saved_polys(court: Dict[str, Any], key: str, single: bool) -> List[List[List[int]]]:
    """Existing polygons for `key` as a list of polygons (so we can EDIT, not redraw)."""
    saved = court.get(key)
    if not saved:
        return []
    if single:
        return [[list(map(int, pt)) for pt in saved]]
    return [[list(map(int, pt)) for pt in poly] for poly in saved]


def _draw_poly(img, poly, color, scale, close=False):
    """Draw one polygon (full-res points) onto a downscaled display image."""
    for i, p in enumerate(poly):
        d = (int(p[0] * scale), int(p[1] * scale))
        cv2.circle(img, d, 4, color, -1)
        if i > 0:
            prev = (int(poly[i - 1][0] * scale), int(poly[i - 1][1] * scale))
            cv2.line(img, prev, d, color, 2)
    if close and len(poly) >= 3:                       # hint the closing edge
        a = (int(poly[-1][0] * scale), int(poly[-1][1] * scale))
        b = (int(poly[0][0] * scale), int(poly[0][1] * scale))
        cv2.line(img, a, b, color, 1)


def _merge_union(polys: List[List[List[int]]], w: int, h: int) -> List[List[List[int]]]:
    """Rasterise all quads into a mask and trace the OUTER contours -- so panels that TOUCH
    merge into one perimeter (internal seams removed) while separate ones stay apart.
    Returns the resulting perimeter polygons (full-res points)."""
    mask = np.zeros((h, w), np.uint8)
    for p in polys:
        if len(p) >= 3:
            cv2.fillPoly(mask, [np.array(p, np.int32)], 255)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[List[List[int]]] = []
    for c in cnts:
        eps = 0.01 * cv2.arcLength(c, True)            # light simplify -> clean corners
        ap = cv2.approxPolyDP(c, eps, True)
        out.append([[int(pt[0][0]), int(pt[0][1])] for pt in ap])
    return out


def _region_label(i: int, names: Optional[List[str]]) -> str:
    if names and i < len(names):
        return names[i]
    return f"region #{i + 1}"


def _edit_polygons(config: Dict[str, Any], config_path: str, *, court_key: str,
                   title: str, color, single: bool = False,
                   corners_per_region: Optional[int] = None, merge: bool = False,
                   region_names: Optional[List[str]] = None) -> None:
    """Shared click editor for ONE court_key. With `corners_per_region` (e.g. 4) each region
    auto-completes after that many corner clicks, then the next region begins (walls/fence).
    Without it, you click a single freeform polygon (net). Shows the homography floor + other
    saved regions as context. `merge` unions touching regions into one perimeter on save."""
    frame = _read_first_frame(config)
    full_h, full_w = frame.shape[:2]
    court = config.setdefault("court", {})
    homog = Homography.from_config(config)

    scale = min(1.0, 1280 / full_w)
    disp_w, disp_h = int(full_w * scale), int(full_h * scale)

    # bake CONTEXT once: homography court (floor) + other saved regions, dim.
    ctx = frame.copy()
    if homog is not None:
        draw_court_lines(ctx, homog, thickness=2)
    for k, ccol in _CONTEXT.items():
        if k == court_key:
            continue
        for poly in _saved_polys(court, k, single=(k == "net_polygon")):
            _draw_poly(ctx, poly, ccol, 1.0)
    base = cv2.resize(ctx, (disp_w, disp_h))

    polys = _saved_polys(court, court_key, single)
    if not polys:
        polys = [[]]                                   # one active polygon to click into
    pending: Dict[str, Any] = {"add": None, "undo": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pending["add"] = (int(round(x / scale)), int(round(y / scale)))
        elif event == cv2.EVENT_RBUTTONDOWN:
            pending["undo"] = True

    win = f"calibrate: {court_key}"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, disp_w, disp_h)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        if pending["add"] is not None:
            polys[-1].append(list(pending["add"]))
            # quad mode: a region auto-completes at N corners -> start the next one
            if (not single and corners_per_region
                    and len(polys[-1]) >= corners_per_region):
                polys.append([])
            pending["add"] = None
        if pending["undo"]:
            if polys[-1]:
                polys[-1].pop()
            elif len(polys) > 1:
                polys.pop()                            # step back into the previous region
            pending["undo"] = False

        disp = base.copy()
        for poly in polys:
            _draw_poly(disp, poly, color, scale, close=True)

        n_done = sum(1 for p in polys if len(p) >= 3)
        nxt = _region_label(len(polys) - 1, region_names)
        cv2.putText(disp, title, (10, 28), _FONT, 0.7, color, 2)
        if single:                                     # net: one freeform polygon
            sub = f"freeform  |  points: {len(polys[-1])}"
            ctrl = "L:add point  R:undo  S:save  Q:quit"
        elif corners_per_region:                       # walls: 4-corner quads, auto-next
            sub = (f"{n_done} panel(s) done  |  now: {nxt}  "
                   f"({len(polys[-1])}/{corners_per_region} corners)")
            ctrl = "L:click 4 corners (auto-next)  R:undo corner  S:save (joins)  Q:quit"
        else:                                          # fence: freeform polygon per region
            sub = f"{n_done} region(s) done  |  now: {nxt}  ({len(polys[-1])} points)"
            ctrl = "L:add point  N:next region  R:undo  S:save  Q:quit"
        cv2.putText(disp, sub, (10, 52), _FONT, 0.55, (235, 235, 235), 1)
        cv2.putText(disp, ctrl, (10, disp_h - 14), _FONT, 0.55, (200, 200, 200), 2)
        cv2.imshow(win, disp)

        key = cv2.waitKey(20) & 0xFF
        if key == ord("n") and not single and not corners_per_region:
            if len(polys[-1]) >= 3:                    # freeform multi: finish region, next
                polys.append([])
        elif key == ord("s"):
            _save(config, config_path, court_key, polys, single,
                  merge=merge, dims=(full_w, full_h))
            break
        elif key in (ord("q"), 27):
            print(f"[calibrate:{court_key}] cancelled, nothing saved.")
            break

    cv2.destroyAllWindows()


def _save(config: Dict[str, Any], config_path: str, court_key: str,
          polys: List[List[List[int]]], single: bool,
          merge: bool = False, dims: Optional[Tuple[int, int]] = None) -> None:
    """Write the complete polygons into config['court'][court_key] and disk. With merge=True
    the panels are unioned (touching ones joined, seams removed) before saving."""
    court = config.setdefault("court", {})
    complete = [p for p in polys if len(p) >= 3]
    if single:
        if complete:
            court[court_key] = complete[0]
        msg = "set" if complete else "kept (nothing new drawn)"
    else:
        if merge and complete and dims is not None:
            complete = _merge_union(complete, dims[0], dims[1])
            msg = f"{len(complete)} joined region(s)"
        else:
            msg = f"{len(complete)} region(s)"
        court[court_key] = complete
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[calibrate:{court_key}] saved to {config_path}: {msg}")


# --------------------------------------------------------------------------- #
# the three public tools -- thin wrappers around the shared editor
# --------------------------------------------------------------------------- #
def calibrate_walls(config: Dict[str, Any], config_path: str) -> None:
    """Mark the THREE glass wall panels (BACK + LEFT + RIGHT) by clicking each one's 4
    corners; on save they are joined into one continuous wall perimeter (seams removed)."""
    _edit_polygons(config, config_path, court_key="walls",
                   title="GLASS WALLS  (4 corners x BACK, LEFT, RIGHT -> joined)",
                   color=(255, 200, 0), single=False, corners_per_region=4, merge=True,
                   region_names=["BACK wall", "LEFT wall", "RIGHT wall"])


def calibrate_fence(config: Dict[str, Any], config_path: str) -> None:
    """Mark the metal fence panels (LEFT + RIGHT) as FREEFORM polygons -> OUT regions.
    Click points around the LEFT panel, press N, click the RIGHT panel. Stored as drawn
    (no union -- the two panels are separate 'out' regions)."""
    _edit_polygons(config, config_path, court_key="fence",
                   title="METAL FENCE  (freeform polygon: LEFT, N, RIGHT = OUT)",
                   color=(0, 0, 255), single=False, corners_per_region=None, merge=False,
                   region_names=["LEFT fence", "RIGHT fence"])


def calibrate_net(config: Dict[str, Any], config_path: str) -> None:
    """Mark the WHOLE net perimeter (one freeform polygon) -> deflect/slow here = net_hit."""
    _edit_polygons(config, config_path, court_key="net_polygon",
                   title="NET PERIMETER (whole net)", color=(255, 0, 255),
                   single=True, corners_per_region=None)


if __name__ == "__main__":
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else "walls"
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config-side1.json"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    {"walls": calibrate_walls, "fence": calibrate_fence,
     "net": calibrate_net}[which](cfg, cfg_path)
