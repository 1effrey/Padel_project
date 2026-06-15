"""utils/homography.py
Court homography -- map between IMAGE pixels and REAL court meters.

WHY THIS EXISTS
  A single flat plane (the court floor) photographed by a camera is related to a
  top-down metric map of that floor by a 3x3 homography matrix H. Once we know H
  we can answer "where on the court, in meters, is this pixel?" -- which is the
  foundation for bounce position, in/out, and (later) ball speed.

  IMPORTANT LIMITATION (read this):
    H is only correct for points that lie ON the floor plane. A player's FEET are
    on the floor, so pixel_to_meters(feet) is valid. A ball in the AIR is above the
    plane, so its mapped meters are wrong by a parallax amount that grows with the
    ball's height. That is why ball SPEED from one camera is only approximate, while
    a ball BOUNCE (touching the floor) maps correctly. Keep this in mind upstream.

COORDINATE FRAME (meters), per camera (local)
    origin (0,0) ........ the NEAR baseline corner (the end UNDER this camera)
    x axis .............. court WIDTH , 0 .. 10 m   (left .. right in THIS camera's view)
    y axis .............. court LENGTH, 0 .. 20 m   (near .. far from THIS camera)
    net ................. runs across the width at y = 10 m

  TWO-CAMERA SETUP (important):
    Each camera is mounted at one END and shoots ACROSS to the opposite side, so
    what it sees clearly is the FAR 3/4 of the court:
        far baseline (y=20) -> far service line (y=16.95) -> net (y=10) -> ...
    The NEAR baseline (y=0), directly under the camera, does NOT appear in this
    camera's frame -- it is the OTHER camera's job (that camera faces this end).
    So during calibration you start at the FAR baseline and the two NEAR baseline
    corners are the ones you skip.

    There are two recordings of the same court from opposite ends. Each is
    calibrated INDEPENDENTLY into its own local frame above (far baseline = y=20
    for whichever camera you are calibrating). Fusing both into a single top-down
    2D court view is Phase 4: the two local frames are related by a 180 deg
    rotation about the court center, and the net (y=10) is the shared anchor both
    cameras can see. We do NOT fuse here -- we just produce one good homography
    per camera.

  The calibration tool lets you skip landmarks you cannot see, and the drawing
  helpers below silently skip any line point that projects behind the camera or
  to an absurd coordinate -- so the unseen near quarter degrades gracefully.

PUBLIC API
    PADEL_LANDMARKS                                  8 clickable court points (meters)
    compute_homography(image_points, world_points)  -> H, H_inv, mask
    reprojection_error_m(H, image_points, world_points) -> (mean_m, max_m)
    Homography(H[, H_inv])                           .pixel_to_meters / .meters_to_pixel
    Homography.from_config(config)                   -> Homography | None
    is_inside_court(point_m[, margin])               -> bool   (in/out test)
    draw_court_overlay(frame, homog)                 magenta grid + court lines
    draw_court_lines(frame, homog)                   service/net/center/base/side lines
    draw_court_grid(frame, homog)                    1 m validation grid
    to_config_dict(...)                              JSON-ready block for config.json
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]

# --------------------------------------------------------------------------- #
# Padel court model (official dimensions, all in meters)
# --------------------------------------------------------------------------- #
COURT_WIDTH_M = 10.0
COURT_LENGTH_M = 20.0
NET_Y_M = COURT_LENGTH_M / 2.0          # 10.0  -- net across the middle
CENTER_X_M = COURT_WIDTH_M / 2.0        # 5.0   -- central service line

# The service line is 6.95 m from the net (== 3.05 m from the back glass).
SERVICE_FROM_NET_M = 6.95
SERVICE_NEAR_Y_M = NET_Y_M - SERVICE_FROM_NET_M   # 3.05  (near the camera)
SERVICE_FAR_Y_M = NET_Y_M + SERVICE_FROM_NET_M    # 16.95 (far side)

# The 8-point calibration system: 4 court corners + 2 net ends + 2 service-line
# ends. This is the minimal set that pins down the homography well while staying
# fast to click. Ordered FAR -> NEAR to match what the camera actually sees: it
# shoots across to the opposite side, so the FAR baseline (the end facing the
# camera) is the clearest and comes first. The two NEAR baseline corners sit under
# the camera and usually do NOT appear -> skip them in the tool. The 2 service-line
# ends are the FAR service line (the visible one). The fit needs >=4 of the 8.
PADEL_LANDMARKS: List[Tuple[str, Point]] = [
    ("FAR baseline - LEFT corner",    (0.0,           COURT_LENGTH_M)),    # corner 1 (faces cam)
    ("FAR baseline - RIGHT corner",   (COURT_WIDTH_M, COURT_LENGTH_M)),    # corner 2 (faces cam)
    ("FAR service line - LEFT end",   (0.0,           SERVICE_FAR_Y_M)),   # service end 1
    ("FAR service line - RIGHT end",  (COURT_WIDTH_M, SERVICE_FAR_Y_M)),   # service end 2
    ("NET - LEFT end",                (0.0,           NET_Y_M)),           # net end 1
    ("NET - RIGHT end",               (COURT_WIDTH_M, NET_Y_M)),           # net end 2
    ("NEAR baseline - RIGHT corner",  (COURT_WIDTH_M, 0.0)),               # corner 3 (under cam: SKIP)
    ("NEAR baseline - LEFT corner",   (0.0,           0.0)),               # corner 4 (under cam: SKIP)
]

# Court line segments to draw, each as (name, p0_m, p1_m, kind). The "kind" picks
# a color so the net stands out from the service/base/side lines.
def court_lines_m() -> List[Tuple[str, Point, Point, str]]:
    W, L = COURT_WIDTH_M, COURT_LENGTH_M
    return [
        ("sideline_left",   (0.0, 0.0),               (0.0, L),                 "side"),
        ("sideline_right",  (W,   0.0),               (W,   L),                 "side"),
        ("baseline_near",   (0.0, 0.0),               (W,   0.0),               "base"),
        ("baseline_far",    (0.0, L),                 (W,   L),                 "base"),
        ("service_near",    (0.0, SERVICE_NEAR_Y_M),  (W,   SERVICE_NEAR_Y_M),  "service"),
        ("service_far",     (0.0, SERVICE_FAR_Y_M),   (W,   SERVICE_FAR_Y_M),   "service"),
        ("center_service",  (CENTER_X_M, SERVICE_NEAR_Y_M), (CENTER_X_M, SERVICE_FAR_Y_M), "center"),
        ("net",             (0.0, NET_Y_M),           (W,   NET_Y_M),           "net"),
    ]

LINE_COLORS: Dict[str, Tuple[int, int, int]] = {  # BGR
    "side":    (0, 255, 255),   # yellow  -- court boundary
    "base":    (0, 255, 255),   # yellow  -- baselines
    "service": (0, 255, 0),     # green   -- service lines
    "center":  (255, 255, 0),   # cyan    -- central service line
    "net":     (0, 0, 255),     # red     -- the net
}


# --------------------------------------------------------------------------- #
# Core math
# --------------------------------------------------------------------------- #
def compute_homography(
    image_points: Sequence[Point],
    world_points: Sequence[Point],
    ransac_thresh_px: float = 5.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the pixel<->meters homography from clicked correspondences.

    We call cv2.findHomography in the WORLD->IMAGE direction so the RANSAC
    reprojection threshold is expressed in PIXELS (intuitive: "a point may be off
    by up to 5 px"). We then invert to get the pixel->meters matrix the analytics
    actually use.

        H      : pixel  -> meters   (what pixel_to_meters uses)
        H_inv  : meters -> pixel    (what we draw court lines with)

    Returns (H, H_inv, inlier_mask). Use reprojection_error_m() for the quality
    number we report to the user (in meters).
    """
    img = np.asarray(image_points, dtype=np.float64)
    wrld = np.asarray(world_points, dtype=np.float64)
    if len(img) < 4 or len(wrld) < 4:
        raise ValueError(
            f"Need at least 4 point correspondences, got {len(img)}."
        )
    if len(img) != len(wrld):
        raise ValueError("image_points and world_points must be the same length.")

    # world -> image, threshold in pixels
    H_inv, mask = cv2.findHomography(wrld, img, cv2.RANSAC, ransac_thresh_px)
    if H_inv is None:
        raise RuntimeError(
            "findHomography failed -- points are likely collinear/degenerate. "
            "Spread your clicks across width AND length of the court."
        )
    H = np.linalg.inv(H_inv)
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(len(img), bool)
    return H, H_inv, inliers


def reprojection_error_m(
    H: np.ndarray,
    image_points: Sequence[Point],
    world_points: Sequence[Point],
) -> Tuple[float, float]:
    """Calibration quality, in METERS.

    We map each clicked pixel to meters via H and measure how far it lands from
    the landmark's true court coordinate. Meters are the right unit here: "the fit
    is off by ~0.05 m on average" is something you can judge against the 10x20 m
    court. Returns (mean_error_m, max_error_m).
    """
    img = np.asarray(image_points, dtype=np.float64)
    wrld = np.asarray(world_points, dtype=np.float64)
    proj_m = transform_points(H, img)               # pixel -> meters
    err = np.linalg.norm(proj_m - wrld, axis=1)     # meters
    return float(err.mean()), float(err.max())


def transform_points(M: np.ndarray, points: Sequence[Point]) -> np.ndarray:
    """Apply a 3x3 homography to an (N,2) set of points -> (N,2). Uses OpenCV's
    perspectiveTransform (handles the homogeneous divide for us)."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    out = cv2.perspectiveTransform(pts, np.asarray(M, dtype=np.float64))
    return out.reshape(-1, 2)


def _project_with_validity(
    M: np.ndarray, pts: np.ndarray, eps: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    """Project (N,2) points through M and report which are USABLE.

    A homography can map a metric point to "behind the camera" (homogeneous w<=0)
    or way past the horizon (huge coordinates). We compute the divide manually so
    we can flag those instead of drawing nonsense -- this is what makes the far,
    barely-visible quarter of the court degrade gracefully.
    """
    pts = np.asarray(pts, dtype=np.float64)
    homog = np.hstack([pts, np.ones((len(pts), 1))])      # N x 3
    out = (np.asarray(M, dtype=np.float64) @ homog.T).T   # N x 3
    w = out[:, 2]
    valid = w > eps
    px = np.full((len(pts), 2), np.nan)
    px[valid] = out[valid, :2] / w[valid, None]
    # also reject absurd coordinates that would overflow cv2.line
    valid &= np.all(np.abs(px) < 1e5, axis=1)
    return px, valid


def _sample_segment(p0: Point, p1: Point, n: int = 60) -> np.ndarray:
    """n points evenly along the segment p0->p1 (meters). We sample (instead of
    using just the 2 endpoints) so we can drop the part of a line that leaves the
    valid region while still drawing the part that stays in view."""
    a = np.asarray(p0, dtype=np.float64)
    b = np.asarray(p1, dtype=np.float64)
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return a + t * (b - a)


# --------------------------------------------------------------------------- #
# Homography handle used by the rest of the pipeline
# --------------------------------------------------------------------------- #
class Homography:
    """Holds H (pixel->meters) and H_inv (meters->pixel) and exposes the two
    conversions the analytics modules call."""

    def __init__(self, H: Any, H_inv: Optional[Any] = None) -> None:
        self.H = np.asarray(H, dtype=np.float64)
        self.H_inv = (np.asarray(H_inv, dtype=np.float64)
                      if H_inv is not None else np.linalg.inv(self.H))

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Optional["Homography"]:
        """Build from config['homography'], or None if not calibrated yet (so the
        pipeline still runs before homography exists -- same idea as the ROI)."""
        h = config.get("homography")
        if not h or h.get("H") is None:
            return None
        return cls(h["H"], h.get("H_inv"))

    def pixel_to_meters(self, point: Point) -> Point:
        """Image pixel (x, y) -> court meters (x, y). Valid for points on the floor."""
        out = transform_points(self.H, [point])[0]
        return float(out[0]), float(out[1])

    def meters_to_pixel(self, point: Point) -> Point:
        """Court meters (x, y) -> image pixel (x, y)."""
        out = transform_points(self.H_inv, [point])[0]
        return float(out[0]), float(out[1])


def is_inside_court(point_m: Point, margin: float = 0.0) -> bool:
    """In/out test in COURT METERS. True if the point lies on the 10x20 m court.

    `margin` (meters) widens the court before testing: use a small positive margin
    to be lenient near the lines (a ball clipping the line counts as in), or 0 for
    a strict boundary. This is the geometric basis for in/out once a bounce point
    has been converted to meters via pixel_to_meters().
    """
    x, y = point_m
    return (-margin <= x <= COURT_WIDTH_M + margin
            and -margin <= y <= COURT_LENGTH_M + margin)


# --------------------------------------------------------------------------- #
# Drawing
# --------------------------------------------------------------------------- #
def _draw_polyline(frame: np.ndarray, px: np.ndarray, valid: np.ndarray,
                   color: Tuple[int, int, int], thickness: int) -> None:
    """Connect consecutive points, skipping any pair where either end is unusable."""
    for i in range(len(px) - 1):
        if valid[i] and valid[i + 1]:
            p0 = (int(px[i, 0]), int(px[i, 1]))
            p1 = (int(px[i + 1, 0]), int(px[i + 1, 1]))
            cv2.line(frame, p0, p1, color, thickness, cv2.LINE_AA)


def draw_court_lines(frame: np.ndarray, homog: Homography, thickness: int = 2) -> None:
    """Draw the padel court markings (service lines, net, central service line,
    baselines, sidelines) onto `frame` using the meters->pixel homography."""
    for _name, p0, p1, kind in court_lines_m():
        seg = _sample_segment(p0, p1, n=60)
        px, valid = _project_with_validity(homog.H_inv, seg)
        # net (red) and central service line (cyan) are the key references ->
        # draw them a touch thicker so they stand out from the boundary lines
        t = thickness + 1 if kind in ("net", "center") else thickness
        _draw_polyline(frame, px, valid, LINE_COLORS[kind], t)


def draw_court_overlay(frame: np.ndarray, homog: Homography,
                       grid: bool = True, line_thickness: int = 2) -> None:
    """The validation overlay: magenta 1 m grid + the colored court lines on top.

    This is what the calibration/verify commands show so you can eyeball the fit --
    if the homography is right, the grid and lines hug the real court markings."""
    if grid:
        draw_court_grid(frame, homog)
    draw_court_lines(frame, homog, thickness=line_thickness)


def draw_court_grid(frame: np.ndarray, homog: Homography, step: float = 1.0,
                    color: Tuple[int, int, int] = (255, 0, 255),
                    thickness: int = 1) -> None:
    """Draw a magenta 1 m grid over the court. Used by the calibration preview so a
    human can SEE whether the homography is right (grid should hug the court lines)."""
    # lines of constant x (run along the length)
    x = 0.0
    while x <= COURT_WIDTH_M + 1e-9:
        seg = _sample_segment((x, 0.0), (x, COURT_LENGTH_M), n=80)
        px, valid = _project_with_validity(homog.H_inv, seg)
        _draw_polyline(frame, px, valid, color, thickness)
        x += step
    # lines of constant y (run across the width)
    y = 0.0
    while y <= COURT_LENGTH_M + 1e-9:
        seg = _sample_segment((0.0, y), (COURT_WIDTH_M, y), n=80)
        px, valid = _project_with_validity(homog.H_inv, seg)
        _draw_polyline(frame, px, valid, color, thickness)
        y += step


# --------------------------------------------------------------------------- #
# Config serialization
# --------------------------------------------------------------------------- #
def to_config_dict(
    image_points: Sequence[Point],
    world_points: Sequence[Point],
    H: np.ndarray,
    H_inv: np.ndarray,
    err_mean_m: float,
    err_max_m: float,
) -> Dict[str, Any]:
    """Build the JSON-ready 'homography' block. We store the clicked points too so
    the calibration can be reviewed or recomputed later without re-clicking."""
    return {
        "image_points": [[float(x), float(y)] for x, y in image_points],
        "world_points": [[float(x), float(y)] for x, y in world_points],
        "H": [[float(v) for v in row] for row in np.asarray(H)],
        "H_inv": [[float(v) for v in row] for row in np.asarray(H_inv)],
        "reprojection_error_m": {"mean": round(err_mean_m, 4), "max": round(err_max_m, 4)},
        "court_dimensions_m": {"width": COURT_WIDTH_M, "length": COURT_LENGTH_M},
    }
