"""core/camera_calib.py
Phase-4 CAMERA CALIBRATION + TRIANGULATION.

GOAL
  Turn each camera's floor HOMOGRAPHY (which we already have) into a full pinhole
  model  P = K [R | t]  so we can TRIANGULATE the ball into real 3D when both
  cameras see it -- the foundation for the fused 3D ball and the Phase-5 physics.

HOW (all from data we already have -- no new calibration target)
  K (intrinsics) from COURT VANISHING POINTS:
    The court has two families of world-parallel lines that are PERPENDICULAR:
    width-direction lines (baselines/net/service, constant y) and length-direction
    lines (sidelines/centre, constant x). Their images meet at two vanishing points
    vp_x, vp_y. For a camera with square pixels, zero skew and principal point at the
    image centre p0, perpendicular directions satisfy
        f^2 = -(vp_x - p0) . (vp_y - p0)
    which gives the focal length in pixels. (We get the vanishing points for free by
    sending the world directions [1,0,0] / [0,1,0] through H_inv.)

  R, t (pose) from the HOMOGRAPHY (Zhang's decomposition):
    The floor homography H_inv maps world floor points (X,Y,0) to image, and equals
    K [r1 r2 t] up to scale. So [r1 r2 t] = K^-1 H_inv / lambda; r3 = r1 x r2; then
    we orthonormalise R. A sign fix puts the court IN FRONT of the camera.

  SHARED COURT FRAME:
    Each camera was calibrated in its OWN local frame (far baseline = y=20). Side-1's
    local frame IS the global frame; side-2's is the 180-degrees rotation about the
    court centre, (x,y) -> (10-x, 20-y) -- the same relation core/fusion.py uses for
    players. We compose that into side-2's pose so both P matrices live in ONE frame.

  TRIANGULATE:
    Given both P matrices and the ball pixel in each camera at the SAME synced frame,
    linear (DLT) triangulation gives the 3D point. We report a numeric covariance
    (perturb the pixels, see how 3D moves) -- it comes out ANISOTROPIC: the cameras
    sit at opposite ends, so the court-LENGTH axis (Y) is weakly constrained, exactly
    as the data realities warned. Epipolar-inconsistent matches (big reprojection
    error) are rejected.

HONEST LIMITS
  Intrinsics from two vanishing points assume the principal point is the image centre;
  if a vanishing point sits near infinity the focal estimate is ill-conditioned (we
  flag it). The self-test (`python -m core.camera_calib`) reports the reprojection
  error so you can SEE how good the model is before trusting it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from utils.homography import COURT_LENGTH_M, COURT_WIDTH_M

# 180-degrees rotation about the court centre, as a 3D rigid transform on the floor:
# (x, y, z) -> (10 - x, 20 - y, z). Used to put side-2's LOCAL frame into the GLOBAL one.
_R180 = np.diag([-1.0, -1.0, 1.0])
_T180 = np.array([COURT_WIDTH_M, COURT_LENGTH_M, 0.0])


@dataclass
class CameraModel:
    """A calibrated pinhole camera in the GLOBAL court frame."""

    K: np.ndarray            # 3x3 intrinsics
    R: np.ndarray            # 3x3 rotation (world -> camera)
    t: np.ndarray            # 3   translation (world -> camera)
    f: float = 0.0           # focal length (px), for reporting
    intrinsics_stable: bool = True

    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix K[R|t]."""
        return self.K @ np.hstack([self.R, self.t.reshape(3, 1)])

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coords, C = -R^T t."""
        return -self.R.T @ self.t

    def project(self, X: np.ndarray) -> np.ndarray:
        """World point (3,) -> image pixel (2,)."""
        x = self.P @ np.append(np.asarray(X, dtype=float), 1.0)
        return x[:2] / x[2]


# --------------------------------------------------------------------------- #
# Intrinsics + pose
# --------------------------------------------------------------------------- #
def _vanishing_point(H_inv: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Image of the world direction (dx, dy, 0) at infinity -> a vanishing point."""
    v = H_inv @ np.array([dx, dy, 0.0])
    return v[:2] / v[2]


def intrinsics_from_court(H_inv: np.ndarray, img_w: int, img_h: int
                          ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """K from the two perpendicular court vanishing points (principal point = centre)."""
    p0 = np.array([img_w / 2.0, img_h / 2.0])
    vpx = _vanishing_point(H_inv, 1.0, 0.0)        # width-direction vanishing point
    vpy = _vanishing_point(H_inv, 0.0, 1.0)        # length-direction vanishing point
    f2 = -float(np.dot(vpx - p0, vpy - p0))
    stable = f2 > 0.0
    f = float(np.sqrt(f2)) if stable else float(img_w)   # fallback ~50deg FOV, flagged
    K = np.array([[f, 0.0, p0[0]], [0.0, f, p0[1]], [0.0, 0.0, 1.0]])
    return K, {"f": f, "vp_x": vpx, "vp_y": vpy, "stable": stable}


def pose_from_homography(H_inv: np.ndarray, K: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Recover (R, t) from the floor homography H_inv (world->image) and K."""
    L = np.linalg.inv(K) @ H_inv                   # = lambda [r1 r2 t]
    lam = 1.0 / np.linalg.norm(L[:, 0])
    r1, r2, t = lam * L[:, 0], lam * L[:, 1], lam * L[:, 2]
    # Sign: the court centre must be IN FRONT of the camera (positive depth).
    if (np.column_stack([r1, r2, np.cross(r1, r2)])
            @ np.array([COURT_WIDTH_M / 2, COURT_LENGTH_M / 2, 0.0]) + t)[2] < 0:
        r1, r2, t = -r1, -r2, -t
    R_raw = np.column_stack([r1, r2, np.cross(r1, r2)])
    # nearest valid rotation (orthonormal, det = +1)
    U, _, Vt = np.linalg.svd(R_raw)
    R = U @ np.diag([1.0, 1.0, float(np.linalg.det(U @ Vt))]) @ Vt
    return R, t


def _refine_camera(K: np.ndarray, R: np.ndarray, t: np.ndarray,
                   obj: np.ndarray, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Least-squares refine [focal, principal point, pose] to minimise reprojection
    error on the court correspondences. Returns (K, R, t, mean_reproj_px)."""
    from scipy.optimize import least_squares

    rvec0 = cv2.Rodrigues(R)[0].reshape(3)
    x0 = np.concatenate([[K[0, 0], K[0, 2], K[1, 2]], rvec0, t])
    ones = np.ones((len(obj), 1))
    obj_h = np.hstack([obj, ones])

    def resid(x: np.ndarray) -> np.ndarray:
        f, cx, cy = x[0], x[1], x[2]
        Kp = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        Rp = cv2.Rodrigues(x[3:6])[0]
        P = Kp @ np.hstack([Rp, x[6:9].reshape(3, 1)])
        proj = (P @ obj_h.T).T
        proj = proj[:, :2] / proj[:, 2:3]
        return (proj - img).ravel()

    sol = least_squares(resid, x0, method="lm")
    x = sol.x
    K2 = np.array([[x[0], 0, x[1]], [0, x[0], x[2]], [0, 0, 1.0]])
    R2 = cv2.Rodrigues(x[3:6])[0]
    t2 = x[6:9]
    mean_px = float(np.mean(np.linalg.norm(resid(x).reshape(-1, 2), axis=1)))
    return K2, R2, t2, mean_px


def build_camera(config: Dict[str, Any], img_w: int = 3840, img_h: int = 2160,
                 is_side2: bool = False, focal_override: Optional[float] = None
                 ) -> Tuple[CameraModel, Dict[str, Any]]:
    """Build a CameraModel in the GLOBAL frame from a config's homography block.

    Pose comes from cv2.solvePnP on the clicked court correspondences (this REFINES
    the raw homography decomposition to minimise reprojection error). If the
    vanishing-point focal is unstable, pass `focal_override` (e.g. the other camera's
    focal -- the two ends usually use the same camera)."""
    h = config.get("homography")
    if not h or h.get("H_inv") is None or not h.get("image_points"):
        raise ValueError("config has no homography correspondences -> cannot calibrate.")
    H_inv = np.asarray(h["H_inv"], dtype=float)    # meters -> pixel

    K, diag = intrinsics_from_court(H_inv, img_w, img_h)
    diag["focal_overridden"] = False
    if focal_override is not None and not diag["stable"]:
        K[0, 0] = K[1, 1] = float(focal_override)
        diag["f"] = float(focal_override)
        diag["focal_overridden"] = True

    # pose by PnP on the court landmarks (in this camera's LOCAL world frame)
    obj = np.array([[wp[0], wp[1], 0.0] for wp in h["world_points"]], dtype=np.float64)
    img = np.array(h["image_points"], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed to recover the camera pose.")
    R = cv2.Rodrigues(rvec)[0]
    t = tvec.reshape(3)
    # refine focal + principal point + pose to minimise reprojection (court cameras
    # have an off-centre principal point the vanishing-point guess misses)
    K, R, t, diag["reproj_px"] = _refine_camera(K, R, t, obj, img)
    diag["f"] = float(K[0, 0])
    if is_side2:                                    # local -> global (180-degrees)
        t = R @ _T180 + t                          # use R_local BEFORE rotating it
        R = R @ _R180
    return CameraModel(K=K, R=R, t=t, f=diag["f"], intrinsics_stable=diag["stable"]), diag


# --------------------------------------------------------------------------- #
# Triangulation
# --------------------------------------------------------------------------- #
def triangulate(camA: CameraModel, camB: CameraModel,
                uvA: Tuple[float, float], uvB: Tuple[float, float]) -> np.ndarray:
    """Linear (DLT) triangulation -> world point (3,)."""
    rows = []
    for cam, (u, v) in ((camA, uvA), (camB, uvB)):
        P = cam.P
        rows.append(u * P[2] - P[0])
        rows.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(rows))
    X = Vt[-1]
    return X[:3] / X[3]


def reprojection_error(cam: CameraModel, uv: Tuple[float, float], X: np.ndarray) -> float:
    return float(np.linalg.norm(cam.project(X) - np.asarray(uv, dtype=float)))


def triangulation_covariance(camA: CameraModel, camB: CameraModel,
                             uvA: Tuple[float, float], uvB: Tuple[float, float],
                             sigma_px: float = 2.0) -> np.ndarray:
    """3x3 covariance of the triangulated point, by propagating a sigma_px pixel noise
    through the triangulation numerically. Comes out anisotropic (Y weakest)."""
    base = triangulate(camA, camB, uvA, uvB)
    flat = [uvA[0], uvA[1], uvB[0], uvB[1]]
    J = np.zeros((3, 4))
    for i in range(4):
        d = list(flat)
        d[i] += sigma_px
        Xp = triangulate(camA, camB, (d[0], d[1]), (d[2], d[3]))
        J[:, i] = (Xp - base) / sigma_px
    return J @ (sigma_px ** 2 * np.eye(4)) @ J.T


def triangulate_ball(camA: CameraModel, camB: CameraModel,
                     uvA: Tuple[float, float], uvB: Tuple[float, float],
                     max_reproj_px: float = 40.0) -> Optional[Dict[str, Any]]:
    """Triangulate + reject epipolar-inconsistent matches. Returns a dict with the 3D
    point, per-camera reprojection error and covariance, or None if rejected."""
    X = triangulate(camA, camB, uvA, uvB)
    eA = reprojection_error(camA, uvA, X)
    eB = reprojection_error(camB, uvB, X)
    if max(eA, eB) > max_reproj_px:
        return None                                # rays don't meet -> wrong match
    cov = triangulation_covariance(camA, camB, uvA, uvB)
    return {"X": X, "reproj_px": max(eA, eB), "cov": cov,
            "std": np.sqrt(np.clip(np.diag(cov), 0.0, None))}


# --------------------------------------------------------------------------- #
# Self-test: verify the calibration by reprojecting court points + round-trip 3D.
#   python -m core.camera_calib
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json

    def _reproj_err(cam, h, to_global):
        errs = []
        for ip, wp in zip(h["image_points"], h["world_points"]):
            g = to_global(wp)
            errs.append(np.linalg.norm(cam.project([g[0], g[1], 0.0]) - np.asarray(ip)))
        return float(np.mean(errs)), float(np.max(errs))

    cfgA = json.load(open("config-side1.json"))
    cfgB = json.load(open("config-side2.json"))
    camA, dA = build_camera(cfgA, is_side2=False)
    # the two ends usually share a camera model -> reuse side-1's focal if side-2's
    # vanishing-point estimate is unstable
    camB, dB = build_camera(cfgB, is_side2=True, focal_override=dA["f"])
    print(f"side-1: f={dA['f']:.0f}px stable={dA['stable']}  vp_x={dA['vp_x'].round(0)} "
          f"vp_y={dA['vp_y'].round(0)}  centre(m)={camA.center.round(2)}")
    print(f"side-2: f={dB['f']:.0f}px stable={dB['stable']}  centre(m)={camB.center.round(2)}")

    mA = _reproj_err(camA, cfgA["homography"], lambda wp: (wp[0], wp[1]))            # side-1 local=global
    mB = _reproj_err(camB, cfgB["homography"],
                     lambda wp: (COURT_WIDTH_M - wp[0], COURT_LENGTH_M - wp[1]))     # side-2 local->global
    print(f"side-1 court-point reprojection error (px): mean={mA[0]:.1f} max={mA[1]:.1f}")
    print(f"side-2 court-point reprojection error (px): mean={mB[0]:.1f} max={mB[1]:.1f}")

    Xtrue = np.array([5.0, 10.0, 2.0])              # 2 m above the net centre (global)
    uvA, uvB = camA.project(Xtrue), camB.project(Xtrue)
    res = triangulate_ball(camA, camB, tuple(uvA), tuple(uvB))
    if res is None:
        print("round-trip triangulation REJECTED (bad calibration)")
    else:
        print(f"triangulate round-trip: true={Xtrue} recovered={res['X'].round(3)} "
              f"err={np.linalg.norm(res['X']-Xtrue):.3f} m  reproj={res['reproj_px']:.2f}px")
        print(f"  covariance std (x,y,z) m = {res['std'].round(3)}  "
              f"<- Y should be the LARGEST (court-length axis weakly seen)")
