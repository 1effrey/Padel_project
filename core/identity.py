"""core/identity.py
4-identity ReID layer -- collapse ByteTrack's many raw track_ids into exactly
FOUR permanent player slots for the whole match.

WHY THIS EXISTS
  ByteTrack hands out a fresh id every time a player is occluded, crosses another
  player, or leaves and re-enters the frame -- so over a full clip it produces
  hundreds of ids for the same 4 people. This module sits ON TOP of the tracker
  (it never modifies it) and re-attaches each detection to one of 4 fixed
  identities, so the rest of the pipeline can label players 1..4 consistently.

THE BIG IDEA: content-addressed, not order-based
  We keep a dictionary  profiles = {1:{...}, 2:{...}, 3:{...}, 4:{...}}  whose 4
  keys are PERMANENT. Each frame we match the live detections to those 4 keys by
  how well their TRAITS line up -- never by "who showed up first". A player who
  left and comes back reclaims THEIR key because their colour/position matches it,
  not because a queue handed it to them.

TWO CUES, FUSED
  1. Court position (PRIMARY) -- each detection's feet are projected to court
     METERS via the homography. Two players who overlap in the image are still
     metres apart on the court, so position separates crossings cleanly. But after
     a long absence the stored position is stale (the player walked off and could
     reappear anywhere), so we stop trusting it past `position_ttl_frames`.
  2. Jersey colour (SECONDARY + re-entry anchor) -- an HSV histogram of the torso
     band. Colour is what re-identifies a player who left entirely: position is
     useless then, but the shirt colour persists.

  cost = w_pos * (court_distance_m / court_diagonal_m) + w_color * colour_distance
  (normalised by the active weights so the gate threshold means the same thing
   whether a cell used both cues or colour only.)

THE ASSIGNMENT (Hungarian, with a hard cap)
  We build a 4 x N cost matrix (4 profiles x N detections) and solve it with
  scipy's linear_sum_assignment for the optimal 1-to-1 matching. Because there are
  only 4 rows, AT MOST 4 detections are ever assigned -- there is no "create a 5th
  id" path. If a frame yields >4 detections (false positives), only the best 4 win.
  Any pairing whose cost exceeds `match_threshold` is rejected (left unassigned)
  rather than forced into a bad match.

STATE MACHINE (players may leave)
  0..4 players can be present in a frame -- we never pad the count up to 4. A
  profile not matched this frame becomes "missing" but keeps its last_pos_m and
  colour reserved; it is NEVER deleted or handed to someone else. When a detection
  later matches it (mainly by colour after a gap) the original id is restored.

This is SINGLE-CAMERA ReID only. Fusing both cameras into one global identity set
is a separate later phase and is deliberately NOT attempted here.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.court_position import player_court_position
from utils.homography import COURT_LENGTH_M, COURT_WIDTH_M, Homography
from utils.metrics import NumpyEncoder

Point = Tuple[float, float]

# The longest possible distance on the court -- used to normalise the metric
# distance into 0..1 so it is comparable to the 0..1 colour distance.
COURT_DIAGONAL_M = float(np.hypot(COURT_WIDTH_M, COURT_LENGTH_M))  # ~22.36 m

# HSV torso histogram: hue + saturation only (drop value/brightness so lighting
# changes matter less). These bin counts are a sensible default, not tuned.
_H_BINS, _S_BINS = 30, 32
_HIST_RANGES = [0, 180, 0, 256]   # OpenCV hue is 0..179, saturation 0..255

# A cost we use for "these two cannot be compared at all" -- far above any real
# fused cost (which is <= 1.0) so the Hungarian solver avoids it, and the gate
# below always rejects it.
_IMPOSSIBLE = 1e6


# --------------------------------------------------------------------------- #
# Colour helpers (pure functions -- easy to test on their own)
# --------------------------------------------------------------------------- #
def _torso_band(img_bgr: np.ndarray) -> np.ndarray:
    """Crop the upper-middle 'torso' region out of a player image.

    Both a reference crop (players/pN.png) and a live bbox crop are roughly a
    standing person, so the jersey sits in the same place: horizontally centred
    (avoid the arms at the edges) and in the upper-middle (chest, above the legs).
    """
    h, w = img_bgr.shape[:2]
    y0, y1 = int(0.15 * h), int(0.50 * h)
    x0, x1 = int(0.25 * w), int(0.75 * w)
    return img_bgr[y0:y1, x0:x1]


def _color_hist(img_bgr: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """HSV hue+saturation histogram of the torso band, normalised to 0..1.

    Returns None if there is no usable region (e.g. a zero-size crop from a box
    that fell off the frame edge) so callers can skip the colour cue cleanly.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    band = _torso_band(img_bgr)
    if band.size == 0:
        return None
    hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [_H_BINS, _S_BINS], _HIST_RANGES)
    cv2.normalize(hist, hist, 0.0, 1.0, cv2.NORM_MINMAX)
    return hist


def _color_distance(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    """Bhattacharyya distance between two histograms: 0 = identical, 1 = disjoint.
    Returns None if either histogram is missing (colour cue unavailable)."""
    if a is None or b is None:
        return None
    return float(cv2.compareHist(a.astype("float32"), b.astype("float32"),
                                 cv2.HISTCMP_BHATTACHARYYA))


def _bbox_crop(frame: np.ndarray, bbox) -> Optional[np.ndarray]:
    """Clip the detection box to the frame and return that sub-image (or None)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _slug(source: Any) -> str:
    """Turn a video path / camera index into a safe filename stem for persistence,
    e.g. 'side-1-full-vid.mp4' -> 'side-1-full-vid'."""
    stem = os.path.splitext(os.path.basename(str(source)))[0]
    return re.sub(r"[^A-Za-z0-9._-]", "_", stem) or "camera"


# --------------------------------------------------------------------------- #
# The manager
# --------------------------------------------------------------------------- #
class IdentityManager:
    """Maintains the 4 permanent player profiles and assigns each frame's
    detections to them. One instance per camera/run."""

    def __init__(
        self,
        reid_cfg: Dict[str, Any],
        homog: Optional[Homography],
        source: Any,
        out_dir: str,
        kp_conf_threshold: float = 0.5,
    ) -> None:
        # homog is used only by the single-camera update() convenience wrapper;
        # the fusion pipeline passes None and calls assign() with global positions.
        # --- tunables (all from config, never hard-coded in the logic) -------
        self.w_pos = float(reid_cfg.get("w_pos", 0.6))
        self.w_color = float(reid_cfg.get("w_color", 0.4))
        self.match_threshold = float(reid_cfg.get("match_threshold", 0.5))
        # how long (frames) a stored position stays trustworthy after a player was
        # last seen; past this we fall back to colour-only matching for that profile
        self.position_ttl_frames = int(reid_cfg.get("position_ttl_frames", 90))
        # optional slow blend of the live colour into the stored one (0 = keep the
        # enrolled reference fixed, which is the safe default)
        self.color_ema = float(reid_cfg.get("color_ema", 0.0))
        self.players_dir = reid_cfg.get("players_dir", "players")

        self.homog = homog
        self.kp_conf_threshold = kp_conf_threshold

        # --- the permanent data structure: 4 fixed keys for the whole match ---
        self.profiles: Dict[int, Dict[str, Any]] = {
            i: {
                "id": i,
                "color_hist": None,      # np.ndarray (H x S) or None
                "last_pos_m": None,      # [x_m, y_m] or None until first seen
                "last_seen_frame": -1,
                "state": "missing",      # "active" | "missing"
            }
            for i in (1, 2, 3, 4)
        }

        os.makedirs(out_dir, exist_ok=True)
        self.persist_path = os.path.join(out_dir, f"identities_{_slug(source)}.json")
        # per-assignment log so the weights can be tuned offline (one JSON line/match)
        self._log_path = os.path.join(out_dir, f"identities_log_{_slug(source)}.jsonl")
        self._log_writer = open(self._log_path, "w")

        # Persistence first (identities survive across runs); fall back to fresh
        # enrollment from the reference crops if there is no saved file yet.
        if not self._load():
            self._enroll()

    # -- startup: build the 4 colour profiles from players/pN.png -------------
    def _enroll(self) -> None:
        """Seed each profile's colour histogram from its reference crop. Positions
        stay unknown -- they are learned live once the player is first matched."""
        for i in (1, 2, 3, 4):
            path = os.path.join(self.players_dir, f"p{i}.png")
            img = cv2.imread(path)
            if img is None:
                print(f"[reid] WARNING: could not read reference {path} -> "
                      f"profile {i} has no colour anchor.")
                continue
            self.profiles[i]["color_hist"] = _color_hist(img)
        have = sum(1 for p in self.profiles.values() if p["color_hist"] is not None)
        print(f"[reid] enrolled {have}/4 colour profiles from '{self.players_dir}'.")

    # -- single-camera entry point ------------------------------------------
    def update(self, frame: np.ndarray, tracked: List[Dict[str, Any]],
               frame_idx: int) -> List[Dict[str, Any]]:
        """Assign each detection in `tracked` to one of the 4 ids (single camera).

        Mutates each detection dict in place, adding:
            det["player_id"]  -> int 1..4, or None if left unassigned
            det["match_cost"] -> float cost of the winning match, or None
        Returns the same list for convenience.

        This is the convenience wrapper: it extracts each detection's traits
        (court metres via this manager's homography + torso colour from `frame`)
        and hands them to assign(). The fusion pipeline skips this and calls
        assign() directly with traits it computed itself from TWO cameras.
        """
        dets_pos: List[Optional[List[float]]] = []
        dets_hist: List[Optional[np.ndarray]] = []
        metas: List[Dict[str, Any]] = []
        for det in tracked:
            pos = player_court_position(det, self.homog, self.kp_conf_threshold)
            dets_pos.append(pos["foot_m"])
            dets_hist.append(_color_hist(_bbox_crop(frame, det["bbox"])))
            metas.append({"track_id": det.get("track_id")})

        results = self.assign(dets_pos, dets_hist, frame_idx, metas)
        for det, (pid, cost) in zip(tracked, results):
            det["player_id"] = pid
            det["match_cost"] = cost
        return tracked

    # -- the core matcher (camera-agnostic) ---------------------------------
    def assign(
        self,
        dets_pos: List[Optional[List[float]]],
        dets_hist: List[Optional[np.ndarray]],
        frame_idx: int,
        metas: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Tuple[Optional[int], Optional[float]]]:
        """Match N pre-measured detections to the 4 profiles.

        Inputs are PARALLEL lists already in the right frame/units:
            dets_pos[c]  -> [x_m, y_m] court metres (GLOBAL frame for fusion), or None
            dets_hist[c] -> torso HSV histogram, or None
            metas[c]     -> dict for logging (e.g. {"track_id":.., "camera":..})
        Returns a list of (player_id|None, cost|None) aligned to the inputs.
        Profiles not matched this frame are marked "missing" (kept reserved).
        """
        n = len(dets_pos)
        results: List[Tuple[Optional[int], Optional[float]]] = [(None, None)] * n
        if metas is None:
            metas = [{} for _ in range(n)]

        if n == 0:
            for k in (1, 2, 3, 4):
                self.profiles[k]["state"] = "missing"
            return results

        # 1) build the 4 x N cost matrix and remember the raw cue values
        keys = [1, 2, 3, 4]
        cost = np.full((4, n), _IMPOSSIBLE, dtype=np.float64)
        pos_dist = np.full((4, n), np.nan)
        col_dist = np.full((4, n), np.nan)
        for r, k in enumerate(keys):
            prof = self.profiles[k]
            for c in range(n):
                cell, pd, cd = self._cell_cost(prof, dets_pos[c], dets_hist[c], frame_idx)
                cost[r, c] = cell
                pos_dist[r, c] = pd if pd is not None else np.nan
                col_dist[r, c] = cd if cd is not None else np.nan

        # 2) optimal 1-to-1 matching. With only 4 rows, at most 4 detections can
        #    win -> hard cap of 4 ids, no "new id" path.
        rows, cols = linear_sum_assignment(cost)
        matched_keys = set()
        for r, c in zip(rows, cols):
            k = keys[r]
            cell = float(cost[r, c])
            # 3) gate: reject a pairing that is too poor rather than forcing it
            if cell >= self.match_threshold or cell >= _IMPOSSIBLE * 0.5:
                continue
            results[c] = (k, round(cell, 4))
            matched_keys.add(k)
            self._apply_match(self.profiles[k], dets_pos[c], dets_hist[c],
                              frame_idx, metas[c], cell, pos_dist[r, c], col_dist[r, c])

        # 4) every profile NOT matched this frame -> missing (kept reserved)
        for k in keys:
            if k not in matched_keys:
                self.profiles[k]["state"] = "missing"
        return results

    # -- cost of matching one profile to one detection -----------------------
    def _cell_cost(
        self,
        prof: Dict[str, Any],
        det_pos: Optional[List[float]],
        det_hist: Optional[np.ndarray],
        frame_idx: int,
    ) -> Tuple[float, Optional[float], Optional[float]]:
        """Return (fused_cost, position_distance_m, colour_distance).

        Position is only trusted while it is FRESH (the profile was seen within
        position_ttl_frames). After a long absence we drop the position term and
        let colour drive the match -- that is the re-entry path. When neither cue
        is usable the cost is _IMPOSSIBLE so the gate will reject it.
        """
        # colour distance (0..1) if both sides have a histogram
        cd = _color_distance(prof["color_hist"], det_hist)

        # position distance, only if we have a fresh stored position AND a det pos
        pd_m: Optional[float] = None
        pos_norm: Optional[float] = None
        if prof["last_pos_m"] is not None and det_pos is not None:
            fresh = (frame_idx - prof["last_seen_frame"]) <= self.position_ttl_frames
            if fresh:
                dx = float(det_pos[0]) - float(prof["last_pos_m"][0])
                dy = float(det_pos[1]) - float(prof["last_pos_m"][1])
                pd_m = float(np.hypot(dx, dy))
                pos_norm = min(1.0, pd_m / COURT_DIAGONAL_M)

        # fuse whatever cues are available, normalised by their active weights so
        # the gate threshold means the same thing in every case
        terms = 0.0
        wsum = 0.0
        if pos_norm is not None:
            terms += self.w_pos * pos_norm
            wsum += self.w_pos
        if cd is not None:
            terms += self.w_color * cd
            wsum += self.w_color
        if wsum == 0.0:
            return _IMPOSSIBLE, pd_m, cd
        return terms / wsum, pd_m, cd

    # -- update a profile after it wins a detection --------------------------
    def _apply_match(
        self,
        prof: Dict[str, Any],
        det_pos: Optional[List[float]],
        det_hist: Optional[np.ndarray],
        frame_idx: int,
        meta: Dict[str, Any],
        cost: float,
        pos_dist: float,
        col_dist: float,
    ) -> None:
        """Record the new position/colour for a matched profile and log the match.
        Prints a console line on a RE-ENTRY (missing -> active after a gap).
        `meta` carries source info for the log (track_id and, in fusion, camera)."""
        was_missing = prof["state"] == "missing"
        gap = frame_idx - prof["last_seen_frame"] if prof["last_seen_frame"] >= 0 else -1

        if det_pos is not None:
            prof["last_pos_m"] = [float(det_pos[0]), float(det_pos[1])]
        prof["last_seen_frame"] = frame_idx
        prof["state"] = "active"

        # optional slow colour adaptation (default off -> keep enrolled anchor)
        if self.color_ema > 0.0 and det_hist is not None and prof["color_hist"] is not None:
            blended = (1.0 - self.color_ema) * prof["color_hist"] + self.color_ema * det_hist
            prof["color_hist"] = blended.astype("float32")

        # one JSON line per assignment so weights can be tuned offline
        self._log_writer.write(json.dumps({
            "frame": frame_idx,
            "player_id": prof["id"],
            "track_id": meta.get("track_id"),
            "camera": meta.get("camera"),
            "cost": round(float(cost), 4),
            "pos_dist_m": None if np.isnan(pos_dist) else round(float(pos_dist), 3),
            "color_dist": None if np.isnan(col_dist) else round(float(col_dist), 3),
            "reentry": bool(was_missing and gap > 1),
        }, cls=NumpyEncoder) + "\n")

        # surface re-entries (the headline behaviour) without spamming the console
        if was_missing and gap > 1:
            cam = meta.get("camera")
            src = f"track {meta.get('track_id')}" + (f" cam {cam}" if cam else "")
            print(f"[reid] frame {frame_idx}: player {prof['id']} RE-ENTERS "
                  f"(gap {gap}f, {src}, cost {cost:.3f})")

    # -- persistence ---------------------------------------------------------
    def save(self) -> None:
        """Write the 4 profiles to JSON so identities survive across runs, and
        flush the per-assignment log. Numpy values go through NumpyEncoder."""
        data = {
            "source_slug": _slug(self.persist_path),
            "profiles": {
                str(i): {
                    "id": p["id"],
                    # histograms are big; store as nested lists via NumpyEncoder
                    "color_hist": (p["color_hist"].tolist()
                                   if p["color_hist"] is not None else None),
                    "last_pos_m": p["last_pos_m"],
                    "last_seen_frame": p["last_seen_frame"],
                    "state": p["state"],
                }
                for i, p in self.profiles.items()
            },
        }
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        if self._log_writer is not None:
            self._log_writer.flush()
            self._log_writer.close()
            self._log_writer = None
        print(f"[reid] saved 4 identity profiles -> {self.persist_path}")

    def _load(self) -> bool:
        """Restore profiles from a previous run if the JSON exists. Returns True on
        success so the caller skips fresh enrollment."""
        if not os.path.exists(self.persist_path):
            return False
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)
            for i in (1, 2, 3, 4):
                saved = data["profiles"][str(i)]
                hist = saved.get("color_hist")
                self.profiles[i]["color_hist"] = (
                    np.asarray(hist, dtype="float32") if hist is not None else None)
                self.profiles[i]["last_pos_m"] = saved.get("last_pos_m")
                # positions from a previous run are stale for matching, but we keep
                # them; the position_ttl_frames gate ignores them until refreshed
                self.profiles[i]["last_seen_frame"] = -1
                self.profiles[i]["state"] = "missing"
            print(f"[reid] loaded identity profiles from {self.persist_path}")
            return True
        except (KeyError, ValueError, OSError) as exc:
            print(f"[reid] WARNING: could not load {self.persist_path} ({exc}); "
                  f"re-enrolling from references.")
            return False
