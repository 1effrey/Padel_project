"""core/fusion.py
Phase-4 CROSS-CAMERA fusion: merge the two end-cameras into ONE global set of
4 player identities on a single top-down court.

PIPELINE (offline, one synced frame at a time)

    read A[f] + B[f+offset]              # time sync (clock-derived, config)
      -> detect + ROI filter (each)
      -> feet -> metres (each camera's own homography)
      -> map side-2 metres into the GLOBAL frame via rot180: (x,y)->(10-x,20-y)
      -> sanity-drop off-court projections (homography extrapolation garbage)
      -> DEDUP the overlap (a mid-court player seen by both -> ONE detection,
         keeping the CLOSER camera's view = bigger bbox)
      -> ONE IdentityManager.assign() over the merged <=4 detections -> ids 1..4
      -> render one fused minimap + log positions

WHY EACH PIECE (verified by measurement before this module was written)
  * offset (config "sync"): side-B frame = side-A frame + offset. Clock-exact.
  * rot180: the two local metre-frames are a 180-degree rotation about court
    centre (5,10); verified empirically (net-region players coincide to ~0.8 m).
  * camera coverage (MEASURED, not assumed): each camera is mounted at one END
    and sees the WHOLE court in front of it -- players are big/reliable when near
    that camera and small when far. Side-1 (y=0 end) detects global y~6..19;
    side-2 (y=20 end) detects global y~1..14. The two OVERLAP in the middle
    (y~6..13) and each EXCLUSIVELY sees the far end (side-1: y>13, side-2: y<5).
    So we must NOT drop a camera's far-field detections (they are the only view of
    those players -- the earlier "near-half ownership drop" deleted 3 of 4
    players). Instead we keep everything and, where both cameras see the same
    player (the overlap), DEDUP to the closer camera's view (bigger bbox). That
    closer-camera preference is the correct reading of architecture decision #2.

This module ADDS a layer; it imports the detector, ROI, homography, court-position,
identity and minimap modules unchanged.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from core.detector import PoseDetector
from core.identity import IdentityManager, _bbox_crop, _color_hist
from utils import roi as roi_utils
from utils.colors import color_for_id
from utils.court_position import foot_pixel
from utils.display import PlaybackThrottle
from utils.homography import COURT_LENGTH_M, COURT_WIDTH_M, Homography
from utils.metrics import NumpyEncoder
from utils.minimap import Minimap

NET_Y_M = COURT_LENGTH_M / 2.0     # 10.0 -- the half boundary


def to_global(cam: str, foot_m: Tuple[float, float]) -> Tuple[float, float]:
    """Map a camera's LOCAL court metres into the shared GLOBAL frame.

    Global frame == side-1's local frame. Side-1 points pass through unchanged;
    side-2 points are rotated 180 degrees about court centre (5,10): the verified
    relationship between the two independently-calibrated cameras.
    """
    x, y = float(foot_m[0]), float(foot_m[1])
    if cam == "B":
        return (COURT_WIDTH_M - x, COURT_LENGTH_M - y)
    return (x, y)


class FusionPipeline:
    """Drives the two clips in lockstep and produces one 4-identity court map."""

    def __init__(self, cfg_a: Dict[str, Any], cfg_b: Dict[str, Any]) -> None:
        self.cfg_a, self.cfg_b = cfg_a, cfg_b
        self.kp_thr = cfg_a.get("skeleton", {}).get("keypoint_conf_threshold", 0.5)
        self.out_dir = cfg_a.get("output", {}).get("dir", "output")
        os.makedirs(self.out_dir, exist_ok=True)

        # time sync (required): side-B frame = side-A frame + offset
        sync = cfg_a.get("sync")
        if not sync or sync.get("offset_frames") is None:
            raise RuntimeError("config-A needs a 'sync' block (run main.py --sync first).")
        self.offset = int(sync["offset_frames"])

        # both homographies are required (we work in metres)
        self.hom_a = Homography.from_config(cfg_a)
        self.hom_b = Homography.from_config(cfg_b)
        if self.hom_a is None or self.hom_b is None:
            raise RuntimeError("Both cameras need a calibrated homography for fusion.")

        # fusion tunables (config-driven). Reuse the reid block + a fusion block.
        reid_cfg = cfg_a.get("reid", {})
        fus = cfg_a.get("fusion", {})
        self.dedup_m = float(fus.get("dedup_m", 1.5))             # same-player merge radius
        self.bounds_margin_m = float(fus.get("bounds_margin_m", 2.0))  # off-court reject margin

        self.det_a = self._build_detector(cfg_a)
        self.det_b = self._build_detector(cfg_b)
        self.poly_a = roi_utils.to_polygon(cfg_a.get("court", {}).get("polygon"))
        self.poly_b = roi_utils.to_polygon(cfg_b.get("court", {}).get("polygon"))

        # ONE identity manager for the whole court. homog=None: fusion feeds it
        # GLOBAL positions directly via assign(), so it needs no homography.
        self.identity = IdentityManager(reid_cfg, None, "fused", self.out_dir, self.kp_thr)

        self.minimap = Minimap(scale_px_per_m=cfg_a.get("minimap", {}).get("scale_px_per_m", 30),
                               margin_px=cfg_a.get("minimap", {}).get("margin_px", 28))

    @staticmethod
    def _build_detector(cfg: Dict[str, Any]) -> PoseDetector:
        d = cfg["detection"]
        return PoseDetector(
            model_path=cfg["model"], device=cfg.get("device", "cuda"),
            conf_threshold=d["conf_threshold"], iou_threshold=d["iou_threshold"],
            imgsz=d.get("imgsz", 640), person_class=d.get("classes", [0])[0],
            enhance=d.get("enhance", False), tiling=d.get("tiling"))

    # -- one camera's detections -> global-frame candidate dicts -------------
    def _candidates(self, frame: np.ndarray, detector: PoseDetector,
                    polygon, homog: Homography, cam: str) -> List[Dict[str, Any]]:
        """Detect, ROI-filter, project feet to GLOBAL metres. Returns a list of
        candidate dicts carrying everything fusion needs downstream."""
        raw = detector.detect(frame)
        kept, _ = roi_utils.filter_detections(raw, polygon)
        out = []
        for det in kept:
            foot_px, _src = foot_pixel(det["keypoints"], det["bbox"], self.kp_thr)
            foot_m = homog.pixel_to_meters(foot_px)
            gpos = to_global(cam, foot_m)
            x1, y1, x2, y2 = det["bbox"]
            out.append({
                "cam": cam, "gpos": gpos, "conf": float(det["conf"]),
                "bh": float(y2 - y1),                  # bbox height = closeness proxy
                "bbox": det["bbox"], "frame": frame,   # frame ref for colour crop
            })
        return out

    # -- sanity: drop projections that land well off the court ---------------
    def _sanity(self, cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reject detections whose GLOBAL foot lands well outside the 10x20 m court
        -- those are homography extrapolation garbage from a far/odd detection.
        We do NOT filter by half: each camera's far-field detections are the ONLY
        view of those players, so dropping them would delete real players."""
        m = self.bounds_margin_m
        out = []
        for c in cands:
            gx, gy = c["gpos"]
            if -m <= gx <= COURT_WIDTH_M + m and -m <= gy <= COURT_LENGTH_M + m:
                out.append(c)
        return out

    # -- cluster the overlap: same physical player seen by both cameras ------
    def _cluster(self, cands: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Greedily group candidates within dedup_m of each other (a mid-court
        player is seen by BOTH cameras -> one cluster of 2). Returns the clusters
        so the caller can pick a representative for matching AND label every
        member for drawing. The representative (used downstream) is the bigger-bbox
        detection = the camera the player is CLOSER to = the more reliable position
        -- the correct form of 'near-half ownership' (a preference in the overlap,
        not a drop of the other camera's far field)."""
        clusters: List[List[Dict[str, Any]]] = []
        used = [False] * len(cands)
        for i, ci in enumerate(cands):
            if used[i]:
                continue
            cluster = [ci]
            used[i] = True
            for j in range(i + 1, len(cands)):
                if used[j]:
                    continue
                cj = cands[j]
                d = np.hypot(ci["gpos"][0] - cj["gpos"][0], ci["gpos"][1] - cj["gpos"][1])
                if d <= self.dedup_m:
                    cluster.append(cj)
                    used[j] = True
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _rep(cluster: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Representative of a cluster = closest camera's view (biggest bbox)."""
        return max(cluster, key=lambda c: c["bh"])

    # -- combined view: side-1 | side-2 | top-down, in one frame -------------
    def _compose(self, frame_a: np.ndarray, frame_b: np.ndarray,
                 cands: List[Dict[str, Any]], minimap: np.ndarray,
                 frame_no: int, panel_h: int = 540) -> np.ndarray:
        """Draw the global P# boxes onto BOTH camera frames and lay them out next
        to the fused top-down map: [ side-1 | side-2 | top-down ]. `cands` carry a
        'pid' (set by run() from the cluster's assigned id) so the SAME player gets
        the SAME number/colour in both videos and on the map."""
        fa, fb = frame_a.copy(), frame_b.copy()
        for c in cands:
            pid = c.get("pid")
            if pid is None:
                continue
            fr = fa if c["cam"] == "A" else fb
            x1, y1, x2, y2 = [int(v) for v in c["bbox"]]
            col = color_for_id(pid)
            cv2.rectangle(fr, (x1, y1), (x2, y2), col, 4)
            cv2.putText(fr, f"P{pid}", (x1, max(0, y1 - 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, col, 4)

        def fit(img: np.ndarray) -> np.ndarray:
            h, w = img.shape[:2]
            return cv2.resize(img, (int(round(w * panel_h / h)), panel_h))

        pa, pb = fit(fa), fit(fb)
        for p, name in ((pa, "side-1"), (pb, "side-2")):
            cv2.putText(p, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        mm = fit(minimap)
        cv2.putText(mm, f"f{frame_no}", (6, panel_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return np.hstack([pa, pb, mm])

    # -- main loop -----------------------------------------------------------
    def run(self, show: bool = False, save_video: bool = True,
            max_frames: Optional[int] = None, start_frame: int = 0) -> Dict[str, Any]:
        cap_a = cv2.VideoCapture(self.cfg_a["source"])
        cap_b = cv2.VideoCapture(self.cfg_b["source"])
        if not cap_a.isOpened() or not cap_b.isOpened():
            raise RuntimeError("Could not open one of the two sources.")
        # seek each so that A[start] aligns with B[start+offset]
        cap_a.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        cap_b.set(cv2.CAP_PROP_POS_FRAMES, start_frame + self.offset)
        fps = cap_a.get(cv2.CAP_PROP_FPS) or 20.0

        # we always save the small top-down map; the combined [A|B|map] view is
        # built when the user wants to watch (show) or save it (save_video)
        mm_writer = cv2.VideoWriter(os.path.join(self.out_dir, "fusion_minimap.mp4"),
                                    cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                    (self.minimap.w, self.minimap.h))
        view_writer = None              # lazily opened once we know the view size
        want_view = show or save_video
        pos_writer = open(os.path.join(self.out_dir, "fusion_positions.jsonl"), "w")
        # playback-speed knob for the preview window (camera-A's config drives it):
        # 0/missing -> as fast as possible; >0 -> throttle to that fps. See utils/display.py
        throttle = PlaybackThrottle(self.cfg_a.get("display", {}).get("playback_fps", 0))
        if show:
            cv2.namedWindow("Fusion: side-1 | side-2 | top-down", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Fusion: side-1 | side-2 | top-down", 1600, 540)

        ids_seen = set()
        n = 0
        while True:
            loop_t0 = time.time()   # start of this frame's work (for playback throttle)
            ok_a, frame_a = cap_a.read()
            ok_b, frame_b = cap_b.read()
            if not ok_a or not ok_b:
                break

            # 1) both cameras -> global-frame candidates
            cands = (self._candidates(frame_a, self.det_a, self.poly_a, self.hom_a, "A")
                     + self._candidates(frame_b, self.det_b, self.poly_b, self.hom_b, "B"))
            # 2) sanity-drop off-court junk, then 3) cluster the overlap
            clusters = self._cluster(self._sanity(cands))
            reps = [self._rep(cl) for cl in clusters]

            # 4) traits for the ONE identity manager (global pos + per-camera colour)
            dets_pos = [list(r["gpos"]) for r in reps]
            dets_hist = [_color_hist(_bbox_crop(r["frame"], r["bbox"])) for r in reps]
            metas = [{"camera": r["cam"], "track_id": None} for r in reps]
            results = self.identity.assign(dets_pos, dets_hist, n, metas)

            # 5) propagate each cluster's assigned id to ALL its members (so both
            #    cameras' boxes get the same P#), build the map + log
            positions = []
            for cl, r, (pid, cost) in zip(clusters, reps, results):
                for c in cl:
                    c["pid"] = pid
                positions.append({"foot_m": list(r["gpos"]), "track_id": pid,
                                  "inside": True, "camera": r["cam"], "cost": cost})
                if pid is not None:
                    ids_seen.add(pid)
            assigned = [p for p in positions if p["track_id"] is not None]
            mm = self.minimap.render(assigned, color_fn=color_for_id)
            cv2.putText(mm, f"f{start_frame + n}  ids:{len(assigned)}/4",
                        (6, self.minimap.h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)
            mm_writer.write(mm)
            pos_writer.write(json.dumps({"frame": start_frame + n, "players": positions},
                                        cls=NumpyEncoder) + "\n")

            if want_view:
                all_cands = [c for cl in clusters for c in cl]
                view = self._compose(frame_a, frame_b, all_cands, mm, start_frame + n)
                if save_video:
                    if view_writer is None:
                        view_writer = cv2.VideoWriter(
                            os.path.join(self.out_dir, "fusion_view.mp4"),
                            cv2.VideoWriter_fourcc(*"mp4v"), fps,
                            (view.shape[1], view.shape[0]))
                    view_writer.write(view)
                if show:
                    cv2.imshow("Fusion: side-1 | side-2 | top-down", view)
                    # pause only the leftover of the frame budget (see utils/display.py)
                    work_ms = (time.time() - loop_t0) * 1000.0
                    if throttle.wait(work_ms) == ord("q"):
                        break

            n += 1
            if max_frames is not None and n >= max_frames:
                break

        cap_a.release()
        cap_b.release()
        mm_writer.release()
        if view_writer is not None:
            view_writer.release()
        pos_writer.close()
        if show:
            cv2.destroyAllWindows()
        self.identity.save()

        summary = {
            "frames_processed": n,
            "offset_frames": self.offset,
            "unique_player_ids_used": sorted(ids_seen),
            "n_unique_ids": len(ids_seen),
        }
        view_msg = f" + combined view -> {self.out_dir}/fusion_view.mp4" if view_writer else ""
        print(f"[fusion] processed {n} synced frames. ids used: {sorted(ids_seen)} "
              f"(<=4 by construction). minimap -> {self.out_dir}/fusion_minimap.mp4{view_msg}")
        return summary
