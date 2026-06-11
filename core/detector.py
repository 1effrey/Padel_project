"""core/detector.py
YOLOv11-pose detector wrapper.

WHAT THIS DOES
  Loads the YOLOv11-pose model ONCE and, for each frame, returns a plain
  Python list of detection dicts:
      [{"bbox", "conf", "keypoints"}, ...]
    - bbox      : np.ndarray [x1, y1, x2, y2]  (pixels)
    - conf      : float                        (person/detection confidence)
    - keypoints : np.ndarray shape (17, 3)     COCO-17 -> (x, y, kp_conf)

WHY A WRAPPER (and why plain dicts, not ultralytics objects)
  The rest of the pipeline must never import ultralytics directly. If we ever
  swap the model (e.g. Phase 6 TensorRT), only THIS file changes. Returning
  simple dicts also means:
    * detector.py has zero dependency on supervision / ByteTrack, and
    * the ROI filter and tests can work on the output without a GPU.

ACCURACY HELPERS (optional, off by default -> behaviour unchanged unless enabled)
  - enhance (CLAHE):  lifts dark, far players out of shadow on night footage
                      before YOLO sees the frame.
  - tiling:           slices the frame into a grid, runs YOLO on each tile at
                      full tile resolution, and maps boxes AND keypoints back
                      to full-frame coords. Far/small players occupy many more
                      pixels inside a tile, so they detect at HIGHER confidence.
                      Overlapping detections are merged with NMS.
                      (We do this manually -- the SAHI library only returns
                      boxes, which would lose the pose keypoints.)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLO


class PoseDetector:
    """Thin wrapper around a single YOLOv11-pose model."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        conf_threshold: float = 0.3,
        iou_threshold: float = 0.5,
        imgsz: int = 640,
        person_class: int = 0,
        enhance: bool = False,
        tiling: Optional[Dict[str, Any]] = None,
    ) -> None:
        # The model file is auto-downloaded by ultralytics on first use.
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.person_class = person_class  # COCO class 0 == "person"

        self.enhance = enhance
        # tiling = {"enabled": bool, "rows": int, "cols": int,
        #           "overlap": float (0-1), "include_full": bool}
        self.tiling = tiling or {}
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------ public
    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run detection on a BGR frame and return detection dicts."""
        if self.enhance:
            frame = self._enhance(frame)

        if not self.tiling.get("enabled", False):
            return self._predict(frame)  # original single-pass behaviour

        # --- tiled path: full-frame pass (optional) + one pass per tile -------
        dets: List[Dict[str, Any]] = []
        if self.tiling.get("include_full", True):
            dets += self._predict(frame)

        rows = int(self.tiling.get("rows", 2))
        cols = int(self.tiling.get("cols", 2))
        overlap = float(self.tiling.get("overlap", 0.2))
        h, w = frame.shape[:2]
        tile_h, tile_w = h / rows, w / cols
        pad_h, pad_w = tile_h * overlap, tile_w * overlap

        for r in range(rows):
            for c in range(cols):
                y1 = int(max(0, r * tile_h - pad_h))
                y2 = int(min(h, (r + 1) * tile_h + pad_h))
                x1 = int(max(0, c * tile_w - pad_w))
                x2 = int(min(w, (c + 1) * tile_w + pad_w))
                tile = frame[y1:y2, x1:x2]
                if tile.size:
                    dets += self._predict(tile, off_x=x1, off_y=y1)

        return self._nms(dets, iou_thr=0.6)

    # ----------------------------------------------------------------- private
    def _enhance(self, frame: np.ndarray) -> np.ndarray:
        """CLAHE on the L channel of LAB -> brighter shadows, same colours."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    def _predict(
        self, image: np.ndarray, off_x: int = 0, off_y: int = 0
    ) -> List[Dict[str, Any]]:
        """One YOLO pass on `image`; boxes/keypoints shifted by (off_x, off_y)."""
        results = self.model.predict(
            image,
            device=self.device,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            classes=[self.person_class],
            verbose=False,
        )[0]

        detections: List[Dict[str, Any]] = []
        if results.boxes is None or len(results.boxes) == 0:
            return detections

        boxes = results.boxes.xyxy.cpu().numpy()          # (N, 4)
        confs = results.boxes.conf.cpu().numpy()          # (N,)
        if results.keypoints is not None:
            kpts = results.keypoints.data.cpu().numpy()   # (N, 17, 3)
        else:
            kpts = np.zeros((len(boxes), 17, 3), dtype=float)

        for i in range(len(boxes)):
            box = boxes[i].astype(float).copy()
            box[[0, 2]] += off_x
            box[[1, 3]] += off_y
            kp = kpts[i].astype(float).copy()
            kp[:, 0] += off_x        # shift keypoint x
            kp[:, 1] += off_y        # shift keypoint y
            detections.append({"bbox": box, "conf": float(confs[i]), "keypoints": kp})
        return detections

    @staticmethod
    def _nms(dets: List[Dict[str, Any]], iou_thr: float = 0.6) -> List[Dict[str, Any]]:
        """Greedy NMS to merge duplicate detections from overlapping tiles."""
        if len(dets) <= 1:
            return dets
        boxes = np.array([d["bbox"] for d in dets])
        scores = np.array([d["conf"] for d in dets])
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size:
            i = order[0]
            keep.append(int(i))
            if order.size == 1:
                break
            rest = order[1:]
            ious = PoseDetector._iou(boxes[i], boxes[rest])
            order = rest[ious < iou_thr]
        return [dets[i] for i in keep]

    @staticmethod
    def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """IoU of one box (4,) against many boxes (M, 4)."""
        xx1 = np.maximum(box[0], boxes[:, 0])
        yy1 = np.maximum(box[1], boxes[:, 1])
        xx2 = np.minimum(box[2], boxes[:, 2])
        yy2 = np.minimum(box[3], boxes[:, 3])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        area = (box[2] - box[0]) * (box[3] - box[1])
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return inter / (area + areas - inter + 1e-9)
