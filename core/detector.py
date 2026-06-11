"""core/detector.py
YOLOv11-pose detector wrapper.

WHAT THIS DOES
  Loads the YOLOv11n-pose model ONCE and, for each frame, returns a plain
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
"""
from __future__ import annotations

from typing import Any, Dict, List

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
    ) -> None:
        # The model file (yolo11n-pose.pt) is auto-downloaded by ultralytics
        # on first use if it isn't present locally.
        self.model = YOLO(model_path)
        self.device = device
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.person_class = person_class  # COCO class 0 == "person"

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run one forward pass on a BGR frame and return detection dicts."""
        # verbose=False keeps the console clean; classes=[0] keeps only people.
        results = self.model.predict(
            frame,
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
            # Should not happen with a -pose model, but stay safe.
            kpts = np.zeros((len(boxes), 17, 3), dtype=float)

        for i in range(len(boxes)):
            detections.append(
                {
                    "bbox": boxes[i].astype(float),
                    "conf": float(confs[i]),
                    "keypoints": kpts[i].astype(float),
                }
            )
        return detections
