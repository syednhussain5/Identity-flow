"""
detector.py
YOLOv8-based face detection wrapper.
Returns bounding boxes and confidence scores per frame.
"""

import logging
from typing import List, Tuple

import numpy as np

logger = logging.getLogger("face_tracker")


class FaceDetector:
    """
    Wraps YOLOv8 (ultralytics) for face detection.

    Attributes:
        model_path:    Path to the .pt weights file.
        conf:          Minimum confidence threshold [0-1].
        input_size:    Inference resolution (square).
        frame_skip:    Detect every Nth frame; interpolate in between.
    """

    def __init__(self, model_path: str, conf: float = 0.5, input_size: int = 640, frame_skip: int = 3):
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.conf = conf
        self.input_size = input_size
        self.frame_skip = frame_skip
        self._frame_count = 0
        self._last_results: List[Tuple] = []

        logger.info(
            "FaceDetector ready | model=%s conf=%.2f frame_skip=%d",
            model_path, conf, frame_skip,
        )

    def detect(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Run detection on the given frame (respecting frame_skip).
        Applies NMS to eliminate overlapping detections.

        Returns:
            List of (x1, y1, x2, y2, confidence) tuples in pixel coords.
        """
        self._frame_count += 1

        if self._frame_count % self.frame_skip != 0:
            return self._last_results

        results = self.model.predict(
            frame,
            conf=self.conf,
            imgsz=self.input_size,
            verbose=False,
        )

        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                detections.append((int(x1), int(y1), int(x2), int(y2), conf))

        # Apply NMS to remove overlapping detections
        detections = self._apply_nms(detections, nms_threshold=0.4)

        self._last_results = detections
        logger.debug("Frame %d | detected %d face(s) after NMS", self._frame_count, len(detections))
        return detections

    @staticmethod
    def _apply_nms(detections: List[Tuple[int, int, int, int, float]], nms_threshold: float = 0.4) -> List[Tuple[int, int, int, int, float]]:
        """
        Apply Non-Maximum Suppression to remove overlapping detections.
        Keeps detections with highest confidence.

        Args:
            detections: List of (x1, y1, x2, y2, confidence) tuples
            nms_threshold: IoU threshold for suppression (0-1, lower = more suppression)

        Returns:
            List of detections after NMS
        """
        if not detections:
            return detections

        # Sort by confidence (descending)
        detections = sorted(detections, key=lambda x: x[4], reverse=True)
        keep = []

        for i, det_i in enumerate(detections):
            keep_det = True
            for det_j in keep:
                iou = FaceDetector._compute_iou(det_i[:4], det_j[:4])
                if iou > nms_threshold:
                    keep_det = False
                    break
            if keep_det:
                keep.append(det_i)

        return keep

    @staticmethod
    def _compute_iou(box1: Tuple, box2: Tuple) -> float:
        """Compute Intersection-over-Union between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2

        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)

        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0

        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area
