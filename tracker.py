"""
tracker.py
Lightweight multi-object tracker using IoU-based assignment (DeepSort-lite).
Falls back to simple centroid tracking when ByteTrack is unavailable.

Track lifecycle:
  - Each detected bounding box is assigned a track_id.
  - Tracks persist for `max_disappeared` frames without a detection match.
  - When a track is confirmed, it is passed to the registry for face recognition.
"""

import logging
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("face_tracker")


def _iou(boxA: Tuple, boxB: Tuple) -> float:
    """Compute Intersection-over-Union for two (x1,y1,x2,y2) boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0

    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


class Track:
    """Single object track state."""

    def __init__(self, track_id: int, bbox: Tuple, face_id: Optional[str] = None):
        self.track_id = track_id
        self.bbox = bbox
        self.face_id = face_id
        self.disappeared = 0
        self.confirmed = False   # set after face_id is assigned


class IoUTracker:
    """
    Frame-to-frame IoU tracker.
    Creates, updates, and removes tracks based on bounding-box overlap.
    """

    def __init__(self, max_disappeared: int = 30, iou_threshold: float = 0.4):
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self._next_id = 0
        self.tracks: Dict[int, Track] = OrderedDict()

    def update(
        self, detections: List[Tuple[int, int, int, int, float]]
    ) -> List[Track]:
        """
        Match new detections to existing tracks.

        Args:
            detections: List of (x1, y1, x2, y2, conf).

        Returns:
            List of active Track objects after update.
        """
        if not detections:
            for track in self.tracks.values():
                track.disappeared += 1
            self._prune()
            return list(self.tracks.values())

        det_boxes = [d[:4] for d in detections]
        track_ids = list(self.tracks.keys())

        if not track_ids:
            # Initialise fresh tracks for all detections
            for box in det_boxes:
                self._create_track(box)
            return list(self.tracks.values())

        # Build IoU matrix: rows=tracks, cols=detections
        iou_matrix = np.zeros((len(track_ids), len(det_boxes)))
        for ti, tid in enumerate(track_ids):
            for di, dbox in enumerate(det_boxes):
                iou_matrix[ti, di] = _iou(self.tracks[tid].bbox, dbox)

        # Greedy max-IoU assignment
        matched_tracks = set()
        matched_dets = set()

        while True:
            if iou_matrix.size == 0:
                break
            max_val = iou_matrix.max()
            if max_val < self.iou_threshold:
                break
            ti, di = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            track_id = track_ids[ti]
            self.tracks[track_id].bbox = det_boxes[di]
            self.tracks[track_id].disappeared = 0
            matched_tracks.add(ti)
            matched_dets.add(di)
            iou_matrix[ti, :] = -1
            iou_matrix[:, di] = -1

        # Unmatched tracks → increment disappeared
        for ti, tid in enumerate(track_ids):
            if ti not in matched_tracks:
                self.tracks[tid].disappeared += 1

        # Unmatched detections → new tracks
        for di, box in enumerate(det_boxes):
            if di not in matched_dets:
                self._create_track(box)

        self._prune()
        return list(self.tracks.values())

    def _create_track(self, bbox: Tuple) -> Track:
        track = Track(track_id=self._next_id, bbox=bbox)
        self.tracks[self._next_id] = track
        logger.debug("Track %d created", self._next_id)
        self._next_id += 1
        return track

    def _prune(self):
        """Remove tracks that have been invisible too long."""
        to_delete = [
            tid for tid, t in self.tracks.items()
            if t.disappeared > self.max_disappeared
        ]
        for tid in to_delete:
            logger.debug("Track %d pruned (disappeared=%d)", tid, self.tracks[tid].disappeared)
            del self.tracks[tid]

    def assign_face_id(self, track_id: int, face_id: str):
        if track_id in self.tracks:
            self.tracks[track_id].face_id = face_id
            self.tracks[track_id].confirmed = True

    def get_track(self, track_id: int) -> Optional[Track]:
        return self.tracks.get(track_id)
