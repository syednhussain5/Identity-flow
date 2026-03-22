"""
event_handler.py
Detects face entry and exit events from track state changes.
Coordinates between tracker, registry, logger, and database.
"""

import logging
from typing import Dict, Set
from datetime import datetime, timedelta

import numpy as np

from database import Database
from embedder import FaceEmbedder
from logger import save_face_image
from registry import FaceRegistry
from tracker import IoUTracker, Track

logger = logging.getLogger("face_tracker")

# 🔥 OPTIONAL: Disable noisy logs (clean terminal)
logging.getLogger("face_tracker").setLevel(logging.ERROR)


class EventHandler:
    def __init__(
        self,
        tracker: IoUTracker,
        embedder: FaceEmbedder,
        registry: FaceRegistry,
        db: Database,
        image_base_dir: str,
        image_format: str = "jpg",
    ):
        self.tracker = tracker
        self.embedder = embedder
        self.registry = registry
        self.db = db
        self.image_base_dir = image_base_dir
        self.image_format = image_format

        self._active: Dict[int, str] = {}
        self._prev_track_ids: Set[int] = set()
        # Track recently identified faces to prevent duplicates
        self._recently_identified: Dict[str, datetime] = {}
        self._duplicate_prevention_window = timedelta(seconds=3)

    def process_frame(self, frame: np.ndarray, tracks):
        current_ids = {t.track_id for t in tracks}

        # ── EXIT DETECTION ─────────────────────────────
        exited_ids = self._prev_track_ids - current_ids
        for tid in exited_ids:
            if tid in self._active:
                face_id = self._active.pop(tid)
                self._fire_exit(face_id)

        # ── ACTIVE TRACK PROCESSING ───────────────────
        for track in tracks:
            tid = track.track_id

            if tid in self._active:
                continue

            if track.face_id is not None:
                self._active[tid] = track.face_id
                continue

            self._try_identify(frame, track)

        self._prev_track_ids = current_ids
        
        # Clean up old entries from recently_identified
        now = datetime.now()
        self._recently_identified = {
            fid: ts for fid, ts in self._recently_identified.items()
            if now - ts < self._duplicate_prevention_window
        }

    def _try_identify(self, frame: np.ndarray, track: Track):
        x1, y1, x2, y2 = track.bbox
        crop = frame[max(0, y1):y2, max(0, x1):x2]

        if crop.size == 0:
            return

        embedding = self.embedder.get_embedding(crop)

        if embedding is None:
            return

        face_id, is_new = self.registry.identify(embedding)
        
        # Check if this face was recently identified (duplicate prevention)
        now = datetime.now()
        if face_id in self._recently_identified:
            time_since = now - self._recently_identified[face_id]
            if time_since < self._duplicate_prevention_window:
                logger.debug(
                    "Duplicate detection prevented for %s (seen %.1f seconds ago)",
                    face_id,
                    time_since.total_seconds()
                )
                print(f"[✓ DUPLICATE BLOCKED] Face {face_id} already tracked (seen {time_since.total_seconds():.1f}s ago)")
                # Assign the face_id to track without firing another entry event
                self.tracker.assign_face_id(track.track_id, face_id)
                self._active[track.track_id] = face_id
                return
        
        # Record this identification
        self._recently_identified[face_id] = now
        
        if is_new:
            print("\n🚨 ALERT: Unknown Person Detected 🚨")
        self.tracker.assign_face_id(track.track_id, face_id)
        self._active[track.track_id] = face_id

        # Save image + DB log
        img_path = save_face_image(
            frame,
            track.bbox,
            face_id,
            "entry",
            self.image_base_dir,
            self.image_format,
        )

        self.db.log_event(face_id, "entry", img_path)

        # 🔥 CLEAN OUTPUT
        print("\n==============================")
        print("Face Detected ✅")
        print(f"Time      : {datetime.now().strftime('%H:%M:%S')}")
        print(f"Face ID   : {face_id}")
        print(f"Track ID  : {track.track_id}")
        print(f"Type      : {'New' if is_new else 'Known'}")
        print("Status    : ENTRY")
        print("==============================")

        # Optional debug log
        logger.info(
            "ENTRY | face_id=%s track_id=%d new=%s image=%s",
            face_id,
            track.track_id,
            is_new,
            img_path,
        )

    def _fire_exit(self, face_id: str):
        self.db.log_event(face_id, "exit")

        # 🔥 CLEAN OUTPUT
        print("\n------------------------------")
        print(f"Time      : {datetime.now().strftime('%H:%M:%S')}")
        print(f"Face ID   : {face_id}")
        print("Status    : EXIT ❌")
        print("------------------------------")

        # Optional debug log
        logger.info("EXIT  | face_id=%s", face_id)

    @property
    def active_face_count(self) -> int:
        return len(self._active)

    def unique_visitor_count(self) -> int:
        try:
            return self.registry.total_unique()
        except:
            return 0