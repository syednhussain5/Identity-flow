"""
registry.py
In-memory face registry backed by the SQLite database.
Handles auto-registration of new faces and re-identification of returning ones.
"""

import logging
import uuid
from typing import Dict, Optional, Tuple

import numpy as np

from database import Database
from embedder import FaceEmbedder

logger = logging.getLogger("face_tracker")


class FaceRegistry:
    """
    Maps embeddings to unique face IDs.

    On startup it loads all known embeddings from the DB so the system
    can re-identify faces even across application restarts.

    Thread safety: all mutations are GIL-protected (CPython dict operations),
    which is sufficient for single-writer / multiple-reader use in this pipeline.
    """

    def __init__(self, db: Database, similarity_threshold: float = 0.45):
        self.db = db
        self.threshold = similarity_threshold
        # face_id → embedding (np.ndarray)
        self._registry: Dict[str, np.ndarray] = {}
        self._load_from_db()

        logger.info(
            "FaceRegistry ready | %d known face(s) loaded | threshold=%.2f",
            len(self._registry),
            self.threshold,
        )

    # ── Startup ───────────────────────────────────────────────────────────────

    def _load_from_db(self):
        """Deserialise all stored embeddings from the database."""
        for row in self.db.get_all_faces():
            if row["embedding"]:
                try:
                    emb = FaceEmbedder.bytes_to_embedding(row["embedding"])
                    self._registry[row["id"]] = emb
                except Exception as exc:
                    logger.warning("Could not load embedding for %s: %s", row["id"], exc)

    # ── Core match / register ─────────────────────────────────────────────────

    def identify(self, embedding: np.ndarray) -> Tuple[str, bool]:
        """
        Match the embedding against all known faces.

        Returns:
            (face_id, is_new):
                face_id  — matched or newly assigned identifier.
                is_new   — True if this is a brand-new registration.
        """
        best_id, best_score = self._find_best_match(embedding)

        if best_score >= self.threshold:
            # Returning visitor
            self.db.update_face_last_seen(best_id)
            logger.info("Recognised face %s (sim=%.3f)", best_id, best_score)
            return best_id, False

        # New visitor
        face_id = self._generate_id()
        self._register(face_id, embedding)
        logger.info("New face registered as %s", face_id)
        return face_id, True

    def _find_best_match(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        best_id = None
        best_score = -1.0
        for fid, stored_emb in self._registry.items():
            score = FaceEmbedder.cosine_similarity(embedding, stored_emb)
            if score > best_score:
                best_score = score
                best_id = fid
        return best_id, best_score

    def _register(self, face_id: str, embedding: np.ndarray):
        self._registry[face_id] = embedding
        self.db.register_face(face_id, FaceEmbedder.embedding_to_bytes(embedding))
        self.db.increment_unique_visitor()

    @staticmethod
    def _generate_id() -> str:
        return "face_" + uuid.uuid4().hex[:8]

    # ── Queries ───────────────────────────────────────────────────────────────

    def total_unique(self) -> int:
        return self.db.get_unique_visitor_count()

    def known_ids(self):
        return list(self._registry.keys())
