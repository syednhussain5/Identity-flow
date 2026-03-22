"""
embedder.py
Generates 512-d ArcFace embeddings using InsightFace.
Used for both registration and recognition of faces.
"""

import logging
import pickle
from typing import Optional

import numpy as np

logger = logging.getLogger("face_tracker")


class FaceEmbedder:
    """
    Wraps InsightFace (buffalo_l / ArcFace) for embedding extraction.

    The embedder accepts a cropped face region (numpy array, BGR)
    and returns a normalised 512-dimensional embedding vector.
    """

    def __init__(self, model_name: str = "buffalo_l"):
        import insightface
        from insightface.app import FaceAnalysis

        self.app = FaceAnalysis(
            name=model_name,
            allowed_modules=["detection", "recognition"],
        )
        self.app.prepare(ctx_id=0, det_size=(160, 160))
        logger.info("FaceEmbedder ready | model=%s", model_name)

    def get_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract a 512-d normalised embedding from a face crop.

        Args:
            face_crop: BGR numpy array of the cropped face region.

        Returns:
            Normalised embedding array (512,) or None on failure.
        """
        try:
            faces = self.app.get(face_crop)
            if not faces:
                return None

            # Use the highest-confidence detected face
            face = max(faces, key=lambda f: f.det_score)
            emb = face.normed_embedding  # already L2-normalised by InsightFace
            return emb.astype(np.float32)

        except Exception as exc:
            logger.warning("Embedding extraction failed: %s", exc)
            return None

    @staticmethod
    def embedding_to_bytes(embedding: np.ndarray) -> bytes:
        """Serialise embedding to bytes for DB storage."""
        return pickle.dumps(embedding)

    @staticmethod
    def bytes_to_embedding(data: bytes) -> np.ndarray:
        """Deserialise embedding from DB bytes."""
        return pickle.loads(data)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Return cosine similarity in [-1, 1]; higher = more similar."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
