"""
async_processor.py
Optional optimization module for batch embedding processing.
Improves performance for multi-face scenarios using ThreadPoolExecutor.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import numpy as np

logger = logging.getLogger("face_tracker")


class AsyncEmbeddingProcessor:
    """
    Process multiple face crops in parallel to generate embeddings.
    Significantly speeds up recognition when multiple faces are detected.
    """

    def __init__(self, embedder, max_workers: int = 4):
        """
        Args:
            embedder: FaceEmbedder instance
            max_workers: Number of parallel embedding threads (2-8 recommended)
        """
        self.embedder = embedder
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        logger.info("AsyncEmbeddingProcessor initialized | workers=%d", max_workers)

    def batch_embed(self, crops: List[np.ndarray]) -> List[Tuple[int, np.ndarray]]:
        """
        Process multiple crops in parallel.

        Args:
            crops: List of face crop images

        Returns:
            List of (crop_index, embedding) tuples for successful crops
        """
        if not crops:
            return []

        embeddings = []
        futures = {}

        # Submit all crops to thread pool
        for idx, crop in enumerate(crops):
            future = self.executor.submit(self.embedder.get_embedding, crop)
            futures[future] = idx

        # Collect results as they complete
        for future in as_completed(futures):
            idx = futures[future]
            try:
                embedding = future.result()
                if embedding is not None:
                    embeddings.append((idx, embedding))
            except Exception as exc:
                logger.warning("Embedding failed for crop %d: %s", idx, exc)

        return embeddings

    def shutdown(self):
        """Cleanly shutdown thread pool."""
        self.executor.shutdown(wait=True)
        logger.info("AsyncEmbeddingProcessor shutdown complete")


class PerformanceMonitor:
    """
    Track key performance metrics for debugging and optimization.
    """

    def __init__(self):
        self.detection_times = []
        self.embedding_times = []
        self.tracking_times = []
        self.recognition_times = []

    def add_detection_time(self, ms: float):
        self.detection_times.append(ms)

    def add_embedding_time(self, ms: float):
        self.embedding_times.append(ms)

    def add_tracking_time(self, ms: float):
        self.tracking_times.append(ms)

    def add_recognition_time(self, ms: float):
        self.recognition_times.append(ms)

    def get_stats(self) -> dict:
        """Return average timings for last N operations."""
        def avg(times, n=100):
            if not times:
                return 0
            return sum(times[-n:]) / len(times[-n:])

        return {
            "detection_ms": round(avg(self.detection_times), 2),
            "embedding_ms": round(avg(self.embedding_times), 2),
            "tracking_ms": round(avg(self.tracking_times), 2),
            "recognition_ms": round(avg(self.recognition_times), 2),
            "total_frames_processed": len(self.detection_times),
        }

    def log_stats(self):
        stats = self.get_stats()
        logger.info(
            "Performance Stats | Detection: %.1fms | Embedding: %.1fms | "
            "Tracking: %.1fms | Recognition: %.1fms | Frames: %d",
            stats["detection_ms"],
            stats["embedding_ms"],
            stats["tracking_ms"],
            stats["recognition_ms"],
            stats["total_frames_processed"],
        )
