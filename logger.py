"""
logger.py
Structured logging to both console and a rotating log file.
Handles saving cropped face images to dated folders.
"""

import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2


def setup_logger(log_file: str, level: str = "INFO") -> logging.Logger:
    """Configure and return the root application logger."""
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("face_tracker")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def save_face_image(
    frame, bbox: tuple, face_id: str, event_type: str, base_dir: str, fmt: str = "jpg"
) -> str:
    """
    Crop the face from frame and save to a dated subfolder.

    Args:
        frame:      Full OpenCV BGR frame.
        bbox:       (x1, y1, x2, y2) in pixel coordinates.
        face_id:    Unique face identifier string.
        event_type: "entry" or "exit".
        base_dir:   Root image store directory.
        fmt:        Image format extension (jpg/png).

    Returns:
        Absolute path to the saved image, or empty string on failure.
    """
    try:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]

        # Clamp to frame boundaries
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return ""

        today = datetime.now().strftime("%Y-%m-%d")
        save_dir = Path(base_dir) / today
        save_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%H%M%S_%f")[:13]
        filename = f"{event_type}_{face_id}_{ts}.{fmt}"
        out_path = save_dir / filename

        cv2.imwrite(str(out_path), crop)
        return str(out_path)

    except Exception as exc:
        logging.getLogger("face_tracker").warning(
            "Failed to save face image for %s: %s", face_id, exc
        )
        return ""
