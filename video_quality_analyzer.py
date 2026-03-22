"""
video_quality_analyzer.py
Analyzes video quality metrics and recommends optimal detection parameters.
Adjusts confidence thresholds, frame skip rates, and IoU thresholds dynamically.
"""

import logging
import cv2
import numpy as np
from typing import Dict, Tuple

logger = logging.getLogger("face_tracker")


class VideoQualityAnalyzer:
    """
    Analyzes video characteristics and recommends detection parameters.
    
    Metrics analyzed:
    - Resolution (width x height)
    - Frame rate (FPS)
    - Brightness/Contrast (Laplacian variance)
    - Motion blur detection
    """

    def __init__(self):
        self.quality_metrics = {}

    def analyze_video(self, cap: cv2.VideoCapture, sample_frames: int = 10) -> Dict:
        """
        Analyze video characteristics by sampling frames.

        Args:
            cap: cv2.VideoCapture object
            sample_frames: Number of frames to sample for analysis

        Returns:
            Dictionary with quality metrics and recommended parameters
        """
        if not cap.isOpened():
            return self._get_default_params()

        # Get basic video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Sample frames and analyze quality
        sharpness_scores = []
        brightness_scores = []
        contrast_scores = []

        current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
        sample_interval = max(1, total_frames // sample_frames)

        for i in range(sample_frames):
            frame_num = i * sample_interval
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = cap.read()

            if not ret:
                break

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Sharpness (Laplacian variance - higher = sharper)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_scores.append(laplacian_var)

            # Brightness (mean pixel value)
            brightness = np.mean(gray)
            brightness_scores.append(brightness)

            # Contrast (standard deviation)
            contrast = np.std(gray)
            contrast_scores.append(contrast)

        # Reset video position
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # Calculate averages
        avg_sharpness = np.mean(sharpness_scores) if sharpness_scores else 100
        avg_brightness = np.mean(brightness_scores) if brightness_scores else 128
        avg_contrast = np.mean(contrast_scores) if contrast_scores else 50

        self.quality_metrics = {
            "resolution": (width, height),
            "fps": fps,
            "total_frames": total_frames,
            "avg_sharpness": avg_sharpness,
            "avg_brightness": avg_brightness,
            "avg_contrast": avg_contrast,
        }

        # Get recommended parameters
        params = self._calculate_parameters(width, height, fps, avg_sharpness, avg_brightness, avg_contrast)
        
        logger.info(
            "Video Quality Analyzed | Resolution: %dx%d | FPS: %.1f | "
            "Sharpness: %.1f | Brightness: %.1f | Contrast: %.1f",
            width, height, fps, avg_sharpness, avg_brightness, avg_contrast
        )

        return params

    def _calculate_parameters(
        self,
        width: int,
        height: int,
        fps: float,
        sharpness: float,
        brightness: float,
        contrast: float,
    ) -> Dict:
        """
        Calculate optimal parameters based on video quality metrics.

        Returns:
            Dict with recommended parameters
        """
        params = {
            "confidence_threshold": 0.5,
            "frame_skip": 3,
            "iou_threshold": 0.4,
            "input_size": 640,
            "similarity_threshold": 0.45,
            "quality_grade": "medium",
        }

        # Resolution-based adjustments
        resolution_grade = self._grade_resolution(width, height)
        
        # Sharpness-based adjustments (Laplacian variance)
        # Low < 100, Medium 100-500, High > 500
        if sharpness < 100:
            sharpness_grade = "low"
            params["confidence_threshold"] = 0.4  # Lower threshold for blurry content
            params["input_size"] = 480  # Smaller input for better processing of low-res
        elif sharpness < 500:
            sharpness_grade = "medium"
            params["confidence_threshold"] = 0.5
            params["input_size"] = 640
        else:
            sharpness_grade = "high"
            params["confidence_threshold"] = 0.6  # Higher threshold for clear content
            params["input_size"] = 768  # Larger input for better accuracy

        # Brightness adjustment (ideal: 80-180)
        if brightness < 60 or brightness > 200:
            brightness_grade = "poor"
            params["confidence_threshold"] -= 0.15  # Lower confidence for bad lighting
            params["similarity_threshold"] = 0.35  # More lenient face matching
        elif brightness < 100 or brightness > 180:
            brightness_grade = "acceptable"
        else:
            brightness_grade = "good"
            params["confidence_threshold"] += 0.05  # Higher confidence for good lighting

        # Contrast adjustment (ideal: 30-80)
        if contrast < 15:
            contrast_grade = "low"
            params["confidence_threshold"] -= 0.1
            params["similarity_threshold"] = 0.40
        elif contrast > 100:
            contrast_grade = "high"
            params["confidence_threshold"] += 0.05
        else:
            contrast_grade = "good"

        # FPS-based frame skip adjustments
        if fps < 15:
            params["frame_skip"] = 1  # Don't skip frames for slow videos
        elif fps < 25:
            params["frame_skip"] = 2
        elif fps < 30:
            params["frame_skip"] = 3
        else:
            params["frame_skip"] = 5  # Skip more for high FPS

        # Resolution-based IoU threshold
        if resolution_grade == "low":
            params["iou_threshold"] = 0.35
            params["input_size"] = 480
        elif resolution_grade == "high":
            params["iou_threshold"] = 0.45
            params["input_size"] = 768

        # Overall quality grade
        quality_scores = {
            "sharpness": self._score_quality(sharpness, 100, 500),
            "brightness": self._score_quality(brightness, 100, 180, is_centered=True),
            "contrast": self._score_quality(contrast, 30, 80, is_centered=True),
            "resolution": self._score_resolution(width, height),
        }

        avg_quality = np.mean(list(quality_scores.values()))
        if avg_quality < 0.4:
            params["quality_grade"] = "low"
        elif avg_quality < 0.7:
            params["quality_grade"] = "medium"
        else:
            params["quality_grade"] = "high"

        # Ensure bounds
        params["confidence_threshold"] = max(0.2, min(0.8, params["confidence_threshold"]))
        params["similarity_threshold"] = max(0.3, min(0.6, params["similarity_threshold"]))
        params["frame_skip"] = max(1, min(10, params["frame_skip"]))

        logger.info(
            "Recommended Parameters | Grade: %s | Confidence: %.2f | Frame Skip: %d | "
            "IoU: %.2f | Input Size: %d | Similarity: %.2f",
            params["quality_grade"],
            params["confidence_threshold"],
            params["frame_skip"],
            params["iou_threshold"],
            params["input_size"],
            params["similarity_threshold"],
        )

        return params

    @staticmethod
    def _grade_resolution(width: int, height: int) -> str:
        """Grade video resolution."""
        area = width * height
        if area < 480 * 360:  # < 720p
            return "low"
        elif area < 1920 * 1080:  # < 1080p
            return "medium"
        else:
            return "high"

    @staticmethod
    def _score_quality(value: float, low_threshold: float, high_threshold: float, is_centered: bool = False) -> float:
        """
        Score a metric on scale 0-1.
        
        Args:
            value: Actual value
            low_threshold: Lower acceptable threshold
            high_threshold: Upper acceptable threshold
            is_centered: If True, both thresholds are acceptable (centered range)
        
        Returns:
            Score 0-1
        """
        if is_centered:
            # For centered values (brightness, contrast)
            center = (low_threshold + high_threshold) / 2
            ideal_range = (high_threshold - low_threshold) / 2
            distance = abs(value - center)
            return max(0, 1 - (distance / (ideal_range * 2)))
        else:
            # For increasing values (sharpness)
            if value < low_threshold:
                return value / low_threshold * 0.5
            elif value < high_threshold:
                return 0.5 + (value - low_threshold) / (high_threshold - low_threshold) * 0.5
            else:
                return 1.0

    @staticmethod
    def _score_resolution(width: int, height: int) -> float:
        """Score resolution quality."""
        area = width * height
        if area < 480 * 360:
            return 0.3
        elif area < 1280 * 720:
            return 0.6
        elif area < 1920 * 1080:
            return 0.85
        else:
            return 1.0

    @staticmethod
    def _get_default_params() -> Dict:
        """Return default parameters when analysis fails."""
        return {
            "confidence_threshold": 0.5,
            "frame_skip": 3,
            "iou_threshold": 0.4,
            "input_size": 640,
            "similarity_threshold": 0.45,
            "quality_grade": "medium",
        }
