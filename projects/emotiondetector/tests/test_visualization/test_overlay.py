"""Tests for face overlay visualization.

This module tests the FaceOverlay class for drawing landmarks and emotions on frames.
"""

import numpy as np
import pytest

from asdrp.face.base import BoundingBox, FaceLandmarks
from asdrp.visualization.overlay import FaceOverlay


class TestFaceOverlay:
    """Test suite for FaceOverlay class."""

    def test_initialization_default(self) -> None:
        """Test FaceOverlay initialization with defaults."""
        overlay = FaceOverlay()

        assert overlay.draw_landmarks is True
        assert overlay.draw_bbox is True
        assert overlay.line_thickness > 0

    def test_initialization_custom(self) -> None:
        """Test FaceOverlay initialization with custom parameters."""
        overlay = FaceOverlay(
            draw_landmarks=False,
            draw_bbox=True,
            landmark_color=(255, 0, 0),
            line_thickness=3,
        )

        assert overlay.draw_landmarks is False
        assert overlay.draw_bbox is True
        assert overlay.landmark_color == (255, 0, 0)
        assert overlay.line_thickness == 3

    def test_draw_landmarks_basic(
        self, sample_face_image: np.ndarray, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test drawing landmarks on image."""
        overlay = FaceOverlay()

        result = overlay.draw_landmarks_on_image(
            sample_face_image, [sample_face_landmarks]
        )

        assert result.shape == sample_face_image.shape
        assert result.dtype == sample_face_image.dtype
        # Image should be modified
        assert not np.array_equal(result, sample_face_image)

    def test_draw_landmarks_empty_list(
        self, sample_face_image: np.ndarray
    ) -> None:
        """Test drawing with empty landmarks list."""
        overlay = FaceOverlay()

        result = overlay.draw_landmarks_on_image(sample_face_image, [])

        # Should return copy of original
        assert result.shape == sample_face_image.shape
        assert np.array_equal(result, sample_face_image)

    def test_draw_bounding_box(
        self, sample_face_image: np.ndarray, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test drawing bounding box on image."""
        overlay = FaceOverlay(draw_landmarks=False, draw_bbox=True)

        result = overlay.draw_landmarks_on_image(
            sample_face_image, [sample_face_landmarks]
        )

        assert result.shape == sample_face_image.shape
        assert not np.array_equal(result, sample_face_image)

    def test_draw_emotion_label(
        self, sample_face_image: np.ndarray, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test drawing emotion label on image."""
        overlay = FaceOverlay()

        result = overlay.draw_emotion_label(
            sample_face_image,
            "happy",
            0.85,
            sample_face_landmarks.bounding_box,
        )

        assert result.shape == sample_face_image.shape
        # Image should be modified with text
        assert not np.array_equal(result, sample_face_image)

    def test_draw_emotion_label_no_bbox(
        self, sample_face_image: np.ndarray
    ) -> None:
        """Test drawing emotion label without bounding box."""
        overlay = FaceOverlay()

        result = overlay.draw_emotion_label(
            sample_face_image,
            "happy",
            0.85,
            None,  # No bbox
        )

        assert result.shape == sample_face_image.shape

    def test_draw_multiple_faces(
        self, sample_face_image: np.ndarray
    ) -> None:
        """Test drawing multiple faces on image."""
        overlay = FaceOverlay()

        # Create multiple face landmarks
        faces = []
        for i in range(3):
            landmarks = np.random.rand(478, 3).astype(np.float32)
            bbox = BoundingBox(
                x_min=0.1 + i * 0.3, y_min=0.2, width=0.25, height=0.3
            )
            face = FaceLandmarks(landmarks=landmarks, bounding_box=bbox)
            faces.append(face)

        result = overlay.draw_landmarks_on_image(sample_face_image, faces)

        assert result.shape == sample_face_image.shape

    def test_color_customization(
        self, sample_face_image: np.ndarray, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test customizing overlay colors."""
        overlay = FaceOverlay(
            landmark_color=(255, 0, 0),  # Red
            bbox_color=(0, 255, 0),  # Green
            text_color=(0, 0, 255),  # Blue
        )

        result = overlay.draw_landmarks_on_image(
            sample_face_image, [sample_face_landmarks]
        )

        assert result.shape == sample_face_image.shape

    def test_line_thickness_effect(
        self, sample_face_image: np.ndarray, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test effect of different line thickness."""
        thin_overlay = FaceOverlay(line_thickness=1)
        thick_overlay = FaceOverlay(line_thickness=5)

        thin_result = thin_overlay.draw_landmarks_on_image(
            sample_face_image, [sample_face_landmarks]
        )
        thick_result = thick_overlay.draw_landmarks_on_image(
            sample_face_image, [sample_face_landmarks]
        )

        assert thin_result.shape == thick_result.shape
        # Results should be different
        assert not np.array_equal(thin_result, thick_result)

    def test_preserves_original_image(
        self, sample_face_image: np.ndarray, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that original image is not modified."""
        overlay = FaceOverlay()
        original = sample_face_image.copy()

        _ = overlay.draw_landmarks_on_image(sample_face_image, [sample_face_landmarks])

        # Original should be unchanged
        assert np.array_equal(sample_face_image, original)


class TestFaceOverlayEdgeCases:
    """Test suite for edge cases in face overlay."""

    def test_empty_image(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test with zero-sized image."""
        overlay = FaceOverlay()
        empty_image = np.zeros((0, 0, 3), dtype=np.uint8)

        # Should handle gracefully or raise appropriate error
        try:
            result = overlay.draw_landmarks_on_image(empty_image, [sample_face_landmarks])
        except (ValueError, IndexError):
            pass  # Expected for invalid image

    def test_landmarks_outside_bounds(self, sample_face_image: np.ndarray) -> None:
        """Test with landmarks outside image bounds."""
        overlay = FaceOverlay()

        # Create landmarks with out-of-bounds coordinates
        landmarks = np.ones((478, 3), dtype=np.float32) * 2.0  # > 1.0
        face = FaceLandmarks(landmarks=landmarks)

        result = overlay.draw_landmarks_on_image(sample_face_image, [face])

        # Should handle without crashing
        assert result.shape == sample_face_image.shape

    def test_grayscale_image(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test with grayscale image."""
        overlay = FaceOverlay()
        gray_image = np.zeros((480, 640), dtype=np.uint8)

        # Should handle or convert appropriately
        try:
            result = overlay.draw_landmarks_on_image(gray_image, [sample_face_landmarks])
        except (ValueError, IndexError):
            pass  # May not support grayscale

    def test_very_long_emotion_text(
        self, sample_face_image: np.ndarray
    ) -> None:
        """Test with very long emotion label."""
        overlay = FaceOverlay()

        long_text = "This is a very long emotion label text that should be handled"
        result = overlay.draw_emotion_label(
            sample_face_image,
            long_text,
            0.99,
            BoundingBox(0.1, 0.1, 0.8, 0.8),
        )

        assert result.shape == sample_face_image.shape

    def test_zero_confidence(self, sample_face_image: np.ndarray) -> None:
        """Test drawing with zero confidence."""
        overlay = FaceOverlay()

        result = overlay.draw_emotion_label(
            sample_face_image,
            "neutral",
            0.0,
            BoundingBox(0.1, 0.1, 0.5, 0.5),
        )

        assert result.shape == sample_face_image.shape

    def test_special_characters_in_label(
        self, sample_face_image: np.ndarray
    ) -> None:
        """Test emotion label with special characters."""
        overlay = FaceOverlay()

        result = overlay.draw_emotion_label(
            sample_face_image,
            "happy 😊",  # Emoji
            0.85,
            BoundingBox(0.1, 0.1, 0.5, 0.5),
        )

        assert result.shape == sample_face_image.shape
