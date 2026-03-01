"""Tests for MediaPipe face detector implementation.

This module tests the MediaPipeFaceDetector class with mocking to avoid
requiring the actual MediaPipe model file.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from asdrp.face.base import FaceLandmarks
from asdrp.face.detector import MediaPipeFaceDetector


class TestMediaPipeFaceDetector:
    """Test suite for MediaPipeFaceDetector."""

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_initialization_valid(
        self, mock_exists: Mock, mock_landmarker: Mock
    ) -> None:
        """Test valid detector initialization."""
        detector = MediaPipeFaceDetector(
            model_path="fake_model.task",
            min_detection_confidence=0.6,
            min_tracking_confidence=0.7,
            num_faces=2,
            running_mode="VIDEO",
        )

        assert detector.min_detection_confidence == 0.6
        assert detector.min_tracking_confidence == 0.7
        assert detector.num_faces == 2
        assert detector.running_mode == "VIDEO"

    def test_initialization_model_not_found(self) -> None:
        """Test that FileNotFoundError is raised for missing model."""
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            MediaPipeFaceDetector(model_path="nonexistent_model.task")

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_initialization_invalid_confidence(
        self, mock_exists: Mock, mock_landmarker: Mock
    ) -> None:
        """Test that invalid confidence values raise ValueError."""
        with pytest.raises(ValueError, match="min_detection_confidence"):
            MediaPipeFaceDetector(
                model_path="fake_model.task", min_detection_confidence=1.5
            )

        with pytest.raises(ValueError, match="min_tracking_confidence"):
            MediaPipeFaceDetector(
                model_path="fake_model.task", min_tracking_confidence=-0.1
            )

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_initialization_invalid_running_mode(
        self, mock_exists: Mock, mock_landmarker: Mock
    ) -> None:
        """Test that invalid running mode raises ValueError."""
        with pytest.raises(ValueError, match="running_mode must be"):
            MediaPipeFaceDetector(
                model_path="fake_model.task", running_mode="INVALID"
            )

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_detect_valid_image(
        self, mock_exists: Mock, mock_landmarker_class: Mock, sample_face_image: np.ndarray
    ) -> None:
        """Test face detection on valid image."""
        # Setup mock
        mock_detector = Mock()
        mock_landmarker_class.create_from_options.return_value = mock_detector

        # Create mock landmarks
        mock_landmark = Mock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.0

        mock_result = Mock()
        mock_result.face_landmarks = [[mock_landmark] * 478]

        mock_detector.detect.return_value = mock_result

        # Test detection
        detector = MediaPipeFaceDetector(model_path="fake_model.task")
        faces = detector.detect(sample_face_image, timestamp_ms=1000.0)

        assert len(faces) >= 0
        assert isinstance(faces, list)

        # Verify detect was called
        mock_detector.detect.assert_called_once()

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_detect_invalid_image_shape(
        self, mock_exists: Mock, mock_landmarker: Mock
    ) -> None:
        """Test that invalid image shape raises ValueError."""
        detector = MediaPipeFaceDetector(model_path="fake_model.task")

        # 2D image (grayscale)
        with pytest.raises(ValueError, match="must have shape"):
            detector.detect(np.zeros((100, 100), dtype=np.uint8))

        # 4D image
        with pytest.raises(ValueError, match="must have shape"):
            detector.detect(np.zeros((1, 100, 100, 3), dtype=np.uint8))

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_detect_invalid_image_dtype(
        self, mock_exists: Mock, mock_landmarker: Mock
    ) -> None:
        """Test that invalid image dtype raises ValueError."""
        detector = MediaPipeFaceDetector(model_path="fake_model.task")

        with pytest.raises(ValueError, match="must have dtype uint8"):
            detector.detect(np.zeros((100, 100, 3), dtype=np.float32))

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_detect_no_faces(
        self, mock_exists: Mock, mock_landmarker_class: Mock, sample_face_image: np.ndarray
    ) -> None:
        """Test detection when no faces are found."""
        mock_detector = Mock()
        mock_landmarker_class.create_from_options.return_value = mock_detector

        # Mock no faces detected
        mock_result = Mock()
        mock_result.face_landmarks = []
        mock_detector.detect.return_value = mock_result

        detector = MediaPipeFaceDetector(model_path="fake_model.task")
        faces = detector.detect(sample_face_image)

        assert len(faces) == 0
        assert isinstance(faces, list)

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_detect_batch_valid(
        self, mock_exists: Mock, mock_landmarker_class: Mock
    ) -> None:
        """Test batch detection on multiple images."""
        mock_detector = Mock()
        mock_landmarker_class.create_from_options.return_value = mock_detector

        # Mock detection result
        mock_result = Mock()
        mock_result.face_landmarks = []
        mock_detector.detect.return_value = mock_result

        detector = MediaPipeFaceDetector(model_path="fake_model.task")

        images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        timestamps = [0.0, 33.0, 66.0]

        results = detector.detect_batch(images, timestamps_ms=timestamps)

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_detect_batch_no_timestamps(
        self, mock_exists: Mock, mock_landmarker_class: Mock
    ) -> None:
        """Test batch detection without explicit timestamps."""
        mock_detector = Mock()
        mock_landmarker_class.create_from_options.return_value = mock_detector

        mock_result = Mock()
        mock_result.face_landmarks = []
        mock_detector.detect.return_value = mock_result

        detector = MediaPipeFaceDetector(model_path="fake_model.task")

        images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        results = detector.detect_batch(images)

        assert len(results) == 3

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_detect_batch_mismatched_timestamps(
        self, mock_exists: Mock, mock_landmarker: Mock
    ) -> None:
        """Test that mismatched timestamps raise ValueError."""
        detector = MediaPipeFaceDetector(model_path="fake_model.task")

        images = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
        timestamps = [0.0, 33.0]  # Wrong length

        with pytest.raises(ValueError, match="timestamps_ms length"):
            detector.detect_batch(images, timestamps_ms=timestamps)

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_detect_batch_empty_list(
        self, mock_exists: Mock, mock_landmarker: Mock
    ) -> None:
        """Test batch detection with empty image list."""
        detector = MediaPipeFaceDetector(model_path="fake_model.task")
        results = detector.detect_batch([])
        assert results == []

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_close(self, mock_exists: Mock, mock_landmarker_class: Mock) -> None:
        """Test detector cleanup."""
        mock_detector = Mock()
        mock_landmarker_class.create_from_options.return_value = mock_detector

        detector = MediaPipeFaceDetector(model_path="fake_model.task")
        detector.close()

        mock_detector.close.assert_called_once()

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_context_manager(
        self, mock_exists: Mock, mock_landmarker_class: Mock
    ) -> None:
        """Test using detector as context manager."""
        mock_detector = Mock()
        mock_landmarker_class.create_from_options.return_value = mock_detector

        with MediaPipeFaceDetector(model_path="fake_model.task") as detector:
            assert detector is not None

        mock_detector.close.assert_called_once()

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_compute_bounding_box(
        self, mock_exists: Mock, mock_landmarker: Mock
    ) -> None:
        """Test bounding box computation from landmarks."""
        detector = MediaPipeFaceDetector(model_path="fake_model.task")

        # Create test landmarks with known bounds
        landmarks = np.array(
            [[0.2, 0.3, 0.0], [0.8, 0.3, 0.0], [0.5, 0.9, 0.0]], dtype=np.float32
        )

        bbox = detector._compute_bounding_box(landmarks)

        # Should have some padding added
        assert bbox.x_min < 0.2
        assert bbox.y_min < 0.3
        assert bbox.x_min + bbox.width > 0.8
        assert bbox.y_min + bbox.height > 0.9

    @patch("asdrp.face.detector.vision.FaceLandmarker")
    @patch("asdrp.face.detector.Path.exists", return_value=True)
    def test_repr(self, mock_exists: Mock, mock_landmarker: Mock) -> None:
        """Test string representation."""
        detector = MediaPipeFaceDetector(
            model_path="fake_model.task", num_faces=2, running_mode="VIDEO"
        )

        repr_str = repr(detector)
        assert "MediaPipeFaceDetector" in repr_str
        assert "num_faces=2" in repr_str
        assert "VIDEO" in repr_str
