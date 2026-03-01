"""Tests for face detection base classes and data structures.

This module tests the FaceLandmarks, BoundingBox dataclasses and related
functionality in the face.base module.
"""

import numpy as np
import pytest

from asdrp.face.base import BoundingBox, FaceLandmarkIndex, FaceLandmarks


class TestBoundingBox:
    """Test suite for BoundingBox dataclass."""

    def test_initialization(self) -> None:
        """Test basic BoundingBox initialization."""
        bbox = BoundingBox(x_min=0.1, y_min=0.2, width=0.5, height=0.6)
        assert bbox.x_min == 0.1
        assert bbox.y_min == 0.2
        assert bbox.width == 0.5
        assert bbox.height == 0.6

    def test_x_max_property(self) -> None:
        """Test x_max computed property."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, width=0.4, height=0.5)
        assert bbox.x_max == pytest.approx(0.6)

    def test_y_max_property(self) -> None:
        """Test y_max computed property."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, width=0.4, height=0.5)
        assert bbox.y_max == pytest.approx(0.8)

    def test_center_property(self) -> None:
        """Test center computed property."""
        bbox = BoundingBox(x_min=0.2, y_min=0.2, width=0.4, height=0.6)
        center_x, center_y = bbox.center
        assert center_x == pytest.approx(0.4)  # 0.2 + 0.4/2
        assert center_y == pytest.approx(0.5)  # 0.2 + 0.6/2

    def test_area_property(self) -> None:
        """Test area computed property."""
        bbox = BoundingBox(x_min=0.0, y_min=0.0, width=0.5, height=0.4)
        assert bbox.area == pytest.approx(0.2)

    def test_to_absolute(self) -> None:
        """Test conversion to absolute pixel coordinates."""
        bbox = BoundingBox(x_min=0.2, y_min=0.3, width=0.4, height=0.5)
        image_width, image_height = 640, 480

        x_min, y_min, x_max, y_max = bbox.to_absolute(image_width, image_height)

        assert x_min == 128  # 0.2 * 640
        assert y_min == 144  # 0.3 * 480
        assert x_max == 384  # (0.2 + 0.4) * 640
        assert y_max == 384  # (0.3 + 0.5) * 480

    def test_to_absolute_edge_cases(self) -> None:
        """Test to_absolute with edge case values."""
        # Full image bbox
        bbox = BoundingBox(x_min=0.0, y_min=0.0, width=1.0, height=1.0)
        x_min, y_min, x_max, y_max = bbox.to_absolute(100, 100)
        assert x_min == 0
        assert y_min == 0
        assert x_max == 100
        assert y_max == 100

        # Small bbox
        bbox = BoundingBox(x_min=0.5, y_min=0.5, width=0.1, height=0.1)
        x_min, y_min, x_max, y_max = bbox.to_absolute(100, 100)
        assert x_min == 50
        assert y_min == 50
        assert x_max == 60
        assert y_max == 60


class TestFaceLandmarks:
    """Test suite for FaceLandmarks dataclass."""

    def test_initialization_valid(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test valid FaceLandmarks initialization."""
        assert sample_face_landmarks.num_landmarks == 478
        assert sample_face_landmarks.landmarks.shape == (478, 3)
        assert sample_face_landmarks.visibility is not None
        assert sample_face_landmarks.bounding_box is not None

    def test_initialization_minimal(self) -> None:
        """Test FaceLandmarks initialization with minimal data."""
        landmarks = np.random.rand(10, 3).astype(np.float32)
        face = FaceLandmarks(landmarks=landmarks)

        assert face.num_landmarks == 10
        assert face.visibility is None
        assert face.bounding_box is None
        assert face.timestamp == 0.0
        assert face.frame_number == 0
        assert face.face_id == 0

    def test_initialization_invalid_shape(self) -> None:
        """Test that invalid landmark shape raises ValueError."""
        # 2D array with wrong second dimension
        with pytest.raises(ValueError, match="must be of shape"):
            landmarks = np.random.rand(10, 2).astype(np.float32)
            FaceLandmarks(landmarks=landmarks)

        # 1D array
        with pytest.raises(ValueError, match="must be of shape"):
            landmarks = np.random.rand(10).astype(np.float32)
            FaceLandmarks(landmarks=landmarks)

    def test_initialization_mismatched_visibility(self) -> None:
        """Test that mismatched visibility length raises ValueError."""
        landmarks = np.random.rand(10, 3).astype(np.float32)
        visibility = np.random.rand(5).astype(np.float32)  # Wrong length

        with pytest.raises(ValueError, match="visibility length"):
            FaceLandmarks(landmarks=landmarks, visibility=visibility)

    def test_initialization_invalid_visibility_shape(self) -> None:
        """Test that multi-dimensional visibility raises ValueError."""
        landmarks = np.random.rand(10, 3).astype(np.float32)
        visibility = np.random.rand(10, 2).astype(np.float32)

        with pytest.raises(ValueError, match="must be 1-dimensional"):
            FaceLandmarks(landmarks=landmarks, visibility=visibility)

    def test_num_landmarks_property(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test num_landmarks property."""
        assert sample_face_landmarks.num_landmarks == len(sample_face_landmarks.landmarks)

    def test_get_landmark_by_index(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test getting a landmark by integer index."""
        landmark = sample_face_landmarks.get_landmark(0)
        assert landmark.shape == (3,)
        assert np.array_equal(landmark, sample_face_landmarks.landmarks[0])

    def test_get_landmark_by_enum(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test getting a landmark by FaceLandmarkIndex enum."""
        landmark = sample_face_landmarks.get_landmark(FaceLandmarkIndex.NOSE_TIP)
        assert landmark.shape == (3,)
        assert np.array_equal(
            landmark, sample_face_landmarks.landmarks[FaceLandmarkIndex.NOSE_TIP.value]
        )

    def test_get_landmark_out_of_bounds(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test that out of bounds index raises IndexError."""
        with pytest.raises(IndexError):
            sample_face_landmarks.get_landmark(1000)

    def test_to_absolute(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test conversion to absolute pixel coordinates."""
        image_width, image_height = 640, 480
        absolute_landmarks = sample_face_landmarks.to_absolute(image_width, image_height)

        assert absolute_landmarks.shape == sample_face_landmarks.landmarks.shape

        # Check that coordinates are scaled correctly
        expected_x = sample_face_landmarks.landmarks[0, 0] * image_width
        expected_y = sample_face_landmarks.landmarks[0, 1] * image_height
        expected_z = sample_face_landmarks.landmarks[0, 2] * image_width

        assert absolute_landmarks[0, 0] == pytest.approx(expected_x)
        assert absolute_landmarks[0, 1] == pytest.approx(expected_y)
        assert absolute_landmarks[0, 2] == pytest.approx(expected_z)

    def test_to_absolute_preserves_original(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that to_absolute doesn't modify original landmarks."""
        original = sample_face_landmarks.landmarks.copy()
        _ = sample_face_landmarks.to_absolute(640, 480)

        assert np.array_equal(sample_face_landmarks.landmarks, original)


class TestFaceLandmarkIndex:
    """Test suite for FaceLandmarkIndex enum."""

    def test_enum_values(self) -> None:
        """Test that enum values are correct integers."""
        assert FaceLandmarkIndex.NOSE_TIP.value == 1
        assert FaceLandmarkIndex.LEFT_EYE_OUTER_CORNER.value == 33
        assert FaceLandmarkIndex.RIGHT_EYE_OUTER_CORNER.value == 362
        assert FaceLandmarkIndex.MOUTH_LEFT_CORNER.value == 61
        assert FaceLandmarkIndex.MOUTH_RIGHT_CORNER.value == 291

    def test_enum_members_exist(self) -> None:
        """Test that key facial landmarks are defined."""
        # Eyes
        assert hasattr(FaceLandmarkIndex, "LEFT_EYE_OUTER_CORNER")
        assert hasattr(FaceLandmarkIndex, "RIGHT_EYE_OUTER_CORNER")

        # Eyebrows
        assert hasattr(FaceLandmarkIndex, "LEFT_EYEBROW_INNER")
        assert hasattr(FaceLandmarkIndex, "RIGHT_EYEBROW_INNER")

        # Nose
        assert hasattr(FaceLandmarkIndex, "NOSE_TIP")
        assert hasattr(FaceLandmarkIndex, "NOSE_BRIDGE_TOP")

        # Mouth
        assert hasattr(FaceLandmarkIndex, "MOUTH_LEFT_CORNER")
        assert hasattr(FaceLandmarkIndex, "MOUTH_RIGHT_CORNER")
        assert hasattr(FaceLandmarkIndex, "MOUTH_UPPER_LIP_TOP_CENTER")
        assert hasattr(FaceLandmarkIndex, "MOUTH_LOWER_LIP_BOTTOM_CENTER")

        # Face oval
        assert hasattr(FaceLandmarkIndex, "FACE_OVAL_CHIN_CENTER")


class TestBaseFaceDetector:
    """Test suite for BaseFaceDetector abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseFaceDetector cannot be instantiated."""
        from asdrp.face.base import BaseFaceDetector

        with pytest.raises(TypeError):
            BaseFaceDetector()  # type: ignore

    def test_context_manager_protocol(self, mock_face_detector: any) -> None:
        """Test that detector can be used as context manager."""
        mock_face_detector.close = lambda: None

        with mock_face_detector as detector:
            assert detector is not None

        # Close should be called after exiting context
        mock_face_detector.close.assert_called_once()
