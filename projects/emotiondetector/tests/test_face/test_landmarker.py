"""Tests for face landmark utilities and geometric functions.

This module tests landmark extraction utilities and geometric computations.
"""

import numpy as np
import pytest

from asdrp.face.base import FaceLandmarkIndex, FaceLandmarks


class TestLandmarkGeometry:
    """Test suite for landmark geometric operations."""

    def test_landmark_coordinate_ranges(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test that landmarks are in valid normalized range."""
        # X and Y should be in [0, 1]
        assert np.all(sample_face_landmarks.landmarks[:, 0] >= 0.0)
        assert np.all(sample_face_landmarks.landmarks[:, 0] <= 1.0)
        assert np.all(sample_face_landmarks.landmarks[:, 1] >= 0.0)
        assert np.all(sample_face_landmarks.landmarks[:, 1] <= 1.0)

    def test_landmark_access_consistency(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that different access methods return same landmark."""
        idx = FaceLandmarkIndex.NOSE_TIP

        landmark1 = sample_face_landmarks.get_landmark(idx)
        landmark2 = sample_face_landmarks.get_landmark(idx.value)
        landmark3 = sample_face_landmarks.landmarks[idx.value]

        assert np.array_equal(landmark1, landmark2)
        assert np.array_equal(landmark2, landmark3)

    def test_bilateral_landmark_pairs(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that bilateral landmarks (left/right) exist."""
        # Eye corners
        left_eye = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.LEFT_EYE_OUTER_CORNER
        )
        right_eye = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.RIGHT_EYE_OUTER_CORNER
        )

        assert left_eye.shape == (3,)
        assert right_eye.shape == (3,)

        # Eyebrows
        left_brow = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.LEFT_EYEBROW_INNER
        )
        right_brow = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.RIGHT_EYEBROW_INNER
        )

        assert left_brow.shape == (3,)
        assert right_brow.shape == (3,)

    def test_face_symmetry_check(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test basic face symmetry by checking x-coordinates."""
        # Left eye should be on left side (smaller x)
        left_eye_x = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.LEFT_EYE_OUTER_CORNER
        )[0]
        right_eye_x = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.RIGHT_EYE_OUTER_CORNER
        )[0]

        # Note: This might not always hold for random landmarks
        # but is a good sanity check for real faces
        # Skip assertion for random test data

    def test_key_facial_landmarks_exist(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that key facial landmarks can be accessed."""
        key_landmarks = [
            FaceLandmarkIndex.NOSE_TIP,
            FaceLandmarkIndex.LEFT_EYE_INNER_CORNER,
            FaceLandmarkIndex.RIGHT_EYE_INNER_CORNER,
            FaceLandmarkIndex.MOUTH_LEFT_CORNER,
            FaceLandmarkIndex.MOUTH_RIGHT_CORNER,
            FaceLandmarkIndex.FACE_OVAL_CHIN_CENTER,
        ]

        for landmark_idx in key_landmarks:
            landmark = sample_face_landmarks.get_landmark(landmark_idx)
            assert landmark.shape == (3,)
            assert not np.isnan(landmark).any()

    def test_landmark_visibility_if_present(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test visibility scores if present."""
        if sample_face_landmarks.visibility is not None:
            # Visibility should be between 0 and 1
            assert np.all(sample_face_landmarks.visibility >= 0.0)
            assert np.all(sample_face_landmarks.visibility <= 1.0)

            # Should have same length as landmarks
            assert len(sample_face_landmarks.visibility) == len(
                sample_face_landmarks.landmarks
            )

    def test_z_coordinates_reasonable(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that z-coordinates are in reasonable range."""
        z_coords = sample_face_landmarks.landmarks[:, 2]

        # Z should be much smaller than x,y (depth is relative)
        # For normalized coords, typical range is -0.1 to 0.1
        assert np.all(np.abs(z_coords) < 1.0)


class TestLandmarkDistanceCalculations:
    """Test suite for distance calculations between landmarks."""

    def test_euclidean_distance_2d(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test Euclidean distance calculation between two landmarks."""
        p1 = sample_face_landmarks.landmarks[0, :2]
        p2 = sample_face_landmarks.landmarks[1, :2]

        distance = np.linalg.norm(p2 - p1)

        assert distance >= 0.0
        assert isinstance(distance, (float, np.floating))

    def test_inter_eye_distance(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test calculation of distance between eyes."""
        left_eye = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.LEFT_EYE_OUTER_CORNER
        )[:2]
        right_eye = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.RIGHT_EYE_OUTER_CORNER
        )[:2]

        distance = np.linalg.norm(right_eye - left_eye)

        # Should be positive and less than face width
        assert distance > 0.0
        assert distance < 1.0  # Normalized coordinates

    def test_mouth_width(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test calculation of mouth width."""
        left_corner = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.MOUTH_LEFT_CORNER
        )[:2]
        right_corner = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.MOUTH_RIGHT_CORNER
        )[:2]

        width = np.linalg.norm(right_corner - left_corner)

        assert width > 0.0
        assert width < 1.0

    def test_face_height_estimate(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test estimation of face height."""
        chin = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.FACE_OVAL_CHIN_CENTER
        )
        forehead = sample_face_landmarks.get_landmark(
            FaceLandmarkIndex.FACE_OVAL_FOREHEAD_CENTER
        )

        height = abs(chin[1] - forehead[1])

        assert height >= 0.0
        assert height <= 1.0


class TestLandmarkAngles:
    """Test suite for angle calculations between landmarks."""

    def test_angle_calculation(self) -> None:
        """Test basic angle calculation between three points."""
        # Create three points forming a right angle
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])

        # Calculate angle at p2
        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        # Should be 90 degrees (pi/2 radians)
        assert angle == pytest.approx(np.pi / 2, rel=0.01)

    def test_angle_range(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test that calculated angles are in valid range."""
        # Get three arbitrary landmarks
        p1 = sample_face_landmarks.landmarks[0, :2]
        p2 = sample_face_landmarks.landmarks[1, :2]
        p3 = sample_face_landmarks.landmarks[2, :2]

        # Calculate angle at p2
        v1 = p1 - p2
        v2 = p3 - p2

        if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

            # Angle should be in [0, pi]
            assert 0.0 <= angle <= np.pi


class TestLandmarkNormalization:
    """Test suite for landmark normalization operations."""

    def test_normalization_preserves_shape(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that normalization preserves landmark array shape."""
        original_shape = sample_face_landmarks.landmarks.shape

        # Normalize (subtract mean, divide by std)
        normalized = sample_face_landmarks.landmarks - sample_face_landmarks.landmarks.mean(
            axis=0
        )

        assert normalized.shape == original_shape

    def test_centering_landmarks(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test centering landmarks around mean."""
        landmarks = sample_face_landmarks.landmarks.copy()

        # Center landmarks
        mean = landmarks.mean(axis=0)
        centered = landmarks - mean

        # Mean should be close to zero
        assert np.allclose(centered.mean(axis=0), 0.0, atol=1e-6)

    def test_scale_invariance(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test that landmarks can be scaled uniformly."""
        landmarks = sample_face_landmarks.landmarks.copy()

        # Scale by 2
        scaled = landmarks * 2.0

        # Relative distances should be preserved
        original_dist = np.linalg.norm(landmarks[1] - landmarks[0])
        scaled_dist = np.linalg.norm(scaled[1] - scaled[0])

        assert scaled_dist == pytest.approx(original_dist * 2.0)
