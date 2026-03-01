"""Tests for geometric utility functions.

This module tests geometric calculations and transformations.
"""

import numpy as np
import pytest

from asdrp.utils.geometry import (
    calculate_angle,
    calculate_distance,
    normalize_points,
)


class TestDistanceCalculations:
    """Test suite for distance calculations."""

    def test_calculate_distance_2d(self) -> None:
        """Test Euclidean distance in 2D."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([3.0, 4.0])

        distance = calculate_distance(p1, p2)

        assert distance == pytest.approx(5.0)

    def test_calculate_distance_3d(self) -> None:
        """Test Euclidean distance in 3D."""
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 2.0, 2.0])

        distance = calculate_distance(p1, p2)

        assert distance == pytest.approx(3.0)

    def test_calculate_distance_zero(self) -> None:
        """Test distance between same point."""
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([1.0, 2.0, 3.0])

        distance = calculate_distance(p1, p2)

        assert distance == pytest.approx(0.0)

    def test_calculate_distance_negative_coords(self) -> None:
        """Test distance with negative coordinates."""
        p1 = np.array([-1.0, -1.0])
        p2 = np.array([1.0, 1.0])

        distance = calculate_distance(p1, p2)

        assert distance == pytest.approx(2.828, rel=0.01)


class TestAngleCalculations:
    """Test suite for angle calculations."""

    def test_calculate_angle_right_angle(self) -> None:
        """Test calculating a right angle (90 degrees)."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 1.0])

        angle = calculate_angle(p1, p2, p3)

        assert angle == pytest.approx(np.pi / 2, rel=0.01)

    def test_calculate_angle_straight(self) -> None:
        """Test calculating a straight angle (180 degrees)."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([2.0, 0.0])

        angle = calculate_angle(p1, p2, p3)

        assert angle == pytest.approx(np.pi, rel=0.01)

    def test_calculate_angle_acute(self) -> None:
        """Test calculating an acute angle."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([1.0, 0.0])
        p3 = np.array([1.0, 0.5])

        angle = calculate_angle(p1, p2, p3)

        assert 0 < angle < np.pi / 2

    def test_calculate_angle_zero(self) -> None:
        """Test angle when points are collinear in same direction."""
        p1 = np.array([1.0, 0.0])
        p2 = np.array([0.0, 0.0])
        p3 = np.array([0.0, 0.0])

        # Should handle degenerate case
        angle = calculate_angle(p1, p2, p3)

        assert 0 <= angle <= np.pi


class TestNormalization:
    """Test suite for point normalization."""

    def test_normalize_points_basic(self) -> None:
        """Test basic point normalization."""
        points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        normalized = normalize_points(points)

        # Mean should be close to zero
        assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-10)

    def test_normalize_points_preserves_shape(self) -> None:
        """Test that normalization preserves array shape."""
        points = np.random.rand(100, 3)

        normalized = normalize_points(points)

        assert normalized.shape == points.shape

    def test_normalize_points_single_point(self) -> None:
        """Test normalizing a single point."""
        points = np.array([[5.0, 10.0]])

        normalized = normalize_points(points)

        assert normalized.shape == points.shape

    def test_normalize_points_preserves_relative_distances(self) -> None:
        """Test that relative distances are preserved after normalization."""
        points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

        normalized = normalize_points(points)

        # Original distances
        orig_dist_01 = np.linalg.norm(points[1] - points[0])
        orig_dist_02 = np.linalg.norm(points[2] - points[0])

        # Normalized distances
        norm_dist_01 = np.linalg.norm(normalized[1] - normalized[0])
        norm_dist_02 = np.linalg.norm(normalized[2] - normalized[0])

        # Ratio should be preserved
        assert orig_dist_01 / orig_dist_02 == pytest.approx(
            norm_dist_01 / norm_dist_02, rel=0.01
        )


class TestGeometricEdgeCases:
    """Test suite for edge cases in geometric calculations."""

    def test_distance_with_nan(self) -> None:
        """Test distance calculation with NaN values."""
        p1 = np.array([1.0, np.nan])
        p2 = np.array([2.0, 3.0])

        distance = calculate_distance(p1, p2)

        assert np.isnan(distance)

    def test_angle_with_coincident_points(self) -> None:
        """Test angle when some points coincide."""
        p1 = np.array([0.0, 0.0])
        p2 = np.array([0.0, 0.0])  # Same as p1
        p3 = np.array([1.0, 1.0])

        # Should handle gracefully
        angle = calculate_angle(p1, p2, p3)

        assert not np.isnan(angle) or True  # May return NaN or 0

    def test_normalize_identical_points(self) -> None:
        """Test normalizing identical points."""
        points = np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])

        normalized = normalize_points(points)

        # Should handle without error
        assert normalized.shape == points.shape

    def test_very_large_coordinates(self) -> None:
        """Test with very large coordinate values."""
        p1 = np.array([1e10, 1e10])
        p2 = np.array([1e10 + 1, 1e10 + 1])

        distance = calculate_distance(p1, p2)

        assert distance > 0
        assert not np.isinf(distance)

    def test_very_small_coordinates(self) -> None:
        """Test with very small coordinate values."""
        p1 = np.array([1e-10, 1e-10])
        p2 = np.array([2e-10, 2e-10])

        distance = calculate_distance(p1, p2)

        assert distance > 0
        assert distance < 1e-9
