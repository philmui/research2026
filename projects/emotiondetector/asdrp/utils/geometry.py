"""Geometric utility functions for facial landmark analysis.

This module provides geometric calculations and transformations for 3D facial
landmarks, including distance computations, angle calculations, centroids,
and point normalization.
"""

from typing import Any, Dict, Tuple

import numpy as np
import numpy.typing as npt


def calculate_distance_3d(
    point1: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
) -> float:
    """Calculate Euclidean distance between two 3D points.

    Args:
        point1: First point as array of shape (3,) with (x, y, z) coordinates.
        point2: Second point as array of shape (3,) with (x, y, z) coordinates.

    Returns:
        Euclidean distance between the two points.

    Raises:
        ValueError: If points are not 3-dimensional.

    Example:
        >>> p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        >>> p2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        >>> distance = calculate_distance_3d(p1, p2)
        >>> print(f"{distance:.3f}")
        1.732
    """
    if point1.shape != (3,) or point2.shape != (3,):
        raise ValueError(
            f"Points must be 3D arrays of shape (3,), got shapes {point1.shape} and {point2.shape}"
        )

    return float(np.linalg.norm(point2 - point1))


def calculate_distance_2d(
    point1: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
) -> float:
    """Calculate Euclidean distance between two 2D points.

    Args:
        point1: First point as array of shape (2,) with (x, y) coordinates.
        point2: Second point as array of shape (2,) with (x, y) coordinates.

    Returns:
        Euclidean distance between the two points.

    Raises:
        ValueError: If points are not 2-dimensional.

    Example:
        >>> p1 = np.array([0.0, 0.0], dtype=np.float32)
        >>> p2 = np.array([3.0, 4.0], dtype=np.float32)
        >>> distance = calculate_distance_2d(p1, p2)
        >>> print(f"{distance:.1f}")
        5.0
    """
    if point1.shape != (2,) or point2.shape != (2,):
        raise ValueError(
            f"Points must be 2D arrays of shape (2,), got shapes {point1.shape} and {point2.shape}"
        )

    return float(np.linalg.norm(point2 - point1))


def calculate_angle_3d(
    point1: npt.NDArray[np.float32],
    vertex: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
) -> float:
    """Calculate angle between three 3D points with vertex as the center point.

    Computes the angle formed by the vectors (vertex -> point1) and (vertex -> point2).

    Args:
        point1: First point as array of shape (3,) with (x, y, z) coordinates.
        vertex: Vertex point (center) as array of shape (3,) with (x, y, z) coordinates.
        point2: Second point as array of shape (3,) with (x, y, z) coordinates.

    Returns:
        Angle in radians between 0 and π (0 to 180 degrees).

    Raises:
        ValueError: If points are not 3-dimensional or if vertex coincides with either point.

    Example:
        >>> p1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        >>> vertex = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        >>> p2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        >>> angle = calculate_angle_3d(p1, vertex, p2)
        >>> print(f"{np.degrees(angle):.1f} degrees")
        90.0 degrees
    """
    if point1.shape != (3,) or vertex.shape != (3,) or point2.shape != (3,):
        raise ValueError(
            f"Points must be 3D arrays of shape (3,), got shapes "
            f"{point1.shape}, {vertex.shape}, {point2.shape}"
        )

    # Create vectors from vertex to each point
    vector1 = point1 - vertex
    vector2 = point2 - vertex

    # Calculate magnitudes
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Vertex must not coincide with either point")

    # Normalize vectors
    vector1_normalized = vector1 / norm1
    vector2_normalized = vector2 / norm2

    # Calculate angle using dot product
    dot_product = np.dot(vector1_normalized, vector2_normalized)

    # Clip to handle numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle = np.arccos(dot_product)
    return float(angle)


def calculate_angle_2d(
    point1: npt.NDArray[np.float32],
    vertex: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
) -> float:
    """Calculate angle between three 2D points with vertex as the center point.

    Computes the angle formed by the vectors (vertex -> point1) and (vertex -> point2).

    Args:
        point1: First point as array of shape (2,) with (x, y) coordinates.
        vertex: Vertex point (center) as array of shape (2,) with (x, y) coordinates.
        point2: Second point as array of shape (2,) with (x, y) coordinates.

    Returns:
        Angle in radians between 0 and π (0 to 180 degrees).

    Raises:
        ValueError: If points are not 2-dimensional or if vertex coincides with either point.

    Example:
        >>> p1 = np.array([1.0, 0.0], dtype=np.float32)
        >>> vertex = np.array([0.0, 0.0], dtype=np.float32)
        >>> p2 = np.array([0.0, 1.0], dtype=np.float32)
        >>> angle = calculate_angle_2d(p1, vertex, p2)
        >>> print(f"{np.degrees(angle):.1f} degrees")
        90.0 degrees
    """
    if point1.shape != (2,) or vertex.shape != (2,) or point2.shape != (2,):
        raise ValueError(
            f"Points must be 2D arrays of shape (2,), got shapes "
            f"{point1.shape}, {vertex.shape}, {point2.shape}"
        )

    # Create vectors from vertex to each point
    vector1 = point1 - vertex
    vector2 = point2 - vertex

    # Calculate magnitudes
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 == 0.0 or norm2 == 0.0:
        raise ValueError("Vertex must not coincide with either point")

    # Normalize vectors
    vector1_normalized = vector1 / norm1
    vector2_normalized = vector2 / norm2

    # Calculate angle using dot product
    dot_product = np.dot(vector1_normalized, vector2_normalized)

    # Clip to handle numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)

    angle = np.arccos(dot_product)
    return float(angle)


def calculate_centroid(
    points: npt.NDArray[np.float32],
    weights: npt.NDArray[np.float32] | None = None,
) -> npt.NDArray[np.float32]:
    """Calculate the centroid (center of mass) of a set of points.

    Computes the weighted or unweighted average position of multiple points.

    Args:
        points: Array of shape (N, D) containing N points in D dimensions.
        weights: Optional array of shape (N,) containing weights for each point.
                If None, all points are weighted equally.

    Returns:
        Centroid as array of shape (D,) representing the center point.

    Raises:
        ValueError: If points array is empty, weights shape doesn't match points,
                   or weights sum to zero.

    Example:
        >>> points = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]], dtype=np.float32)
        >>> centroid = calculate_centroid(points)
        >>> print(centroid)
        [0.5 0.33333334]

        >>> weights = np.array([1.0, 1.0, 2.0], dtype=np.float32)
        >>> weighted_centroid = calculate_centroid(points, weights)
        >>> print(weighted_centroid)
        [0.5 0.5]
    """
    if points.size == 0:
        raise ValueError("Points array must not be empty")

    if points.ndim != 2:
        raise ValueError(f"Points must be 2D array of shape (N, D), got shape {points.shape}")

    if weights is None:
        # Unweighted centroid (simple mean)
        return np.mean(points, axis=0).astype(np.float32)

    if weights.ndim != 1:
        raise ValueError(f"Weights must be 1D array of shape (N,), got shape {weights.shape}")

    if len(weights) != len(points):
        raise ValueError(
            f"Number of weights ({len(weights)}) must match number of points ({len(points)})"
        )

    weight_sum = np.sum(weights)
    if weight_sum == 0.0:
        raise ValueError("Sum of weights must be non-zero")

    # Weighted centroid
    weighted_sum = np.sum(points * weights[:, np.newaxis], axis=0)
    centroid = (weighted_sum / weight_sum).astype(np.float32)
    return centroid


def point_line_distance(
    point: npt.NDArray[np.float32],
    line_point1: npt.NDArray[np.float32],
    line_point2: npt.NDArray[np.float32],
) -> float:
    """Calculate perpendicular distance from a point to a line in 3D space.

    Computes the shortest distance from a point to an infinite line defined by
    two points.

    Args:
        point: Point as array of shape (3,) with (x, y, z) coordinates.
        line_point1: First point on the line as array of shape (3,).
        line_point2: Second point on the line as array of shape (3,).

    Returns:
        Perpendicular distance from the point to the line.

    Raises:
        ValueError: If arrays are not 3-dimensional or if line points are identical.

    Example:
        >>> point = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        >>> line_p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        >>> line_p2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        >>> distance = point_line_distance(point, line_p1, line_p2)
        >>> print(f"{distance:.1f}")
        1.0
    """
    if point.shape != (3,) or line_point1.shape != (3,) or line_point2.shape != (3,):
        raise ValueError(
            f"All inputs must be 3D arrays of shape (3,), got shapes "
            f"{point.shape}, {line_point1.shape}, {line_point2.shape}"
        )

    # Vector from line_point1 to line_point2
    line_vector = line_point2 - line_point1
    line_length = np.linalg.norm(line_vector)

    if line_length == 0.0:
        raise ValueError("Line points must not be identical")

    # Vector from line_point1 to point
    point_vector = point - line_point1

    # Cross product gives area of parallelogram, divide by base to get height
    cross_product = np.cross(line_vector, point_vector)
    distance = np.linalg.norm(cross_product) / line_length

    return float(distance)


def point_segment_distance(
    point: npt.NDArray[np.float32],
    segment_point1: npt.NDArray[np.float32],
    segment_point2: npt.NDArray[np.float32],
) -> float:
    """Calculate minimum distance from a point to a line segment in 3D space.

    Unlike point_line_distance, this computes the distance to a finite line segment,
    which may be the distance to one of the endpoints.

    Args:
        point: Point as array of shape (3,) with (x, y, z) coordinates.
        segment_point1: First endpoint of the segment as array of shape (3,).
        segment_point2: Second endpoint of the segment as array of shape (3,).

    Returns:
        Minimum distance from the point to the line segment.

    Raises:
        ValueError: If arrays are not 3-dimensional.

    Example:
        >>> point = np.array([2.0, 0.0, 0.0], dtype=np.float32)
        >>> seg_p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        >>> seg_p2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        >>> distance = point_segment_distance(point, seg_p1, seg_p2)
        >>> print(f"{distance:.1f}")
        1.0
    """
    if point.shape != (3,) or segment_point1.shape != (3,) or segment_point2.shape != (3,):
        raise ValueError(
            f"All inputs must be 3D arrays of shape (3,), got shapes "
            f"{point.shape}, {segment_point1.shape}, {segment_point2.shape}"
        )

    # Vector from segment_point1 to segment_point2
    segment_vector = segment_point2 - segment_point1
    segment_length_squared = np.dot(segment_vector, segment_vector)

    # If segment is a point
    if segment_length_squared == 0.0:
        return calculate_distance_3d(point, segment_point1)

    # Calculate projection parameter t
    t = np.dot(point - segment_point1, segment_vector) / segment_length_squared

    # Clamp t to [0, 1] to stay on the segment
    t = np.clip(t, 0.0, 1.0)

    # Calculate closest point on segment
    closest_point = segment_point1 + t * segment_vector

    # Return distance to closest point
    return calculate_distance_3d(point, closest_point)


def normalize_points(
    points: npt.NDArray[np.float32],
    method: str = "standard",
) -> Tuple[npt.NDArray[np.float32], Dict[str, npt.NDArray[np.float32]]]:
    """Normalize a set of points using specified normalization method.

    Args:
        points: Array of shape (N, D) containing N points in D dimensions.
        method: Normalization method. Options:
               - 'standard': Zero mean and unit variance (z-score normalization)
               - 'minmax': Scale to [0, 1] range
               - 'center': Center around origin (zero mean only)
               - 'unit': Scale to unit norm

    Returns:
        Tuple containing:
        - Normalized points as array of shape (N, D)
        - Dictionary with normalization parameters for inverse transformation

    Raises:
        ValueError: If points array is empty or method is invalid.

    Example:
        >>> points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        >>> normalized, params = normalize_points(points, method='standard')
        >>> print(normalized)
        [[-1.2247449 -1.2247449]
         [ 0.        0.       ]
         [ 1.2247449  1.2247449]]
    """
    if points.size == 0:
        raise ValueError("Points array must not be empty")

    if points.ndim != 2:
        raise ValueError(f"Points must be 2D array of shape (N, D), got shape {points.shape}")

    normalized_points = points.copy()
    params: Dict[str, npt.NDArray[np.float32]] = {}

    if method == "standard":
        # Z-score normalization: (x - mean) / std
        mean = np.mean(points, axis=0)
        std = np.std(points, axis=0)
        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)
        normalized_points = (points - mean) / std
        params["mean"] = mean.astype(np.float32)
        params["std"] = std.astype(np.float32)

    elif method == "minmax":
        # Min-Max normalization: (x - min) / (max - min)
        min_vals = np.min(points, axis=0)
        max_vals = np.max(points, axis=0)
        range_vals = max_vals - min_vals
        # Avoid division by zero
        range_vals = np.where(range_vals == 0, 1.0, range_vals)
        normalized_points = (points - min_vals) / range_vals
        params["min"] = min_vals.astype(np.float32)
        params["max"] = max_vals.astype(np.float32)

    elif method == "center":
        # Center around origin (zero mean)
        mean = np.mean(points, axis=0)
        normalized_points = points - mean
        params["mean"] = mean.astype(np.float32)

    elif method == "unit":
        # Scale to unit norm
        centroid = calculate_centroid(points)
        centered_points = points - centroid
        max_distance = np.max(np.linalg.norm(centered_points, axis=1))
        if max_distance > 0:
            normalized_points = centered_points / max_distance
        else:
            normalized_points = centered_points
        params["centroid"] = centroid
        params["max_distance"] = np.array([max_distance], dtype=np.float32)

    else:
        raise ValueError(
            f"Invalid normalization method '{method}'. "
            f"Valid options: 'standard', 'minmax', 'center', 'unit'"
        )

    return normalized_points.astype(np.float32), params


def denormalize_points(
    normalized_points: npt.NDArray[np.float32],
    params: Dict[str, npt.NDArray[np.float32]],
    method: str = "standard",
) -> npt.NDArray[np.float32]:
    """Reverse normalization to get original point coordinates.

    Args:
        normalized_points: Array of shape (N, D) containing normalized points.
        params: Dictionary with normalization parameters (from normalize_points).
        method: Normalization method used (must match the one used for normalization).

    Returns:
        Denormalized points as array of shape (N, D).

    Raises:
        ValueError: If method is invalid or required parameters are missing.

    Example:
        >>> points = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        >>> normalized, params = normalize_points(points, method='standard')
        >>> original = denormalize_points(normalized, params, method='standard')
        >>> np.allclose(points, original)
        True
    """
    if method == "standard":
        if "mean" not in params or "std" not in params:
            raise ValueError("Missing 'mean' or 'std' in params for standard normalization")
        return (normalized_points * params["std"] + params["mean"]).astype(np.float32)

    elif method == "minmax":
        if "min" not in params or "max" not in params:
            raise ValueError("Missing 'min' or 'max' in params for minmax normalization")
        range_vals = params["max"] - params["min"]
        return (normalized_points * range_vals + params["min"]).astype(np.float32)

    elif method == "center":
        if "mean" not in params:
            raise ValueError("Missing 'mean' in params for center normalization")
        return (normalized_points + params["mean"]).astype(np.float32)

    elif method == "unit":
        if "centroid" not in params or "max_distance" not in params:
            raise ValueError("Missing 'centroid' or 'max_distance' in params for unit normalization")
        return (normalized_points * params["max_distance"] + params["centroid"]).astype(np.float32)

    else:
        raise ValueError(
            f"Invalid normalization method '{method}'. "
            f"Valid options: 'standard', 'minmax', 'center', 'unit'"
        )
