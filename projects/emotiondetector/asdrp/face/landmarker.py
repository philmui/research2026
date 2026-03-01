"""Utility functions for working with facial landmarks.

This module provides helper functions for extracting specific landmark groups,
calculating geometric properties, and normalizing landmark data for analysis.
"""

from typing import Literal

import numpy as np
import numpy.typing as npt

from .base import FaceLandmarkIndex, FaceLandmarks


# Landmark group definitions for easy access
LANDMARK_GROUPS = {
    "left_eye": [
        FaceLandmarkIndex.LEFT_EYE_OUTER_CORNER,
        FaceLandmarkIndex.LEFT_EYE_TOP_UPPER,
        FaceLandmarkIndex.LEFT_EYE_INNER_CORNER,
        FaceLandmarkIndex.LEFT_EYE_TOP_LOWER,
        FaceLandmarkIndex.LEFT_EYE_BOTTOM_LOWER,
        FaceLandmarkIndex.LEFT_EYE_BOTTOM_UPPER,
    ],
    "right_eye": [
        FaceLandmarkIndex.RIGHT_EYE_OUTER_CORNER,
        FaceLandmarkIndex.RIGHT_EYE_TOP_UPPER,
        FaceLandmarkIndex.RIGHT_EYE_INNER_CORNER,
        FaceLandmarkIndex.RIGHT_EYE_TOP_LOWER,
        FaceLandmarkIndex.RIGHT_EYE_BOTTOM_LOWER,
        FaceLandmarkIndex.RIGHT_EYE_BOTTOM_UPPER,
    ],
    "left_eyebrow": [
        FaceLandmarkIndex.LEFT_EYEBROW_INNER,
        FaceLandmarkIndex.LEFT_EYEBROW_CENTER,
        FaceLandmarkIndex.LEFT_EYEBROW_OUTER,
        FaceLandmarkIndex.LEFT_EYEBROW_UPPER_INNER,
        FaceLandmarkIndex.LEFT_EYEBROW_UPPER_OUTER,
    ],
    "right_eyebrow": [
        FaceLandmarkIndex.RIGHT_EYEBROW_INNER,
        FaceLandmarkIndex.RIGHT_EYEBROW_CENTER,
        FaceLandmarkIndex.RIGHT_EYEBROW_OUTER,
        FaceLandmarkIndex.RIGHT_EYEBROW_UPPER_INNER,
        FaceLandmarkIndex.RIGHT_EYEBROW_UPPER_OUTER,
    ],
    "nose": [
        FaceLandmarkIndex.NOSE_TIP,
        FaceLandmarkIndex.NOSE_BRIDGE_TOP,
        FaceLandmarkIndex.NOSE_BRIDGE_CENTER,
        FaceLandmarkIndex.NOSE_LEFT_NOSTRIL,
        FaceLandmarkIndex.NOSE_RIGHT_NOSTRIL,
        FaceLandmarkIndex.NOSE_LEFT_ALAR,
        FaceLandmarkIndex.NOSE_RIGHT_ALAR,
    ],
    "mouth_outer": [
        FaceLandmarkIndex.MOUTH_LEFT_CORNER,
        FaceLandmarkIndex.MOUTH_UPPER_LIP_TOP_LEFT,
        FaceLandmarkIndex.MOUTH_UPPER_LIP_TOP_CENTER,
        FaceLandmarkIndex.MOUTH_UPPER_LIP_TOP_RIGHT,
        FaceLandmarkIndex.MOUTH_RIGHT_CORNER,
        FaceLandmarkIndex.MOUTH_LOWER_LIP_BOTTOM_RIGHT,
        FaceLandmarkIndex.MOUTH_LOWER_LIP_BOTTOM_CENTER,
        FaceLandmarkIndex.MOUTH_LOWER_LIP_BOTTOM_LEFT,
    ],
    "mouth_inner": [
        FaceLandmarkIndex.MOUTH_INNER_UPPER_LEFT,
        FaceLandmarkIndex.MOUTH_INNER_UPPER_CENTER,
        FaceLandmarkIndex.MOUTH_INNER_UPPER_RIGHT,
        FaceLandmarkIndex.MOUTH_INNER_LOWER_RIGHT,
        FaceLandmarkIndex.MOUTH_INNER_LOWER_CENTER,
        FaceLandmarkIndex.MOUTH_INNER_LOWER_LEFT,
    ],
    "face_oval": [
        FaceLandmarkIndex.FACE_OVAL_LEFT_TOP,
        FaceLandmarkIndex.FACE_OVAL_LEFT_MIDDLE,
        FaceLandmarkIndex.FACE_OVAL_LEFT_BOTTOM,
        FaceLandmarkIndex.FACE_OVAL_CHIN_LEFT,
        FaceLandmarkIndex.FACE_OVAL_CHIN_CENTER,
        FaceLandmarkIndex.FACE_OVAL_CHIN_RIGHT,
        FaceLandmarkIndex.FACE_OVAL_RIGHT_BOTTOM,
        FaceLandmarkIndex.FACE_OVAL_RIGHT_MIDDLE,
        FaceLandmarkIndex.FACE_OVAL_RIGHT_TOP,
        FaceLandmarkIndex.FACE_OVAL_FOREHEAD_LEFT,
        FaceLandmarkIndex.FACE_OVAL_FOREHEAD_CENTER,
        FaceLandmarkIndex.FACE_OVAL_FOREHEAD_RIGHT,
    ],
}


def get_landmark_group(
    face_landmarks: FaceLandmarks,
    group_name: Literal[
        "left_eye",
        "right_eye",
        "left_eyebrow",
        "right_eyebrow",
        "nose",
        "mouth_outer",
        "mouth_inner",
        "face_oval",
    ],
) -> npt.NDArray[np.float32]:
    """Get landmarks for a specific facial region.

    Args:
        face_landmarks: FaceLandmarks object containing all facial landmarks
        group_name: Name of the landmark group to extract

    Returns:
        Array of shape (N, 3) containing the landmarks for the specified group

    Raises:
        ValueError: If group_name is not recognized

    Example:
        >>> left_eye_landmarks = get_landmark_group(face_landmarks, "left_eye")
        >>> print(left_eye_landmarks.shape)  # (6, 3)
    """
    if group_name not in LANDMARK_GROUPS:
        raise ValueError(
            f"Unknown landmark group: {group_name}. "
            f"Valid groups: {list(LANDMARK_GROUPS.keys())}"
        )

    indices = [idx.value for idx in LANDMARK_GROUPS[group_name]]
    return face_landmarks.landmarks[indices]


def get_left_eye(face_landmarks: FaceLandmarks) -> npt.NDArray[np.float32]:
    """Get left eye landmarks.

    Args:
        face_landmarks: FaceLandmarks object

    Returns:
        Array of shape (6, 3) with left eye landmarks
    """
    return get_landmark_group(face_landmarks, "left_eye")


def get_right_eye(face_landmarks: FaceLandmarks) -> npt.NDArray[np.float32]:
    """Get right eye landmarks.

    Args:
        face_landmarks: FaceLandmarks object

    Returns:
        Array of shape (6, 3) with right eye landmarks
    """
    return get_landmark_group(face_landmarks, "right_eye")


def get_left_eyebrow(face_landmarks: FaceLandmarks) -> npt.NDArray[np.float32]:
    """Get left eyebrow landmarks.

    Args:
        face_landmarks: FaceLandmarks object

    Returns:
        Array of shape (5, 3) with left eyebrow landmarks
    """
    return get_landmark_group(face_landmarks, "left_eyebrow")


def get_right_eyebrow(face_landmarks: FaceLandmarks) -> npt.NDArray[np.float32]:
    """Get right eyebrow landmarks.

    Args:
        face_landmarks: FaceLandmarks object

    Returns:
        Array of shape (5, 3) with right eyebrow landmarks
    """
    return get_landmark_group(face_landmarks, "right_eyebrow")


def get_nose(face_landmarks: FaceLandmarks) -> npt.NDArray[np.float32]:
    """Get nose landmarks.

    Args:
        face_landmarks: FaceLandmarks object

    Returns:
        Array of shape (7, 3) with nose landmarks
    """
    return get_landmark_group(face_landmarks, "nose")


def get_mouth(face_landmarks: FaceLandmarks) -> npt.NDArray[np.float32]:
    """Get outer mouth landmarks.

    Args:
        face_landmarks: FaceLandmarks object

    Returns:
        Array of shape (8, 3) with outer mouth landmarks
    """
    return get_landmark_group(face_landmarks, "mouth_outer")


def get_face_oval(face_landmarks: FaceLandmarks) -> npt.NDArray[np.float32]:
    """Get face oval landmarks.

    Args:
        face_landmarks: FaceLandmarks object

    Returns:
        Array of shape (12, 3) with face oval landmarks
    """
    return get_landmark_group(face_landmarks, "face_oval")


def calculate_distance(
    point1: npt.NDArray[np.float32], point2: npt.NDArray[np.float32]
) -> float:
    """Calculate Euclidean distance between two 3D points.

    Args:
        point1: First point as array of shape (3,) with (x, y, z) coordinates
        point2: Second point as array of shape (3,) with (x, y, z) coordinates

    Returns:
        Euclidean distance between the two points

    Example:
        >>> p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        >>> p2 = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        >>> distance = calculate_distance(p1, p2)
        >>> print(f"{distance:.3f}")  # 1.732
    """
    return float(np.linalg.norm(point1 - point2))


def calculate_eye_aspect_ratio(eye_landmarks: npt.NDArray[np.float32]) -> float:
    """Calculate Eye Aspect Ratio (EAR) for blink detection.

    The EAR measures the ratio of eye height to eye width. Lower values
    indicate closed or partially closed eyes.

    Args:
        eye_landmarks: Array of shape (6, 3) with eye landmarks in order:
                      [outer_corner, top_upper, inner_corner, top_lower,
                       bottom_lower, bottom_upper]

    Returns:
        Eye aspect ratio (typically 0.2-0.4, lower when eyes are closed)

    Reference:
        Soukupová, T., & Čech, J. (2016). Real-time eye blink detection using
        facial landmarks. 21st computer vision winter workshop.

    Example:
        >>> left_eye = get_left_eye(face_landmarks)
        >>> ear = calculate_eye_aspect_ratio(left_eye)
        >>> is_blinking = ear < 0.2
    """
    # Vertical distances
    v1 = calculate_distance(eye_landmarks[1], eye_landmarks[4])  # top to bottom
    v2 = calculate_distance(eye_landmarks[5], eye_landmarks[3])  # top to bottom (inner)

    # Horizontal distance
    h = calculate_distance(eye_landmarks[0], eye_landmarks[2])  # outer to inner corner

    # Avoid division by zero
    if h < 1e-6:
        return 0.0

    ear = (v1 + v2) / (2.0 * h)
    return ear


def calculate_mouth_aspect_ratio(mouth_landmarks: npt.NDArray[np.float32]) -> float:
    """Calculate Mouth Aspect Ratio (MAR) for mouth opening detection.

    The MAR measures the ratio of mouth height to mouth width. Higher values
    indicate more open mouths.

    Args:
        mouth_landmarks: Array of shape (8, 3) with outer mouth landmarks

    Returns:
        Mouth aspect ratio (higher values indicate open mouth)

    Example:
        >>> mouth = get_mouth(face_landmarks)
        >>> mar = calculate_mouth_aspect_ratio(mouth)
        >>> is_mouth_open = mar > 0.5
    """
    # Vertical distance (top to bottom at center)
    v = calculate_distance(mouth_landmarks[2], mouth_landmarks[6])

    # Horizontal distance (left corner to right corner)
    h = calculate_distance(mouth_landmarks[0], mouth_landmarks[4])

    # Avoid division by zero
    if h < 1e-6:
        return 0.0

    mar = v / h
    return mar


def calculate_angle(
    point1: npt.NDArray[np.float32],
    vertex: npt.NDArray[np.float32],
    point2: npt.NDArray[np.float32],
) -> float:
    """Calculate angle in degrees between three points.

    Calculates the angle at the vertex formed by the three points.

    Args:
        point1: First point as array of shape (3,)
        vertex: Vertex point as array of shape (3,)
        point2: Second point as array of shape (3,)

    Returns:
        Angle in degrees (0 to 180)

    Example:
        >>> p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        >>> vertex = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        >>> p2 = np.array([1.0, 1.0, 0.0], dtype=np.float32)
        >>> angle = calculate_angle(p1, vertex, p2)
        >>> print(f"{angle:.1f}")  # 90.0
    """
    # Create vectors
    vector1 = point1 - vertex
    vector2 = point2 - vertex

    # Calculate magnitudes
    mag1 = np.linalg.norm(vector1)
    mag2 = np.linalg.norm(vector2)

    # Avoid division by zero
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0

    # Calculate angle using dot product
    cos_angle = np.dot(vector1, vector2) / (mag1 * mag2)
    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = float(np.degrees(angle_rad))

    return angle_deg


def normalize_landmarks(
    face_landmarks: FaceLandmarks, method: Literal["center", "eyes"] = "center"
) -> npt.NDArray[np.float32]:
    """Normalize landmarks to be pose-invariant.

    Normalization helps make landmark features invariant to head pose,
    position, and scale, which improves emotion detection robustness.

    Args:
        face_landmarks: FaceLandmarks object to normalize
        method: Normalization method:
               - 'center': Center landmarks at origin and scale by face size
               - 'eyes': Align landmarks based on eye positions

    Returns:
        Normalized landmarks array of shape (N, 3)

    Example:
        >>> normalized = normalize_landmarks(face_landmarks, method="center")
        >>> # Landmarks now centered at origin with unit scale
    """
    landmarks = face_landmarks.landmarks.copy()

    if method == "center":
        # Center at mean position
        center = np.mean(landmarks, axis=0)
        landmarks = landmarks - center

        # Scale by standard deviation to normalize size
        scale = np.std(landmarks)
        if scale > 1e-6:
            landmarks = landmarks / scale

    elif method == "eyes":
        # Get eye positions
        left_eye_center = face_landmarks.get_landmark(FaceLandmarkIndex.LEFT_EYE_INNER_CORNER)
        right_eye_center = face_landmarks.get_landmark(FaceLandmarkIndex.RIGHT_EYE_INNER_CORNER)

        # Calculate eye distance for scaling
        eye_distance = calculate_distance(left_eye_center, right_eye_center)

        # Center between eyes
        eyes_center = (left_eye_center + right_eye_center) / 2.0
        landmarks = landmarks - eyes_center

        # Scale by eye distance
        if eye_distance > 1e-6:
            landmarks = landmarks / eye_distance

        # Rotate to align eyes horizontally (only in x-y plane)
        eye_vector = right_eye_center - left_eye_center
        angle = np.arctan2(eye_vector[1], eye_vector[0])

        # Create 2D rotation matrix
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)

        # Rotate x and y coordinates
        x_rot = landmarks[:, 0] * cos_a - landmarks[:, 1] * sin_a
        y_rot = landmarks[:, 0] * sin_a + landmarks[:, 1] * cos_a
        landmarks[:, 0] = x_rot
        landmarks[:, 1] = y_rot

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return landmarks


def get_inter_landmark_distances(
    face_landmarks: FaceLandmarks, landmark_pairs: list[tuple[int, int]]
) -> npt.NDArray[np.float32]:
    """Calculate distances between specified pairs of landmarks.

    Useful for extracting geometric features for emotion detection.

    Args:
        face_landmarks: FaceLandmarks object
        landmark_pairs: List of (index1, index2) tuples specifying landmark pairs

    Returns:
        Array of distances, one for each pair

    Example:
        >>> # Calculate distances between eye corners and mouth corners
        >>> pairs = [
        ...     (FaceLandmarkIndex.LEFT_EYE_OUTER_CORNER, FaceLandmarkIndex.RIGHT_EYE_OUTER_CORNER),
        ...     (FaceLandmarkIndex.MOUTH_LEFT_CORNER, FaceLandmarkIndex.MOUTH_RIGHT_CORNER),
        ... ]
        >>> distances = get_inter_landmark_distances(face_landmarks, pairs)
    """
    distances = []
    for idx1, idx2 in landmark_pairs:
        point1 = face_landmarks.get_landmark(idx1)
        point2 = face_landmarks.get_landmark(idx2)
        distance = calculate_distance(point1, point2)
        distances.append(distance)

    return np.array(distances, dtype=np.float32)


def extract_geometric_features(face_landmarks: FaceLandmarks) -> dict[str, float]:
    """Extract comprehensive geometric features from face landmarks.

    Extracts various geometric features useful for emotion detection, including
    eye aspect ratios, mouth aspect ratio, and inter-landmark distances.

    Args:
        face_landmarks: FaceLandmarks object

    Returns:
        Dictionary containing geometric features:
        - left_ear: Left eye aspect ratio
        - right_ear: Right eye aspect ratio
        - mar: Mouth aspect ratio
        - eye_distance: Distance between eye centers
        - eyebrow_eye_distance_left: Distance between left eyebrow and eye
        - eyebrow_eye_distance_right: Distance between right eyebrow and eye
        - mouth_width: Width of mouth
        - mouth_height: Height of mouth

    Example:
        >>> features = extract_geometric_features(face_landmarks)
        >>> print(f"Left eye open: {features['left_ear'] > 0.2}")
        >>> print(f"Mouth open: {features['mar'] > 0.5}")
    """
    # Eye aspect ratios
    left_eye = get_left_eye(face_landmarks)
    right_eye = get_right_eye(face_landmarks)
    left_ear = calculate_eye_aspect_ratio(left_eye)
    right_ear = calculate_eye_aspect_ratio(right_eye)

    # Mouth aspect ratio
    mouth = get_mouth(face_landmarks)
    mar = calculate_mouth_aspect_ratio(mouth)

    # Inter-landmark distances
    left_eye_center = face_landmarks.get_landmark(FaceLandmarkIndex.LEFT_EYE_INNER_CORNER)
    right_eye_center = face_landmarks.get_landmark(FaceLandmarkIndex.RIGHT_EYE_INNER_CORNER)
    eye_distance = calculate_distance(left_eye_center, right_eye_center)

    # Eyebrow to eye distances
    left_eyebrow_center = face_landmarks.get_landmark(FaceLandmarkIndex.LEFT_EYEBROW_CENTER)
    right_eyebrow_center = face_landmarks.get_landmark(FaceLandmarkIndex.RIGHT_EYEBROW_CENTER)
    left_eye_top = face_landmarks.get_landmark(FaceLandmarkIndex.LEFT_EYE_CENTER_TOP)
    right_eye_top = face_landmarks.get_landmark(FaceLandmarkIndex.RIGHT_EYE_CENTER_TOP)

    eyebrow_eye_distance_left = calculate_distance(left_eyebrow_center, left_eye_top)
    eyebrow_eye_distance_right = calculate_distance(right_eyebrow_center, right_eye_top)

    # Mouth dimensions
    mouth_left = face_landmarks.get_landmark(FaceLandmarkIndex.MOUTH_LEFT_CORNER)
    mouth_right = face_landmarks.get_landmark(FaceLandmarkIndex.MOUTH_RIGHT_CORNER)
    mouth_top = face_landmarks.get_landmark(FaceLandmarkIndex.MOUTH_UPPER_LIP_TOP_CENTER)
    mouth_bottom = face_landmarks.get_landmark(FaceLandmarkIndex.MOUTH_LOWER_LIP_BOTTOM_CENTER)

    mouth_width = calculate_distance(mouth_left, mouth_right)
    mouth_height = calculate_distance(mouth_top, mouth_bottom)

    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "mar": mar,
        "eye_distance": eye_distance,
        "eyebrow_eye_distance_left": eyebrow_eye_distance_left,
        "eyebrow_eye_distance_right": eyebrow_eye_distance_right,
        "mouth_width": mouth_width,
        "mouth_height": mouth_height,
    }
