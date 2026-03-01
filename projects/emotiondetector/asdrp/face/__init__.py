"""Face detection and landmark extraction module.

This module provides face detection and facial landmark extraction capabilities
using MediaPipe's Face Landmarker. It includes:

- Base classes and data structures for face landmarks
- MediaPipe-based face detector implementation
- Utility functions for landmark analysis and feature extraction

Example usage:
    >>> from asdrp.face import MediaPipeFaceDetector, extract_geometric_features
    >>> import cv2
    >>>
    >>> # Initialize detector
    >>> detector = MediaPipeFaceDetector("path/to/model.task")
    >>>
    >>> # Load and process image
    >>> image = cv2.imread("face.jpg")
    >>> faces = detector.detect(image)
    >>>
    >>> # Extract features from first detected face
    >>> if faces:
    ...     features = extract_geometric_features(faces[0])
    ...     print(f"Left eye aspect ratio: {features['left_ear']:.3f}")
    ...     print(f"Mouth aspect ratio: {features['mar']:.3f}")
    >>>
    >>> detector.close()
"""

from .base import (
    BaseFaceDetector,
    BoundingBox,
    FaceLandmarkIndex,
    FaceLandmarks,
)
from .detector import MediaPipeFaceDetector
from .landmarker import (
    LANDMARK_GROUPS,
    calculate_angle,
    calculate_distance,
    calculate_eye_aspect_ratio,
    calculate_mouth_aspect_ratio,
    extract_geometric_features,
    get_face_oval,
    get_inter_landmark_distances,
    get_landmark_group,
    get_left_eye,
    get_left_eyebrow,
    get_mouth,
    get_nose,
    get_right_eye,
    get_right_eyebrow,
    normalize_landmarks,
)

__all__ = [
    # Base classes and data structures
    "BaseFaceDetector",
    "BoundingBox",
    "FaceLandmarkIndex",
    "FaceLandmarks",
    # Detector implementation
    "MediaPipeFaceDetector",
    # Landmark utilities
    "LANDMARK_GROUPS",
    "calculate_angle",
    "calculate_distance",
    "calculate_eye_aspect_ratio",
    "calculate_mouth_aspect_ratio",
    "extract_geometric_features",
    "get_face_oval",
    "get_inter_landmark_distances",
    "get_landmark_group",
    "get_left_eye",
    "get_left_eyebrow",
    "get_mouth",
    "get_nose",
    "get_right_eye",
    "get_right_eyebrow",
    "normalize_landmarks",
]
