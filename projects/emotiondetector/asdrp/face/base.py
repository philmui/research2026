"""Base classes and data structures for face detection and landmark extraction.

This module provides the foundational classes and enums for face detection,
including dataclasses for landmarks and bounding boxes, and an abstract base
class for face detectors.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np
import numpy.typing as npt


class FaceLandmarkIndex(IntEnum):
    """MediaPipe Face Landmarker indices organized by facial region.

    MediaPipe Face Landmarker provides 478 3D face landmarks that follow the
    face geometry defined by the Canonical Face Model. This enum organizes
    the key landmarks by facial region for easier access.

    Reference:
        https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
    """

    # Left Eye (8 landmarks around the eye contour)
    LEFT_EYE_OUTER_CORNER = 33
    LEFT_EYE_INNER_CORNER = 133
    LEFT_EYE_TOP_UPPER = 159
    LEFT_EYE_TOP_LOWER = 145
    LEFT_EYE_BOTTOM_LOWER = 153
    LEFT_EYE_BOTTOM_UPPER = 154
    LEFT_EYE_CENTER_TOP = 157
    LEFT_EYE_CENTER_BOTTOM = 144

    # Right Eye (8 landmarks around the eye contour)
    RIGHT_EYE_OUTER_CORNER = 362
    RIGHT_EYE_INNER_CORNER = 263
    RIGHT_EYE_TOP_UPPER = 386
    RIGHT_EYE_TOP_LOWER = 374
    RIGHT_EYE_BOTTOM_LOWER = 380
    RIGHT_EYE_BOTTOM_UPPER = 381
    RIGHT_EYE_CENTER_TOP = 384
    RIGHT_EYE_CENTER_BOTTOM = 373

    # Left Eyebrow (5 landmarks)
    LEFT_EYEBROW_INNER = 70
    LEFT_EYEBROW_CENTER = 63
    LEFT_EYEBROW_OUTER = 105
    LEFT_EYEBROW_UPPER_INNER = 52
    LEFT_EYEBROW_UPPER_OUTER = 107

    # Right Eyebrow (5 landmarks)
    RIGHT_EYEBROW_INNER = 300
    RIGHT_EYEBROW_CENTER = 293
    RIGHT_EYEBROW_OUTER = 334
    RIGHT_EYEBROW_UPPER_INNER = 282
    RIGHT_EYEBROW_UPPER_OUTER = 336

    # Nose (7 landmarks)
    NOSE_TIP = 1
    NOSE_BRIDGE_TOP = 6
    NOSE_BRIDGE_CENTER = 168
    NOSE_LEFT_NOSTRIL = 98
    NOSE_RIGHT_NOSTRIL = 327
    NOSE_LEFT_ALAR = 129
    NOSE_RIGHT_ALAR = 358

    # Mouth Outer (12 landmarks around outer lip contour)
    MOUTH_LEFT_CORNER = 61
    MOUTH_RIGHT_CORNER = 291
    MOUTH_UPPER_LIP_TOP_LEFT = 185
    MOUTH_UPPER_LIP_TOP_CENTER = 40
    MOUTH_UPPER_LIP_TOP_RIGHT = 409
    MOUTH_LOWER_LIP_BOTTOM_LEFT = 146
    MOUTH_LOWER_LIP_BOTTOM_CENTER = 17
    MOUTH_LOWER_LIP_BOTTOM_RIGHT = 375
    MOUTH_UPPER_LIP_BOTTOM_LEFT = 78
    MOUTH_UPPER_LIP_BOTTOM_CENTER = 13
    MOUTH_UPPER_LIP_BOTTOM_RIGHT = 308
    MOUTH_LOWER_LIP_TOP_CENTER = 14

    # Mouth Inner (8 landmarks for inner lip contour)
    MOUTH_INNER_UPPER_LEFT = 78
    MOUTH_INNER_UPPER_CENTER = 13
    MOUTH_INNER_UPPER_RIGHT = 308
    MOUTH_INNER_LOWER_LEFT = 95
    MOUTH_INNER_LOWER_CENTER = 14
    MOUTH_INNER_LOWER_RIGHT = 324

    # Face Oval (key points defining the face boundary)
    FACE_OVAL_LEFT_TOP = 234
    FACE_OVAL_LEFT_MIDDLE = 127
    FACE_OVAL_LEFT_BOTTOM = 162
    FACE_OVAL_CHIN_LEFT = 172
    FACE_OVAL_CHIN_CENTER = 152
    FACE_OVAL_CHIN_RIGHT = 397
    FACE_OVAL_RIGHT_BOTTOM = 389
    FACE_OVAL_RIGHT_MIDDLE = 356
    FACE_OVAL_RIGHT_TOP = 454
    FACE_OVAL_FOREHEAD_LEFT = 10
    FACE_OVAL_FOREHEAD_CENTER = 8
    FACE_OVAL_FOREHEAD_RIGHT = 297


@dataclass
class BoundingBox:
    """Bounding box for a detected face.

    Represents a rectangular region containing a detected face, with normalized
    coordinates relative to the image dimensions.

    Attributes:
        x_min: Normalized x-coordinate of the top-left corner (0.0 to 1.0)
        y_min: Normalized y-coordinate of the top-left corner (0.0 to 1.0)
        width: Normalized width of the bounding box (0.0 to 1.0)
        height: Normalized height of the bounding box (0.0 to 1.0)
    """

    x_min: float
    y_min: float
    width: float
    height: float

    def to_absolute(self, image_width: int, image_height: int) -> tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute pixel coordinates.

        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in absolute pixel coordinates
        """
        x_min_abs = int(self.x_min * image_width)
        y_min_abs = int(self.y_min * image_height)
        x_max_abs = int((self.x_min + self.width) * image_width)
        y_max_abs = int((self.y_min + self.height) * image_height)
        return x_min_abs, y_min_abs, x_max_abs, y_max_abs

    @property
    def x_max(self) -> float:
        """Get normalized x-coordinate of the bottom-right corner."""
        return self.x_min + self.width

    @property
    def y_max(self) -> float:
        """Get normalized y-coordinate of the bottom-right corner."""
        return self.y_min + self.height

    @property
    def center(self) -> tuple[float, float]:
        """Get normalized center coordinates of the bounding box."""
        return (self.x_min + self.width / 2, self.y_min + self.height / 2)

    @property
    def area(self) -> float:
        """Get normalized area of the bounding box."""
        return self.width * self.height


@dataclass
class FaceLandmarks:
    """Face landmarks detected from a single frame.

    Contains the 3D coordinates of facial landmarks detected by MediaPipe Face
    Landmarker, along with metadata about the detection.

    Attributes:
        landmarks: Array of shape (N, 3) containing (x, y, z) coordinates for N landmarks.
                  Coordinates are normalized: x and y are in [0, 1] relative to image
                  dimensions, z represents depth relative to face center.
        visibility: Array of shape (N,) containing visibility scores for each landmark,
                   or None if not provided by the detector. Values typically in [0, 1]
                   where higher values indicate higher confidence.
        bounding_box: Bounding box around the detected face, or None if not computed.
        timestamp: Timestamp in milliseconds of the frame from which landmarks were detected.
        frame_number: Sequential frame number in the video, or 0 for standalone images.
        face_id: Identifier for tracking the same face across multiple frames, or 0 if
                tracking is not enabled.
    """

    landmarks: npt.NDArray[np.float32]
    visibility: Optional[npt.NDArray[np.float32]] = None
    bounding_box: Optional[BoundingBox] = None
    timestamp: float = 0.0
    frame_number: int = 0
    face_id: int = 0

    def __post_init__(self) -> None:
        """Validate landmark array shape after initialization."""
        if self.landmarks.ndim != 2 or self.landmarks.shape[1] != 3:
            raise ValueError(
                f"landmarks must be of shape (N, 3), got {self.landmarks.shape}"
            )
        if self.visibility is not None:
            if self.visibility.ndim != 1:
                raise ValueError(
                    f"visibility must be 1-dimensional, got shape {self.visibility.shape}"
                )
            if len(self.visibility) != len(self.landmarks):
                raise ValueError(
                    f"visibility length ({len(self.visibility)}) must match "
                    f"landmarks length ({len(self.landmarks)})"
                )

    @property
    def num_landmarks(self) -> int:
        """Get the number of detected landmarks."""
        return len(self.landmarks)

    def get_landmark(self, index: int | FaceLandmarkIndex) -> npt.NDArray[np.float32]:
        """Get a specific landmark by index.

        Args:
            index: Landmark index (integer or FaceLandmarkIndex enum member)

        Returns:
            Array of shape (3,) containing (x, y, z) coordinates

        Raises:
            IndexError: If index is out of bounds
        """
        if isinstance(index, FaceLandmarkIndex):
            index = index.value
        return self.landmarks[index]

    def to_absolute(
        self, image_width: int, image_height: int
    ) -> npt.NDArray[np.float32]:
        """Convert normalized landmark coordinates to absolute pixel coordinates.

        Args:
            image_width: Width of the image in pixels
            image_height: Height of the image in pixels

        Returns:
            Array of shape (N, 3) with absolute coordinates. The z-coordinate is
            scaled by the image width for consistency.
        """
        absolute_landmarks = self.landmarks.copy()
        absolute_landmarks[:, 0] *= image_width
        absolute_landmarks[:, 1] *= image_height
        absolute_landmarks[:, 2] *= image_width  # Scale z by width as per MediaPipe convention
        return absolute_landmarks


class BaseFaceDetector(ABC):
    """Abstract base class for face detectors.

    Defines the interface that all face detection implementations must follow.
    Implementations should handle initialization of the underlying detection model,
    processing of single frames and batches, and proper resource cleanup.
    """

    @abstractmethod
    def detect(
        self, image: npt.NDArray[np.uint8], timestamp_ms: float = 0.0
    ) -> list[FaceLandmarks]:
        """Detect faces and extract landmarks from a single image frame.

        Args:
            image: Input image as a numpy array of shape (height, width, 3) with
                  BGR or RGB color channels and uint8 dtype.
            timestamp_ms: Timestamp in milliseconds associated with this frame.
                         Used for tracking and temporal analysis.

        Returns:
            List of FaceLandmarks objects, one for each detected face. Returns
            empty list if no faces are detected.

        Raises:
            RuntimeError: If the detector has not been properly initialized or
                         if detection fails.
        """
        pass

    @abstractmethod
    def detect_batch(
        self, images: list[npt.NDArray[np.uint8]], timestamps_ms: Optional[list[float]] = None
    ) -> list[list[FaceLandmarks]]:
        """Detect faces and extract landmarks from multiple image frames.

        Args:
            images: List of input images, each as a numpy array of shape
                   (height, width, 3) with BGR or RGB color channels and uint8 dtype.
            timestamps_ms: Optional list of timestamps in milliseconds for each frame.
                          If None, timestamps default to 0.0 for all frames.

        Returns:
            List of lists of FaceLandmarks objects. Each inner list contains the
            landmarks for all faces detected in the corresponding input image.

        Raises:
            ValueError: If timestamps_ms is provided but has different length than images.
            RuntimeError: If the detector has not been properly initialized or
                         if detection fails.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Release resources held by the detector.

        Should be called when the detector is no longer needed to free up
        memory and computational resources. After calling close(), the detector
        should not be used for further detections.
        """
        pass

    def __enter__(self) -> "BaseFaceDetector":
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """Context manager exit point. Ensures proper cleanup."""
        self.close()
