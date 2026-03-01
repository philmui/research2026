"""MediaPipe-based face detector implementation.

This module provides a concrete implementation of face detection using Google's
MediaPipe Face Landmarker, which detects 478 3D facial landmarks in real-time.
"""

from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .base import BaseFaceDetector, BoundingBox, FaceLandmarks


class MediaPipeFaceDetector(BaseFaceDetector):
    """Face detector using MediaPipe Face Landmarker.

    This detector uses MediaPipe's Face Landmarker to detect faces and extract
    478 3D facial landmarks. It supports both single frame and batch processing,
    with configurable detection parameters.

    Attributes:
        model_path: Path to the MediaPipe Face Landmarker model file (.task)
        min_detection_confidence: Minimum confidence for face detection (0.0 to 1.0)
        min_tracking_confidence: Minimum confidence for face tracking (0.0 to 1.0)
        num_faces: Maximum number of faces to detect
        running_mode: Processing mode ('IMAGE' or 'VIDEO')

    Example:
        >>> detector = MediaPipeFaceDetector(
        ...     model_path="face_landmarker.task",
        ...     num_faces=1
        ... )
        >>> image = cv2.imread("face.jpg")
        >>> faces = detector.detect(image)
        >>> for face in faces:
        ...     print(f"Detected {face.num_landmarks} landmarks")
        >>> detector.close()

        Using context manager:
        >>> with MediaPipeFaceDetector("face_landmarker.task") as detector:
        ...     faces = detector.detect(image)
    """

    def __init__(
        self,
        model_path: str | Path,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        num_faces: int = 1,
        running_mode: str = "IMAGE",
    ) -> None:
        """Initialize the MediaPipe face detector.

        Args:
            model_path: Path to the MediaPipe Face Landmarker model file (.task).
                       Can be downloaded from:
                       https://developers.google.com/mediapipe/solutions/vision/face_landmarker
            min_detection_confidence: Minimum confidence threshold for face detection.
                                     Values range from 0.0 to 1.0. Default is 0.5.
            min_tracking_confidence: Minimum confidence threshold for face tracking
                                    between frames. Values range from 0.0 to 1.0.
                                    Only used in VIDEO mode. Default is 0.5.
            num_faces: Maximum number of faces to detect. Default is 1.
            running_mode: Processing mode, either 'IMAGE' for single frames or
                         'VIDEO' for video sequences with tracking. Default is 'IMAGE'.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If confidence thresholds are not in range [0.0, 1.0] or
                       if running_mode is not 'IMAGE' or 'VIDEO'.
            RuntimeError: If MediaPipe initialization fails.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError(
                f"min_detection_confidence must be in [0.0, 1.0], got {min_detection_confidence}"
            )
        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError(
                f"min_tracking_confidence must be in [0.0, 1.0], got {min_tracking_confidence}"
            )
        if running_mode not in ("IMAGE", "VIDEO"):
            raise ValueError(f"running_mode must be 'IMAGE' or 'VIDEO', got {running_mode}")

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.num_faces = num_faces
        self.running_mode = running_mode

        self._detector: Optional[vision.FaceLandmarker] = None
        self._initialize_detector()

    def _initialize_detector(self) -> None:
        """Initialize the MediaPipe Face Landmarker.

        Raises:
            RuntimeError: If detector initialization fails.
        """
        try:
            # Map string running mode to MediaPipe enum
            mode_map = {
                "IMAGE": vision.RunningMode.IMAGE,
                "VIDEO": vision.RunningMode.VIDEO,
            }

            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mode_map[self.running_mode],
                min_face_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
                num_faces=self.num_faces,
                output_face_blendshapes=False,  # Not needed for emotion detection
                output_facial_transformation_matrixes=False,
            )

            self._detector = vision.FaceLandmarker.create_from_options(options)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize MediaPipe Face Landmarker: {e}") from e

    def _compute_bounding_box(self, landmarks_array: npt.NDArray[np.float32]) -> BoundingBox:
        """Compute bounding box from landmark coordinates.

        Args:
            landmarks_array: Array of shape (N, 3) with normalized landmark coordinates

        Returns:
            BoundingBox containing the face region
        """
        x_coords = landmarks_array[:, 0]
        y_coords = landmarks_array[:, 1]

        x_min = float(np.min(x_coords))
        y_min = float(np.min(y_coords))
        x_max = float(np.max(x_coords))
        y_max = float(np.max(y_coords))

        # Add small padding (5% of size)
        width = x_max - x_min
        height = y_max - y_min
        padding_x = width * 0.05
        padding_y = height * 0.05

        x_min = max(0.0, x_min - padding_x)
        y_min = max(0.0, y_min - padding_y)
        width = min(1.0 - x_min, width + 2 * padding_x)
        height = min(1.0 - y_min, height + 2 * padding_y)

        return BoundingBox(x_min=x_min, y_min=y_min, width=width, height=height)

    def _convert_mediapipe_landmarks(
        self,
        mp_landmarks: list,
        timestamp_ms: float = 0.0,
        frame_number: int = 0,
    ) -> list[FaceLandmarks]:
        """Convert MediaPipe landmarks to FaceLandmarks objects.

        Args:
            mp_landmarks: List of MediaPipe NormalizedLandmark lists
            timestamp_ms: Timestamp in milliseconds
            frame_number: Frame number in sequence

        Returns:
            List of FaceLandmarks objects
        """
        if not mp_landmarks:
            return []

        face_landmarks_list = []

        for face_idx, face_landmarks in enumerate(mp_landmarks):
            # Convert landmarks to numpy array
            landmarks_array = np.array(
                [[lm.x, lm.y, lm.z] for lm in face_landmarks],
                dtype=np.float32,
            )

            # Extract visibility if available
            visibility_array = None
            if hasattr(face_landmarks[0], "visibility"):
                visibility_array = np.array(
                    [lm.visibility for lm in face_landmarks],
                    dtype=np.float32,
                )

            # Compute bounding box
            bounding_box = self._compute_bounding_box(landmarks_array)

            face_landmarks_list.append(
                FaceLandmarks(
                    landmarks=landmarks_array,
                    visibility=visibility_array,
                    bounding_box=bounding_box,
                    timestamp=timestamp_ms,
                    frame_number=frame_number,
                    face_id=face_idx,
                )
            )

        return face_landmarks_list

    def detect(
        self, image: npt.NDArray[np.uint8], timestamp_ms: float = 0.0
    ) -> list[FaceLandmarks]:
        """Detect faces and extract landmarks from a single image frame.

        Args:
            image: Input image as numpy array of shape (height, width, 3) in BGR
                  or RGB format with uint8 dtype.
            timestamp_ms: Timestamp in milliseconds for this frame. Default is 0.0.

        Returns:
            List of FaceLandmarks objects, one per detected face. Empty list if
            no faces detected.

        Raises:
            RuntimeError: If detector is not initialized or detection fails.
            ValueError: If image format is invalid.
        """
        if self._detector is None:
            raise RuntimeError("Detector not initialized. Call _initialize_detector() first.")

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must have shape (H, W, 3), got {image.shape}")

        if image.dtype != np.uint8:
            raise ValueError(f"Image must have dtype uint8, got {image.dtype}")

        try:
            # Convert BGR to RGB if needed (OpenCV uses BGR by default)
            # MediaPipe expects RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Create MediaPipe Image object
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Perform detection
            if self.running_mode == "VIDEO":
                # VIDEO mode requires timestamp in milliseconds as integer
                timestamp_int = int(timestamp_ms)
                detection_result = self._detector.detect_for_video(mp_image, timestamp_int)
            else:
                detection_result = self._detector.detect(mp_image)

            # Convert results
            return self._convert_mediapipe_landmarks(
                detection_result.face_landmarks,
                timestamp_ms=timestamp_ms,
                frame_number=0,
            )

        except Exception as e:
            raise RuntimeError(f"Face detection failed: {e}") from e

    def detect_batch(
        self, images: list[npt.NDArray[np.uint8]], timestamps_ms: Optional[list[float]] = None
    ) -> list[list[FaceLandmarks]]:
        """Detect faces and extract landmarks from multiple image frames.

        Args:
            images: List of input images, each as numpy array of shape (H, W, 3)
                   in BGR or RGB format with uint8 dtype.
            timestamps_ms: Optional list of timestamps in milliseconds, one per frame.
                          If None, defaults to sequential indices (0, 1, 2, ...).

        Returns:
            List of lists of FaceLandmarks. Each inner list contains the detected
            faces for the corresponding input image.

        Raises:
            ValueError: If timestamps_ms length doesn't match images length.
            RuntimeError: If detector is not initialized or detection fails.
        """
        if self._detector is None:
            raise RuntimeError("Detector not initialized.")

        if not images:
            return []

        if timestamps_ms is None:
            timestamps_ms = [float(i) for i in range(len(images))]
        elif len(timestamps_ms) != len(images):
            raise ValueError(
                f"timestamps_ms length ({len(timestamps_ms)}) must match "
                f"images length ({len(images)})"
            )

        results = []
        for frame_idx, (image, timestamp) in enumerate(zip(images, timestamps_ms)):
            try:
                face_landmarks = self.detect(image, timestamp_ms=timestamp)
                # Update frame numbers
                for face in face_landmarks:
                    face.frame_number = frame_idx
                results.append(face_landmarks)
            except Exception as e:
                # Log warning but continue processing other frames
                print(f"Warning: Detection failed for frame {frame_idx}: {e}")
                results.append([])

        return results

    def close(self) -> None:
        """Release resources held by the detector.

        Closes the MediaPipe Face Landmarker and releases associated resources.
        After calling close(), the detector cannot be used for further detections
        unless reinitialized.
        """
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        self.close()

    def __repr__(self) -> str:
        """String representation of the detector."""
        return (
            f"MediaPipeFaceDetector("
            f"model_path={self.model_path}, "
            f"num_faces={self.num_faces}, "
            f"mode={self.running_mode})"
        )
