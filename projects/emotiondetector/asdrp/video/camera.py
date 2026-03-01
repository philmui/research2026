"""Camera capture classes for real-time video input.

This module provides classes for capturing frames from webcams and other
camera devices in real-time using OpenCV.
"""

from typing import Iterator, Optional

import cv2

from .frame import FrameData, VideoMetadata


class CameraCaptureError(Exception):
    """Base exception for camera capture errors."""

    pass


class CameraCapture:
    """Captures frames from a camera device in real-time.

    This class provides functionality for accessing webcams and other camera
    devices, with support for iteration and context management.

    Attributes:
        camera_id: The camera device ID (0 for default camera).
        fps: Target frames per second for capture.
        width: Frame width in pixels (if set).
        height: Frame height in pixels (if set).

    Example:
        >>> # Using context manager
        >>> with CameraCapture(camera_id=0) as camera:
        ...     for i, frame in enumerate(camera):
        ...         # Process frame
        ...         if i >= 100:  # Capture 100 frames
        ...             break
        >>>
        >>> # Manual usage with custom resolution
        >>> camera = CameraCapture(camera_id=0, width=1280, height=720)
        >>> frame = camera.read_frame()
        >>> camera.close()
    """

    def __init__(
        self,
        camera_id: int = 0,
        fps: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        backend: Optional[int] = None,
    ) -> None:
        """Initialize the camera capture.

        Args:
            camera_id: The camera device ID (0 for default camera).
            fps: Target frames per second (None to use camera default).
            width: Target frame width in pixels (None to use camera default).
            height: Target frame height in pixels (None to use camera default).
            backend: OpenCV capture backend (e.g., cv2.CAP_DSHOW, cv2.CAP_AVFOUNDATION).
                     None to use default backend.

        Raises:
            CameraCaptureError: If the camera cannot be opened.
        """
        self.camera_id = camera_id
        self.target_fps = fps
        self.target_width = width
        self.target_height = height
        self.backend = backend

        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._frame_count: int = 0
        self._is_open: bool = False

        self._open()

    def _open(self) -> None:
        """Open the camera device and configure settings."""
        # Open camera with or without backend specification
        if self.backend is not None:
            self._capture = cv2.VideoCapture(self.camera_id, self.backend)
        else:
            self._capture = cv2.VideoCapture(self.camera_id)

        if not self._capture.isOpened():
            raise CameraCaptureError(f"Failed to open camera with ID: {self.camera_id}")

        self._is_open = True

        # Set camera properties if specified
        if self.target_width is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)

        if self.target_height is not None:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        if self.target_fps is not None:
            self._capture.set(cv2.CAP_PROP_FPS, self.target_fps)

        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Extract and store camera metadata."""
        if self._capture is None:
            raise CameraCaptureError("Camera capture not initialized")

        # Get actual camera properties (may differ from requested)
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Get codec information
        fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        self._metadata = VideoMetadata(
            fps=fps if fps > 0 else 30.0,  # Default to 30 fps if not available
            width=width,
            height=height,
            total_frames=0,  # Live stream has no fixed total
            duration=0.0,  # Live stream has no fixed duration
            codec=codec,
            additional_info={
                "camera_id": self.camera_id,
                "backend": self.backend,
                "is_live": True,
            },
        )

    def read_frame(self) -> Optional[FrameData]:
        """Read the next frame from the camera.

        Returns:
            FrameData if a frame was successfully captured, None otherwise.

        Raises:
            CameraCaptureError: If the camera is not open or there's a capture error.
        """
        if not self._is_open or self._capture is None:
            raise CameraCaptureError("Camera is not open")

        ret, frame = self._capture.read()

        if not ret:
            return None

        # Calculate timestamp based on frame count and FPS
        timestamp = (
            self._frame_count / self._metadata.fps
            if self._metadata and self._metadata.fps > 0
            else 0.0
        )

        frame_data = FrameData(
            frame=frame,
            frame_number=self._frame_count,
            timestamp=timestamp,
            metadata={"camera_id": self.camera_id, "is_live": True},
        )

        self._frame_count += 1
        return frame_data

    def get_metadata(self) -> VideoMetadata:
        """Get metadata about the camera.

        Returns:
            VideoMetadata containing camera properties.

        Raises:
            CameraCaptureError: If metadata is not available.
        """
        if self._metadata is None:
            raise CameraCaptureError("Camera metadata not available")

        return self._metadata

    def is_open(self) -> bool:
        """Check if the camera is open and ready to capture.

        Returns:
            True if the camera is open, False otherwise.
        """
        return self._is_open

    def get_frame_count(self) -> int:
        """Get the number of frames captured so far.

        Returns:
            Number of frames captured.
        """
        return self._frame_count

    def set_property(self, property_id: int, value: float) -> bool:
        """Set a camera property.

        Args:
            property_id: OpenCV property ID (e.g., cv2.CAP_PROP_BRIGHTNESS).
            value: The value to set.

        Returns:
            True if the property was set successfully, False otherwise.

        Raises:
            CameraCaptureError: If the camera is not open.
        """
        if not self._is_open or self._capture is None:
            raise CameraCaptureError("Camera is not open")

        return self._capture.set(property_id, value)

    def get_property(self, property_id: int) -> float:
        """Get a camera property.

        Args:
            property_id: OpenCV property ID (e.g., cv2.CAP_PROP_BRIGHTNESS).

        Returns:
            The current value of the property.

        Raises:
            CameraCaptureError: If the camera is not open.
        """
        if not self._is_open or self._capture is None:
            raise CameraCaptureError("Camera is not open")

        return self._capture.get(property_id)

    def close(self) -> None:
        """Close the camera and release resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._is_open = False

    # Iterator protocol
    def __iter__(self) -> Iterator[FrameData]:
        """Make the camera capture iterable.

        Returns:
            Self as an iterator.

        Example:
            >>> with CameraCapture(0) as camera:
            ...     for frame in camera:
            ...         print(f"Frame {frame.frame_number}")
            ...         if frame.frame_number >= 10:
            ...             break
        """
        return self

    def __next__(self) -> FrameData:
        """Get the next frame in iteration.

        Returns:
            The next FrameData.

        Raises:
            StopIteration: If the camera is closed or frame capture fails.
        """
        frame = self.read_frame()
        if frame is None:
            raise StopIteration
        return frame

    # Context manager protocol
    def __enter__(self) -> "CameraCapture":
        """Enter context manager.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """Exit context manager and clean up resources.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self.close()

    def __repr__(self) -> str:
        """Return a string representation of the camera capture."""
        status = "open" if self._is_open else "closed"
        return (
            f"CameraCapture(camera_id={self.camera_id}, "
            f"status='{status}', frames_captured={self._frame_count})"
        )

    def __del__(self) -> None:
        """Destructor to ensure camera is released."""
        if self._is_open:
            self.close()


class MultiCameraCapture:
    """Captures frames from multiple cameras simultaneously.

    This class manages multiple camera captures and provides synchronized
    frame access across all cameras.

    Example:
        >>> cameras = MultiCameraCapture([0, 1])  # Two cameras
        >>> frames = cameras.read_all_frames()
        >>> if frames:
        ...     frame_cam0, frame_cam1 = frames
        >>> cameras.close()
    """

    def __init__(self, camera_ids: list[int], **kwargs: object) -> None:
        """Initialize multiple camera captures.

        Args:
            camera_ids: List of camera device IDs to open.
            **kwargs: Additional arguments passed to each CameraCapture.

        Raises:
            CameraCaptureError: If any camera cannot be opened.
        """
        self.camera_ids = camera_ids
        self.cameras: list[CameraCapture] = []

        # Open all cameras
        for camera_id in camera_ids:
            try:
                camera = CameraCapture(camera_id=camera_id, **kwargs)  # type: ignore
                self.cameras.append(camera)
            except CameraCaptureError as e:
                # Close any already opened cameras
                self.close()
                raise CameraCaptureError(
                    f"Failed to open camera {camera_id}: {e}"
                ) from e

    def read_all_frames(self) -> Optional[list[FrameData]]:
        """Read one frame from each camera.

        Returns:
            List of FrameData objects, one per camera, or None if any camera fails.
        """
        frames = []
        for camera in self.cameras:
            frame = camera.read_frame()
            if frame is None:
                return None
            frames.append(frame)
        return frames

    def close(self) -> None:
        """Close all cameras and release resources."""
        for camera in self.cameras:
            camera.close()
        self.cameras.clear()

    def __enter__(self) -> "MultiCameraCapture":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """Exit context manager and clean up resources."""
        self.close()

    def __len__(self) -> int:
        """Get the number of cameras."""
        return len(self.cameras)

    def __getitem__(self, index: int) -> CameraCapture:
        """Get a specific camera by index."""
        return self.cameras[index]
