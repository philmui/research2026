"""Video reading and frame extraction classes.

This module provides classes for reading video files and extracting frames
using OpenCV. It supports iteration, seeking, and context management.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

from .frame import FrameData, VideoMetadata


class VideoReaderError(Exception):
    """Base exception for video reader errors."""

    pass


class VideoReader(ABC):
    """Abstract base class for video readers.

    This class defines the interface that all video readers must implement,
    providing a consistent API for reading frames from various video sources.
    """

    @abstractmethod
    def read_frame(self) -> Optional[FrameData]:
        """Read the next frame from the video.

        Returns:
            FrameData if a frame was successfully read, None if end of video.

        Raises:
            VideoReaderError: If there's an error reading the frame.
        """
        pass

    @abstractmethod
    def get_frame_at(self, frame_number: int) -> Optional[FrameData]:
        """Get a specific frame by frame number.

        Args:
            frame_number: The frame number to retrieve (0-indexed).

        Returns:
            FrameData if the frame exists, None otherwise.

        Raises:
            VideoReaderError: If there's an error accessing the frame.
        """
        pass

    @abstractmethod
    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame position.

        Args:
            frame_number: The frame number to seek to (0-indexed).

        Returns:
            True if seek was successful, False otherwise.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> VideoMetadata:
        """Get metadata about the video.

        Returns:
            VideoMetadata containing video properties.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the video reader to the beginning."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the video reader and release resources."""
        pass


class VideoFileReader(VideoReader):
    """Reads frames from a video file using OpenCV.

    This class provides comprehensive functionality for reading video files,
    including iteration support, seeking, and context management.

    Attributes:
        file_path: Path to the video file.
        current_frame_number: Current position in the video (0-indexed).

    Example:
        >>> # Using context manager
        >>> with VideoFileReader("video.mp4") as reader:
        ...     for frame_data in reader:
        ...         # Process frame
        ...         pass
        >>>
        >>> # Manual usage
        >>> reader = VideoFileReader("video.mp4")
        >>> metadata = reader.get_metadata()
        >>> frame = reader.read_frame()
        >>> reader.close()
    """

    def __init__(self, file_path: str | Path) -> None:
        """Initialize the video file reader.

        Args:
            file_path: Path to the video file to read.

        Raises:
            VideoReaderError: If the file doesn't exist or can't be opened.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise VideoReaderError(f"Video file not found: {self.file_path}")

        self._capture: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self.current_frame_number: int = 0
        self._is_open: bool = False

        self._open()

    def _open(self) -> None:
        """Open the video file and initialize metadata."""
        self._capture = cv2.VideoCapture(str(self.file_path))

        if not self._capture.isOpened():
            raise VideoReaderError(f"Failed to open video file: {self.file_path}")

        self._is_open = True
        self._initialize_metadata()

    def _initialize_metadata(self) -> None:
        """Extract and store video metadata."""
        if self._capture is None:
            raise VideoReaderError("Video capture not initialized")

        fps = self._capture.get(cv2.CAP_PROP_FPS)
        width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self._capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0.0

        # Get codec information
        fourcc = int(self._capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])

        self._metadata = VideoMetadata(
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration=duration,
            codec=codec,
            additional_info={"file_path": str(self.file_path)},
        )

    def read_frame(self) -> Optional[FrameData]:
        """Read the next frame from the video.

        Returns:
            FrameData if a frame was successfully read, None if end of video.

        Raises:
            VideoReaderError: If the reader is closed or there's a read error.
        """
        if not self._is_open or self._capture is None:
            raise VideoReaderError("Video reader is not open")

        ret, frame = self._capture.read()

        if not ret:
            return None

        timestamp = self.current_frame_number / self._metadata.fps if self._metadata else 0.0

        frame_data = FrameData(
            frame=frame,
            frame_number=self.current_frame_number,
            timestamp=timestamp,
            metadata={"source": str(self.file_path)},
        )

        self.current_frame_number += 1
        return frame_data

    def get_frame_at(self, frame_number: int) -> Optional[FrameData]:
        """Get a specific frame by frame number.

        Args:
            frame_number: The frame number to retrieve (0-indexed).

        Returns:
            FrameData if the frame exists, None otherwise.

        Raises:
            VideoReaderError: If the reader is closed or frame_number is invalid.
        """
        if not self._is_open or self._capture is None:
            raise VideoReaderError("Video reader is not open")

        if frame_number < 0:
            raise VideoReaderError(f"Frame number must be non-negative, got {frame_number}")

        if self._metadata and frame_number >= self._metadata.total_frames:
            return None

        # Seek to the desired frame
        if not self.seek(frame_number):
            return None

        # Read the frame
        return self.read_frame()

    def seek(self, frame_number: int) -> bool:
        """Seek to a specific frame position.

        Args:
            frame_number: The frame number to seek to (0-indexed).

        Returns:
            True if seek was successful, False otherwise.

        Raises:
            VideoReaderError: If the reader is closed.
        """
        if not self._is_open or self._capture is None:
            raise VideoReaderError("Video reader is not open")

        if frame_number < 0:
            return False

        success = self._capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        if success:
            self.current_frame_number = frame_number

        return success

    def get_metadata(self) -> VideoMetadata:
        """Get metadata about the video.

        Returns:
            VideoMetadata containing video properties.

        Raises:
            VideoReaderError: If metadata is not available.
        """
        if self._metadata is None:
            raise VideoReaderError("Video metadata not available")

        return self._metadata

    def reset(self) -> None:
        """Reset the video reader to the beginning.

        Raises:
            VideoReaderError: If the reader is closed.
        """
        if not self._is_open or self._capture is None:
            raise VideoReaderError("Video reader is not open")

        self.seek(0)

    def close(self) -> None:
        """Close the video reader and release resources."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None

        self._is_open = False
        self.current_frame_number = 0

    # Iterator protocol
    def __iter__(self) -> Iterator[FrameData]:
        """Make the reader iterable.

        Returns:
            Self as an iterator.

        Example:
            >>> with VideoFileReader("video.mp4") as reader:
            ...     for frame in reader:
            ...         print(f"Frame {frame.frame_number}")
        """
        self.reset()
        return self

    def __next__(self) -> FrameData:
        """Get the next frame in iteration.

        Returns:
            The next FrameData.

        Raises:
            StopIteration: When the end of the video is reached.
        """
        frame = self.read_frame()
        if frame is None:
            raise StopIteration
        return frame

    # Context manager protocol
    def __enter__(self) -> "VideoFileReader":
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
        """Return a string representation of the reader."""
        status = "open" if self._is_open else "closed"
        return f"VideoFileReader(file='{self.file_path}', status='{status}', frame={self.current_frame_number})"

    def __len__(self) -> int:
        """Get the total number of frames in the video.

        Returns:
            Total number of frames.

        Raises:
            VideoReaderError: If metadata is not available.
        """
        if self._metadata is None:
            raise VideoReaderError("Video metadata not available")

        return self._metadata.total_frames
