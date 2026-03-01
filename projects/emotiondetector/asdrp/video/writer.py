"""Video writing classes for creating video files.

This module provides classes for writing video files using OpenCV,
with support for various codecs and output formats.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from .frame import FrameData, VideoMetadata


class VideoWriterError(Exception):
    """Base exception for video writer errors."""

    pass


class VideoWriter(ABC):
    """Abstract base class for video writers.

    This class defines the interface that all video writers must implement,
    providing a consistent API for writing frames to various video outputs.
    """

    @abstractmethod
    def write_frame(self, frame: npt.NDArray[np.uint8] | FrameData) -> None:
        """Write a frame to the video output.

        Args:
            frame: Frame data as numpy array or FrameData object.

        Raises:
            VideoWriterError: If there's an error writing the frame.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the video writer and finalize the output."""
        pass

    @abstractmethod
    def is_open(self) -> bool:
        """Check if the writer is open and ready to write.

        Returns:
            True if the writer is open, False otherwise.
        """
        pass


class VideoFileWriter(VideoWriter):
    """Writes frames to a video file using OpenCV.

    This class provides functionality for creating video files with various
    codecs and formats, with support for context management.

    Attributes:
        file_path: Path to the output video file.
        fps: Frames per second for the output video.
        frame_size: Size of the video frames (width, height).
        codec: FourCC codec code.

    Example:
        >>> # Using context manager
        >>> with VideoFileWriter("output.mp4", fps=30.0, frame_size=(640, 480)) as writer:
        ...     for frame in frames:
        ...         writer.write_frame(frame)
        >>>
        >>> # Manual usage
        >>> writer = VideoFileWriter("output.mp4", fps=30.0, frame_size=(640, 480))
        >>> writer.write_frame(frame_array)
        >>> writer.close()
    """

    # Common codec mappings
    CODECS = {
        "mp4": "mp4v",
        "avi": "XVID",
        "mov": "mp4v",
        "mkv": "X264",
    }

    def __init__(
        self,
        file_path: str | Path,
        fps: float,
        frame_size: tuple[int, int],
        codec: Optional[str] = None,
        is_color: bool = True,
    ) -> None:
        """Initialize the video file writer.

        Args:
            file_path: Path to the output video file.
            fps: Frames per second for the output video.
            frame_size: Size of the video frames as (width, height).
            codec: FourCC codec code (e.g., 'XVID', 'mp4v', 'H264').
                   If None, automatically selected based on file extension.
            is_color: Whether the video is color (True) or grayscale (False).

        Raises:
            VideoWriterError: If the parameters are invalid or writer can't be initialized.
        """
        self.file_path = Path(file_path)
        self.fps = fps
        self.frame_size = frame_size
        self.is_color = is_color

        if self.fps <= 0:
            raise VideoWriterError(f"FPS must be positive, got {self.fps}")

        if self.frame_size[0] <= 0 or self.frame_size[1] <= 0:
            raise VideoWriterError(f"Invalid frame size: {self.frame_size}")

        # Determine codec
        if codec is None:
            codec = self._get_default_codec()
        self.codec = codec

        self._writer: Optional[cv2.VideoWriter] = None
        self._is_open: bool = False
        self._frame_count: int = 0

        self._open()

    def _get_default_codec(self) -> str:
        """Get the default codec based on file extension.

        Returns:
            FourCC codec code as a string.
        """
        extension = self.file_path.suffix.lower().lstrip(".")
        return self.CODECS.get(extension, "mp4v")

    def _open(self) -> None:
        """Open the video writer and initialize output."""
        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert codec string to FourCC integer
        fourcc = cv2.VideoWriter_fourcc(*self.codec)

        # Create video writer
        self._writer = cv2.VideoWriter(
            str(self.file_path), fourcc, self.fps, self.frame_size, self.is_color
        )

        if not self._writer.isOpened():
            raise VideoWriterError(
                f"Failed to open video writer for file: {self.file_path} "
                f"with codec: {self.codec}"
            )

        self._is_open = True

    def write_frame(self, frame: npt.NDArray[np.uint8] | FrameData) -> None:
        """Write a frame to the video output.

        Args:
            frame: Frame data as numpy array or FrameData object.

        Raises:
            VideoWriterError: If the writer is closed or there's a write error.
        """
        if not self._is_open or self._writer is None:
            raise VideoWriterError("Video writer is not open")

        # Extract frame array if FrameData is provided
        frame_array = frame.frame if isinstance(frame, FrameData) else frame

        # Validate frame dimensions
        frame_height, frame_width = frame_array.shape[:2]
        expected_width, expected_height = self.frame_size

        if frame_width != expected_width or frame_height != expected_height:
            raise VideoWriterError(
                f"Frame size mismatch: expected {self.frame_size}, "
                f"got ({frame_width}, {frame_height})"
            )

        # Validate color channels
        is_frame_color = frame_array.ndim == 3 and frame_array.shape[2] == 3
        if is_frame_color != self.is_color:
            if self.is_color:
                raise VideoWriterError(
                    "Expected color frame (3 channels), got grayscale or wrong format"
                )
            else:
                raise VideoWriterError("Expected grayscale frame, got color or wrong format")

        # Write frame
        self._writer.write(frame_array)
        self._frame_count += 1

    def close(self) -> None:
        """Close the video writer and finalize the output."""
        if self._writer is not None:
            self._writer.release()
            self._writer = None

        self._is_open = False

    def is_open(self) -> bool:
        """Check if the writer is open and ready to write.

        Returns:
            True if the writer is open, False otherwise.
        """
        return self._is_open

    def get_frame_count(self) -> int:
        """Get the number of frames written so far.

        Returns:
            Number of frames written.
        """
        return self._frame_count

    def get_metadata(self) -> VideoMetadata:
        """Get metadata about the video being written.

        Returns:
            VideoMetadata containing video properties.
        """
        duration = self._frame_count / self.fps if self.fps > 0 else 0.0

        return VideoMetadata(
            fps=self.fps,
            width=self.frame_size[0],
            height=self.frame_size[1],
            total_frames=self._frame_count,
            duration=duration,
            codec=self.codec,
            additional_info={"file_path": str(self.file_path), "is_color": self.is_color},
        )

    # Context manager protocol
    def __enter__(self) -> "VideoFileWriter":
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
        """Return a string representation of the writer."""
        status = "open" if self._is_open else "closed"
        return (
            f"VideoFileWriter(file='{self.file_path}', "
            f"fps={self.fps}, size={self.frame_size}, "
            f"codec='{self.codec}', status='{status}', frames={self._frame_count})"
        )

    def __del__(self) -> None:
        """Destructor to ensure writer is closed."""
        if self._is_open:
            self.close()


class VideoFileWriterFromReader(VideoFileWriter):
    """Creates a video writer with settings matching a video reader.

    This convenience class automatically configures the writer based on
    metadata from an existing video reader, making it easy to process
    and save videos with the same properties.

    Example:
        >>> with VideoFileReader("input.mp4") as reader:
        ...     with VideoFileWriterFromReader("output.mp4", reader) as writer:
        ...         for frame in reader:
        ...             # Process frame
        ...             processed_frame = process(frame.frame)
        ...             writer.write_frame(processed_frame)
    """

    def __init__(
        self,
        file_path: str | Path,
        source_reader: "VideoReader",
        codec: Optional[str] = None,
        fps: Optional[float] = None,
        frame_size: Optional[tuple[int, int]] = None,
    ) -> None:
        """Initialize writer from a video reader's metadata.

        Args:
            file_path: Path to the output video file.
            source_reader: VideoReader to copy settings from.
            codec: Optional codec override (uses source codec if None).
            fps: Optional FPS override (uses source FPS if None).
            frame_size: Optional frame size override (uses source size if None).

        Raises:
            VideoWriterError: If the reader metadata is invalid.
        """
        from .reader import VideoReader

        if not isinstance(source_reader, VideoReader):
            raise VideoWriterError("source_reader must be a VideoReader instance")

        metadata = source_reader.get_metadata()

        # Use provided values or fall back to source metadata
        final_fps = fps if fps is not None else metadata.fps
        final_size = (
            frame_size if frame_size is not None else (metadata.width, metadata.height)
        )
        final_codec = codec if codec is not None else metadata.codec

        super().__init__(
            file_path=file_path,
            fps=final_fps,
            frame_size=final_size,
            codec=final_codec,
            is_color=True,  # Assume color by default
        )
