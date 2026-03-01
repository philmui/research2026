"""Data structures for video frames and metadata.

This module provides dataclasses for representing video frames and metadata,
which are used throughout the video processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import numpy.typing as npt


@dataclass
class FrameData:
    """Represents a single video frame with associated metadata.

    This class encapsulates all data related to a single frame from a video source,
    including the frame image data, temporal information, and custom metadata.

    Attributes:
        frame: The frame image data as a numpy array (height, width, channels).
        frame_number: The sequential frame number (0-indexed).
        timestamp: The timestamp in seconds from the start of the video.
        metadata: Optional dictionary for storing custom frame-specific metadata.

    Example:
        >>> frame_data = FrameData(
        ...     frame=np.zeros((480, 640, 3), dtype=np.uint8),
        ...     frame_number=42,
        ...     timestamp=1.4,
        ...     metadata={"brightness": 128}
        ... )
    """

    frame: npt.NDArray[np.uint8]
    frame_number: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate frame data after initialization."""
        if self.frame_number < 0:
            raise ValueError(f"frame_number must be non-negative, got {self.frame_number}")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")
        if self.frame.ndim not in (2, 3):
            raise ValueError(
                f"frame must be 2D (grayscale) or 3D (color), got shape {self.frame.shape}"
            )

    @property
    def height(self) -> int:
        """Get the frame height in pixels."""
        return self.frame.shape[0]

    @property
    def width(self) -> int:
        """Get the frame width in pixels."""
        return self.frame.shape[1]

    @property
    def channels(self) -> int:
        """Get the number of color channels (1 for grayscale, 3 for color)."""
        return self.frame.shape[2] if self.frame.ndim == 3 else 1

    @property
    def shape(self) -> tuple[int, ...]:
        """Get the frame shape (height, width, channels)."""
        return self.frame.shape

    def copy(self) -> "FrameData":
        """Create a deep copy of the frame data.

        Returns:
            A new FrameData instance with copied frame array and metadata.
        """
        return FrameData(
            frame=self.frame.copy(),
            frame_number=self.frame_number,
            timestamp=self.timestamp,
            metadata=self.metadata.copy(),
        )


@dataclass
class VideoMetadata:
    """Represents metadata for a video source.

    This class contains all relevant information about a video's properties,
    including dimensions, frame rate, duration, and codec information.

    Attributes:
        fps: Frames per second of the video.
        width: Video frame width in pixels.
        height: Video frame height in pixels.
        total_frames: Total number of frames in the video.
        duration: Total duration in seconds.
        codec: FourCC codec code as a string (e.g., 'XVID', 'MP4V', 'H264').
        additional_info: Optional dictionary for storing extra video properties.

    Example:
        >>> metadata = VideoMetadata(
        ...     fps=30.0,
        ...     width=1920,
        ...     height=1080,
        ...     total_frames=900,
        ...     duration=30.0,
        ...     codec="H264"
        ... )
    """

    fps: float
    width: int
    height: int
    total_frames: int
    duration: float
    codec: str
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate video metadata after initialization."""
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.width <= 0:
            raise ValueError(f"width must be positive, got {self.width}")
        if self.height <= 0:
            raise ValueError(f"height must be positive, got {self.height}")
        if self.total_frames < 0:
            raise ValueError(f"total_frames must be non-negative, got {self.total_frames}")
        if self.duration < 0:
            raise ValueError(f"duration must be non-negative, got {self.duration}")

    @property
    def resolution(self) -> tuple[int, int]:
        """Get the video resolution as (width, height)."""
        return (self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        """Calculate the aspect ratio (width / height)."""
        return self.width / self.height

    @property
    def frame_time(self) -> float:
        """Get the time duration of a single frame in seconds."""
        return 1.0 / self.fps if self.fps > 0 else 0.0

    def __str__(self) -> str:
        """Return a human-readable string representation."""
        return (
            f"VideoMetadata(resolution={self.width}x{self.height}, "
            f"fps={self.fps:.2f}, duration={self.duration:.2f}s, "
            f"frames={self.total_frames}, codec={self.codec})"
        )
