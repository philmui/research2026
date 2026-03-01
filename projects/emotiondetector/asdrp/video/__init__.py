"""Video processing and frame extraction module.

This module provides comprehensive video processing capabilities including:
- Reading frames from video files
- Writing frames to video files
- Capturing frames from cameras in real-time
- Frame and video metadata structures

Example:
    Reading from a video file:
        >>> from asdrp.video import VideoFileReader
        >>> with VideoFileReader("input.mp4") as reader:
        ...     for frame in reader:
        ...         print(f"Frame {frame.frame_number}")

    Writing to a video file:
        >>> from asdrp.video import VideoFileWriter
        >>> with VideoFileWriter("output.mp4", fps=30.0, frame_size=(640, 480)) as writer:
        ...     writer.write_frame(frame_array)

    Capturing from a camera:
        >>> from asdrp.video import CameraCapture
        >>> with CameraCapture(camera_id=0) as camera:
        ...     frame = camera.read_frame()
"""

from .camera import CameraCapture, CameraCaptureError, MultiCameraCapture
from .frame import FrameData, VideoMetadata
from .reader import VideoFileReader, VideoReader, VideoReaderError
from .writer import (
    VideoFileWriter,
    VideoFileWriterFromReader,
    VideoWriter,
    VideoWriterError,
)

__all__ = [
    # Frame and metadata
    "FrameData",
    "VideoMetadata",
    # Readers
    "VideoReader",
    "VideoFileReader",
    "VideoReaderError",
    # Writers
    "VideoWriter",
    "VideoFileWriter",
    "VideoFileWriterFromReader",
    "VideoWriterError",
    # Camera
    "CameraCapture",
    "MultiCameraCapture",
    "CameraCaptureError",
]
