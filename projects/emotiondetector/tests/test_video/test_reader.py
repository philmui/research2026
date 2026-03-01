"""Tests for video file reader.

This module tests the VideoFileReader class for reading video files.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from asdrp.video.frame import FrameData
from asdrp.video.reader import VideoFileReader


class TestVideoFileReader:
    """Test suite for VideoFileReader class."""

    def test_initialization_valid_file(self, temp_video_file: Path) -> None:
        """Test initialization with valid video file."""
        reader = VideoFileReader(str(temp_video_file))
        assert reader.video_path == temp_video_file
        reader.close()

    def test_initialization_invalid_file(self) -> None:
        """Test initialization with non-existent file."""
        with pytest.raises(FileNotFoundError):
            VideoFileReader("nonexistent_video.mp4")

    def test_context_manager(self, temp_video_file: Path) -> None:
        """Test using reader as context manager."""
        with VideoFileReader(str(temp_video_file)) as reader:
            assert reader is not None
            metadata = reader.get_metadata()
            assert metadata is not None

    def test_get_metadata(self, temp_video_file: Path) -> None:
        """Test getting video metadata."""
        with VideoFileReader(str(temp_video_file)) as reader:
            metadata = reader.get_metadata()

            assert metadata.width > 0
            assert metadata.height > 0
            assert metadata.fps > 0
            assert metadata.total_frames > 0

    def test_read_frame(self, temp_video_file: Path) -> None:
        """Test reading a single frame."""
        with VideoFileReader(str(temp_video_file)) as reader:
            frame_data = reader.read_frame()

            assert frame_data is not None
            assert isinstance(frame_data, FrameData)
            assert frame_data.frame.shape[2] == 3  # BGR
            assert frame_data.frame.dtype == np.uint8

    def test_read_all_frames(self, temp_video_file: Path) -> None:
        """Test reading all frames via iteration."""
        with VideoFileReader(str(temp_video_file)) as reader:
            frames = list(reader)

            assert len(frames) > 0
            assert all(isinstance(f, FrameData) for f in frames)

    def test_frame_numbers_sequential(self, temp_video_file: Path) -> None:
        """Test that frame numbers are sequential."""
        with VideoFileReader(str(temp_video_file)) as reader:
            frames = list(reader)

            for i, frame_data in enumerate(frames):
                assert frame_data.frame_number == i

    def test_seek(self, temp_video_file: Path) -> None:
        """Test seeking to a specific frame."""
        with VideoFileReader(str(temp_video_file)) as reader:
            reader.seek(5)
            frame_data = reader.read_frame()

            assert frame_data is not None
            # Frame number might not be exactly 5 depending on codec
            assert frame_data.frame_number >= 5

    def test_close(self, temp_video_file: Path) -> None:
        """Test closing the reader."""
        reader = VideoFileReader(str(temp_video_file))
        reader.close()

        # Reading after close should fail
        with pytest.raises((RuntimeError, AttributeError)):
            reader.read_frame()

    def test_double_close(self, temp_video_file: Path) -> None:
        """Test that double close doesn't cause errors."""
        reader = VideoFileReader(str(temp_video_file))
        reader.close()
        reader.close()  # Should not raise

    @pytest.mark.slow
    def test_read_large_video(self, temp_video_file: Path) -> None:
        """Test reading video with many frames."""
        # This test is marked as slow
        with VideoFileReader(str(temp_video_file)) as reader:
            count = 0
            for frame_data in reader:
                count += 1
                if count >= 100:  # Limit for test
                    break

            assert count > 0


class TestVideoFileReaderEdgeCases:
    """Test suite for edge cases in video reading."""

    def test_empty_video_path(self) -> None:
        """Test initialization with empty path."""
        with pytest.raises(FileNotFoundError):
            VideoFileReader("")

    def test_invalid_video_format(self, temp_output_dir: Path) -> None:
        """Test opening invalid video format."""
        # Create a text file with .mp4 extension
        fake_video = temp_output_dir / "fake.mp4"
        fake_video.write_text("not a video")

        with pytest.raises((RuntimeError, FileNotFoundError)):
            VideoFileReader(str(fake_video))

    def test_corrupted_video(self, temp_output_dir: Path) -> None:
        """Test handling corrupted video file."""
        # Create a partially written video file
        corrupted_video = temp_output_dir / "corrupted.mp4"

        # Write some random bytes
        with open(corrupted_video, "wb") as f:
            f.write(b"ftyp" + b"\x00" * 100)

        # Should either fail to open or handle gracefully
        try:
            reader = VideoFileReader(str(corrupted_video))
            reader.close()
        except (RuntimeError, FileNotFoundError):
            pass  # Expected

    def test_read_after_end(self, temp_video_file: Path) -> None:
        """Test reading after reaching end of video."""
        with VideoFileReader(str(temp_video_file)) as reader:
            # Read all frames
            frames = list(reader)

            # Try to read one more
            frame_data = reader.read_frame()
            assert frame_data is None


class TestFrameData:
    """Test suite for FrameData dataclass."""

    def test_frame_data_creation(self, sample_face_image: np.ndarray) -> None:
        """Test creating FrameData object."""
        frame_data = FrameData(
            frame=sample_face_image,
            frame_number=10,
            timestamp=0.333,
        )

        assert np.array_equal(frame_data.frame, sample_face_image)
        assert frame_data.frame_number == 10
        assert frame_data.timestamp == pytest.approx(0.333)

    def test_frame_data_copy(self, sample_face_image: np.ndarray) -> None:
        """Test that frame data can be copied."""
        frame_data = FrameData(
            frame=sample_face_image,
            frame_number=1,
            timestamp=0.0,
        )

        # Modify original image
        frame_copy = frame_data.frame.copy()
        frame_data.frame[0, 0, 0] = 255

        # Copy should be different
        assert not np.array_equal(frame_copy[0, 0], frame_data.frame[0, 0])
