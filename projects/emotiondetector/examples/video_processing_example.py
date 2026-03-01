"""Example usage of the video processing module.

This script demonstrates how to use the asdrp.video module for various
video processing tasks including reading, writing, and camera capture.
"""

import numpy as np

from asdrp.video import (
    CameraCapture,
    FrameData,
    VideoFileReader,
    VideoFileWriter,
    VideoFileWriterFromReader,
)


def example_read_video() -> None:
    """Example: Read frames from a video file."""
    print("Example 1: Reading video frames")
    print("-" * 40)

    video_path = "data/videos/sample.mp4"

    # Using context manager (recommended)
    with VideoFileReader(video_path) as reader:
        # Get video metadata
        metadata = reader.get_metadata()
        print(f"Video: {metadata}")
        print(f"Total frames: {len(reader)}")

        # Iterate through frames
        for i, frame_data in enumerate(reader):
            print(f"  Frame {frame_data.frame_number}: {frame_data.shape}")

            if i >= 10:  # Process first 10 frames
                break

    print()


def example_write_video() -> None:
    """Example: Write frames to a video file."""
    print("Example 2: Writing video frames")
    print("-" * 40)

    output_path = "data/output/generated.mp4"
    fps = 30.0
    frame_size = (640, 480)

    with VideoFileWriter(output_path, fps=fps, frame_size=frame_size) as writer:
        # Generate and write 90 frames (3 seconds)
        for i in range(90):
            # Create a test frame (gradient pattern)
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:, :, 0] = (i * 255) // 90  # Blue channel
            frame[:, :, 1] = 128  # Green channel
            frame[:, :, 2] = 255 - (i * 255) // 90  # Red channel

            writer.write_frame(frame)

        print(f"Wrote {writer.get_frame_count()} frames to {output_path}")
        print(f"Video metadata: {writer.get_metadata()}")

    print()


def example_process_video() -> None:
    """Example: Read, process, and write video."""
    print("Example 3: Processing video (read and write)")
    print("-" * 40)

    input_path = "data/videos/sample.mp4"
    output_path = "data/output/processed.mp4"

    # Open input video
    with VideoFileReader(input_path) as reader:
        # Create output writer with same settings as input
        with VideoFileWriterFromReader(output_path, reader) as writer:
            # Process each frame
            for frame_data in reader:
                # Apply simple processing (convert to grayscale and back to BGR)
                gray = np.mean(frame_data.frame, axis=2, keepdims=True).astype(np.uint8)
                processed = np.repeat(gray, 3, axis=2)

                # Write processed frame
                writer.write_frame(processed)

        print(f"Processed {writer.get_frame_count()} frames")

    print()


def example_camera_capture() -> None:
    """Example: Capture frames from webcam."""
    print("Example 4: Camera capture")
    print("-" * 40)

    try:
        # Open default camera (camera_id=0)
        with CameraCapture(camera_id=0, width=640, height=480) as camera:
            # Get camera metadata
            metadata = camera.get_metadata()
            print(f"Camera: {metadata}")

            # Capture 30 frames
            for i in range(30):
                frame_data = camera.read_frame()
                if frame_data:
                    print(f"  Captured frame {frame_data.frame_number}: {frame_data.shape}")
                else:
                    print("  Failed to capture frame")
                    break

            print(f"Total frames captured: {camera.get_frame_count()}")

    except Exception as e:
        print(f"Camera error: {e}")

    print()


def example_seek_and_random_access() -> None:
    """Example: Random access to video frames."""
    print("Example 5: Random frame access")
    print("-" * 40)

    video_path = "data/videos/sample.mp4"

    with VideoFileReader(video_path) as reader:
        metadata = reader.get_metadata()
        total_frames = metadata.total_frames

        # Access specific frames
        frame_numbers = [0, total_frames // 4, total_frames // 2, total_frames - 1]

        for frame_num in frame_numbers:
            frame_data = reader.get_frame_at(frame_num)
            if frame_data:
                print(
                    f"  Frame {frame_data.frame_number} "
                    f"at {frame_data.timestamp:.2f}s: {frame_data.shape}"
                )

    print()


def example_frame_data() -> None:
    """Example: Working with FrameData objects."""
    print("Example 6: FrameData operations")
    print("-" * 40)

    # Create a frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    frame_data = FrameData(
        frame=frame,
        frame_number=42,
        timestamp=1.4,
        metadata={"brightness": 128, "contrast": 1.5},
    )

    # Access properties
    print(f"Frame number: {frame_data.frame_number}")
    print(f"Timestamp: {frame_data.timestamp}s")
    print(f"Dimensions: {frame_data.width}x{frame_data.height}")
    print(f"Channels: {frame_data.channels}")
    print(f"Metadata: {frame_data.metadata}")

    # Copy frame
    frame_copy = frame_data.copy()
    print(f"Copied frame number: {frame_copy.frame_number}")

    print()


def main() -> None:
    """Run all examples."""
    print("=" * 60)
    print("Video Processing Module Examples")
    print("=" * 60)
    print()

    # Example 6: FrameData (works without video files)
    example_frame_data()

    # Example 2: Write video (works without input files)
    # Uncomment to run:
    # example_write_video()

    # Examples requiring video files (uncomment to run):
    # example_read_video()
    # example_process_video()
    # example_seek_and_random_access()

    # Example requiring camera (uncomment to run):
    # example_camera_capture()

    print("=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
