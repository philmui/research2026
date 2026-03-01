"""Example script demonstrating the EmotionDetectionPipeline usage.

This script shows various ways to use the pipeline for emotion detection,
including batch video processing, real-time camera processing, and different
configuration options.
"""

from pathlib import Path

from asdrp import EmotionDetectionPipeline, PipelineConfig


def example_basic_video_processing():
    """Example 1: Basic video processing with default settings."""
    print("\n" + "=" * 60)
    print("Example 1: Basic Video Processing")
    print("=" * 60)

    # Create configuration with defaults
    config = PipelineConfig.from_defaults(
        model_path="models/face_landmarker.task",
        input_path="input_video.mp4",
        output_path="output_video.mp4",
    )

    # Process video
    with EmotionDetectionPipeline(config) as pipeline:
        print("Processing video...")
        results = pipeline.process_video(show_progress=True)

        print(f"\nProcessed {len(results)} frames")
        print(f"Total faces detected: {sum(len(r['faces']) for r in results)}")

        # Save results
        pipeline.save_results("results.json")
        print("Results saved to results.json")


def example_realtime_camera():
    """Example 2: Real-time camera processing."""
    print("\n" + "=" * 60)
    print("Example 2: Real-time Camera Processing")
    print("=" * 60)

    # Create configuration for real-time processing
    config = PipelineConfig.for_realtime_processing(
        model_path="models/face_landmarker.task",
        input_path="0",  # Use default webcam
    )

    # Process camera stream
    with EmotionDetectionPipeline(config) as pipeline:
        print("Starting camera stream... (Press 'q' to quit)")
        frame_count = 0

        for result in pipeline.process_stream(max_frames=300, display=True):
            frame_count += 1

            # Print emotion for first face
            if result["faces"]:
                emotion = result["faces"][0]["emotion"]
                confidence = result["faces"][0]["confidence"]
                print(f"Frame {frame_count}: {emotion} ({confidence:.2f})")

        print(f"\nProcessed {frame_count} frames from camera")


def example_batch_processing():
    """Example 3: Batch processing with custom configuration."""
    print("\n" + "=" * 60)
    print("Example 3: Batch Processing with Custom Configuration")
    print("=" * 60)

    # Create configuration for batch processing
    config = PipelineConfig.for_batch_processing(
        model_path="models/face_landmarker.task",
        input_path="input_video.mp4",
        output_path="output_batch.mp4",
        batch_size=8,
    )

    # Customize settings
    config.video.skip_frames = 2  # Process every 3rd frame
    config.visualization.draw_landmarks = True
    config.visualization.landmark_connections = True
    config.output_format = "both"  # Save both JSON and CSV

    # Process video
    with EmotionDetectionPipeline(config) as pipeline:
        print("Processing video in batch mode...")
        results = pipeline.process_video(show_progress=True)

        print(f"\nProcessed {len(results)} frames")

        # Analyze results
        emotions_count = {}
        for result in results:
            for face in result["faces"]:
                emotion = face["emotion"]
                emotions_count[emotion] = emotions_count.get(emotion, 0) + 1

        print("\nEmotion distribution:")
        for emotion, count in sorted(
            emotions_count.items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {emotion}: {count} frames ({count/len(results)*100:.1f}%)")

        # Save results in both formats
        pipeline.save_results("results_batch.json")
        print("Results saved to results_batch.json and results_batch.csv")


def example_analysis_only():
    """Example 4: Analysis without video output."""
    print("\n" + "=" * 60)
    print("Example 4: Analysis Only (No Video Output)")
    print("=" * 60)

    # Create configuration for analysis only
    config = PipelineConfig.for_analysis_only(
        model_path="models/face_landmarker.task",
        input_path="input_video.mp4",
    )

    # Process video
    with EmotionDetectionPipeline(config) as pipeline:
        print("Analyzing video (no visualization)...")
        results = pipeline.process_video(show_progress=True)

        print(f"\nProcessed {len(results)} frames")

        # Calculate emotion statistics
        emotion_scores = {
            "neutral": [],
            "happy": [],
            "sad": [],
            "angry": [],
            "surprised": [],
            "fearful": [],
        }

        for result in results:
            if result["faces"]:
                probs = result["faces"][0]["probabilities"]
                for emotion, score in probs.items():
                    if emotion in emotion_scores:
                        emotion_scores[emotion].append(score)

        print("\nAverage emotion scores:")
        for emotion, scores in emotion_scores.items():
            if scores:
                avg_score = sum(scores) / len(scores)
                print(f"  {emotion}: {avg_score:.3f}")

        # Save results
        pipeline.save_results("analysis_results.csv")
        print("Analysis saved to analysis_results.csv")


def example_custom_configuration():
    """Example 5: Custom configuration with fine-tuned parameters."""
    print("\n" + "=" * 60)
    print("Example 5: Custom Configuration")
    print("=" * 60)

    # Import individual config classes
    from asdrp import (
        EmotionAnalysisConfig,
        FaceDetectionConfig,
        VideoConfig,
        VisualizationConfig,
    )

    # Create custom configuration
    config = PipelineConfig(
        face_detection=FaceDetectionConfig(
            model_path="models/face_landmarker.task",
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            num_faces=2,  # Detect up to 2 faces
            running_mode="VIDEO",
            enable_smoothing=True,
            smoothing_window_size=5,
        ),
        emotion_analysis=EmotionAnalysisConfig(
            analyzer_type="geometric",
            confidence_threshold=0.6,
            enable_temporal_smoothing=True,
            smoothing_window_size=10,
        ),
        video=VideoConfig(
            input_path="input_video.mp4",
            output_path="output_custom.mp4",
            start_frame=0,
            end_frame=300,  # Process first 300 frames only
            skip_frames=0,
            display_realtime=False,
        ),
        visualization=VisualizationConfig(
            draw_landmarks=True,
            draw_bounding_box=True,
            show_emotion=True,
            show_confidence=True,
            show_timestamp=True,
            landmark_color=(0, 255, 0),  # Green
            bbox_color=(255, 0, 0),  # Blue
            text_color=(255, 255, 255),  # White
            font_scale=0.8,
        ),
        output_format="json",
        save_annotated_video=True,
        save_emotions=True,
        log_level="INFO",
    )

    # Process video
    with EmotionDetectionPipeline(config) as pipeline:
        print("Processing with custom configuration...")
        results = pipeline.process_video(show_progress=True)

        print(f"\nProcessed {len(results)} frames")
        print(f"Configuration: {config.to_dict()}")

        # Save results
        pipeline.save_results("custom_results.json")
        print("Results saved to custom_results.json")


def example_single_frame():
    """Example 6: Process single frames individually."""
    print("\n" + "=" * 60)
    print("Example 6: Single Frame Processing")
    print("=" * 60)

    from asdrp import VideoFileReader

    # Create configuration
    config = PipelineConfig.from_defaults(
        model_path="models/face_landmarker.task",
        input_path="input_video.mp4",
    )

    # Process specific frames
    with EmotionDetectionPipeline(config) as pipeline:
        print("Processing individual frames...")

        with VideoFileReader("input_video.mp4") as reader:
            # Get frame 100
            frame_data = reader.get_frame_at(100)
            if frame_data:
                result = pipeline.process_frame(frame_data, visualize=False)
                print(f"\nFrame 100:")
                print(f"  Faces detected: {len(result['faces'])}")
                if result["faces"]:
                    print(f"  Emotion: {result['faces'][0]['emotion']}")
                    print(f"  Confidence: {result['faces'][0]['confidence']:.2f}")

            # Get frame 200
            frame_data = reader.get_frame_at(200)
            if frame_data:
                result = pipeline.process_frame(frame_data, visualize=False)
                print(f"\nFrame 200:")
                print(f"  Faces detected: {len(result['faces'])}")
                if result["faces"]:
                    print(f"  Emotion: {result['faces'][0]['emotion']}")
                    print(f"  Confidence: {result['faces'][0]['confidence']:.2f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Emotion Detection Pipeline Examples")
    print("=" * 60)

    # Check if model file exists
    model_path = Path("models/face_landmarker.task")
    if not model_path.exists():
        print("\nERROR: Model file not found!")
        print(f"Please download the MediaPipe Face Landmarker model to: {model_path}")
        print("\nDownload from:")
        print("https://developers.google.com/mediapipe/solutions/vision/face_landmarker")
        return

    # Run examples
    try:
        # Uncomment the examples you want to run:

        # example_basic_video_processing()
        # example_realtime_camera()
        # example_batch_processing()
        # example_analysis_only()
        # example_custom_configuration()
        # example_single_frame()

        print("\n" + "=" * 60)
        print("Examples completed!")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
