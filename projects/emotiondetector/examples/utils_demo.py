"""Demonstration of the utilities module for the emotion detector.

This example shows how to use the various utility classes and functions
including configuration, geometry calculations, smoothing filters, and export.
"""

import numpy as np
from pathlib import Path

from asdrp.utils import (
    # Configuration classes
    PipelineConfig,
    FaceDetectionConfig,
    EmotionAnalysisConfig,
    VideoConfig,
    VisualizationConfig,
    # Geometry functions
    calculate_distance_3d,
    calculate_angle_3d,
    calculate_centroid,
    normalize_points,
    # Smoothing filters
    MovingAverageFilter,
    ExponentialMovingAverageFilter,
    KalmanFilter,
    MedianFilter,
    # Export functions
    export_emotions_to_json,
    export_emotions_to_csv,
    export_landmarks_to_json,
)


def demo_configuration() -> None:
    """Demonstrate configuration creation and usage."""
    print("=" * 70)
    print("Configuration Demo")
    print("=" * 70)

    # Create pipeline config with factory method
    config = PipelineConfig.from_defaults(
        model_path="data/models/face_landmarker.task",
        input_path="data/videos/sample.mp4",
        output_path="output/processed.mp4"
    )
    print(f"\n1. Default Pipeline Config:")
    print(f"   - Batch size: {config.batch_size}")
    print(f"   - Device: {config.device}")
    print(f"   - Output format: {config.output_format}")

    # Create config for real-time processing
    realtime_config = PipelineConfig.for_realtime_processing(
        model_path="data/models/face_landmarker.task",
        input_path="0"  # Webcam
    )
    print(f"\n2. Real-time Processing Config:")
    print(f"   - Display real-time: {realtime_config.video.display_realtime}")
    print(f"   - Smoothing enabled: {realtime_config.face_detection.enable_smoothing}")
    print(f"   - Smoothing window: {realtime_config.face_detection.smoothing_window_size}")

    # Create config for batch processing
    batch_config = PipelineConfig.for_batch_processing(
        model_path="data/models/face_landmarker.task",
        input_path="data/videos/sample.mp4",
        output_path="output/batch_processed.mp4",
        batch_size=16
    )
    print(f"\n3. Batch Processing Config:")
    print(f"   - Batch size: {batch_config.batch_size}")
    print(f"   - Num workers: {batch_config.num_workers}")
    print(f"   - Buffer size: {batch_config.video.buffer_size}")

    # Create custom config
    custom_config = PipelineConfig(
        face_detection=FaceDetectionConfig(
            model_path="data/models/face_landmarker.task",
            num_faces=2,
            min_detection_confidence=0.7,
            enable_smoothing=True,
        ),
        emotion_analysis=EmotionAnalysisConfig(
            analyzer_type='geometric',
            emotion_classes=['happy', 'sad', 'angry', 'neutral'],
        ),
        video=VideoConfig(
            input_path="data/videos/sample.mp4",
            skip_frames=2,
        ),
        visualization=VisualizationConfig(
            draw_landmarks=True,
            show_emotion=True,
            landmark_color=(0, 255, 0),
        ),
    )
    print(f"\n4. Custom Config:")
    print(f"   - Num faces: {custom_config.face_detection.num_faces}")
    print(f"   - Emotion classes: {custom_config.emotion_analysis.emotion_classes}")
    print(f"   - Skip frames: {custom_config.video.skip_frames}")


def demo_geometry() -> None:
    """Demonstrate geometry utility functions."""
    print("\n" + "=" * 70)
    print("Geometry Demo")
    print("=" * 70)

    # Calculate 3D distance
    point1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    point2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)
    distance = calculate_distance_3d(point1, point2)
    print(f"\n1. Distance between {point1} and {point2}:")
    print(f"   Distance = {distance:.2f}")

    # Calculate angle
    vertex = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    p1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    p2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    angle = calculate_angle_3d(p1, vertex, p2)
    print(f"\n2. Angle at vertex {vertex}:")
    print(f"   Angle = {np.degrees(angle):.1f} degrees")

    # Calculate centroid
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
    ], dtype=np.float32)
    centroid = calculate_centroid(points)
    print(f"\n3. Centroid of triangle:")
    print(f"   Points: {points}")
    print(f"   Centroid: {centroid}")

    # Normalize points
    random_points = np.random.rand(10, 3).astype(np.float32) * 10
    normalized, params = normalize_points(random_points, method='standard')
    print(f"\n4. Point normalization (z-score):")
    print(f"   Original mean: {np.mean(random_points, axis=0)}")
    print(f"   Original std: {np.std(random_points, axis=0)}")
    print(f"   Normalized mean: {np.mean(normalized, axis=0)}")
    print(f"   Normalized std: {np.std(normalized, axis=0)}")


def demo_smoothing() -> None:
    """Demonstrate smoothing filters."""
    print("\n" + "=" * 70)
    print("Smoothing Filters Demo")
    print("=" * 70)

    # Noisy signal (simulating landmark coordinates or emotion scores)
    np.random.seed(42)
    true_signal = np.linspace(0, 1, 20)
    noisy_signal = true_signal + np.random.normal(0, 0.1, 20)

    print(f"\n1. Moving Average Filter:")
    ma_filter = MovingAverageFilter(window_size=5)
    smoothed_ma = [ma_filter.update(val) for val in noisy_signal]
    print(f"   Window size: {ma_filter.window_size}")
    print(f"   Original noise std: {np.std(noisy_signal - true_signal):.3f}")
    print(f"   Smoothed error std: {np.std(np.array(smoothed_ma) - true_signal):.3f}")

    print(f"\n2. Exponential Moving Average Filter:")
    ema_filter = ExponentialMovingAverageFilter(alpha=0.3)
    smoothed_ema = [ema_filter.update(val) for val in noisy_signal]
    print(f"   Alpha: {ema_filter.alpha}")
    print(f"   Original noise std: {np.std(noisy_signal - true_signal):.3f}")
    print(f"   Smoothed error std: {np.std(np.array(smoothed_ema) - true_signal):.3f}")

    print(f"\n3. Kalman Filter:")
    kalman_filter = KalmanFilter(process_variance=0.01, measurement_variance=0.1)
    smoothed_kalman = [kalman_filter.update(val) for val in noisy_signal]
    print(f"   Process variance: {kalman_filter.process_variance}")
    print(f"   Measurement variance: {kalman_filter.measurement_variance}")
    print(f"   Original noise std: {np.std(noisy_signal - true_signal):.3f}")
    print(f"   Smoothed error std: {np.std(np.array(smoothed_kalman) - true_signal):.3f}")

    print(f"\n4. Median Filter:")
    # Add some outliers to demonstrate median filter effectiveness
    outlier_signal = noisy_signal.copy()
    outlier_signal[5] = 5.0  # Large outlier
    outlier_signal[10] = -2.0  # Negative outlier

    median_filter = MedianFilter(window_size=5)
    smoothed_median = [median_filter.update(val) for val in outlier_signal]
    print(f"   Window size: {median_filter.window_size}")
    print(f"   Signal with outliers - max error: {np.max(np.abs(outlier_signal - true_signal)):.3f}")
    print(f"   Median filtered - max error: {np.max(np.abs(np.array(smoothed_median) - true_signal)):.3f}")


def demo_export() -> None:
    """Demonstrate export functions."""
    print("\n" + "=" * 70)
    print("Export Demo")
    print("=" * 70)

    # Create sample data
    num_frames = 10

    # Emotion data
    emotions_list = []
    for i in range(num_frames):
        emotions = {
            'happy': np.random.rand(),
            'sad': np.random.rand(),
            'angry': np.random.rand(),
            'neutral': np.random.rand(),
        }
        # Normalize to sum to 1
        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}
        emotions_list.append(emotions)

    frame_numbers = list(range(num_frames))
    timestamps = [i * 0.033 for i in range(num_frames)]  # 30 fps

    # Export to JSON
    output_dir = Path("output/demo")
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "emotions.json"
    export_emotions_to_json(
        emotions_list,
        json_path,
        frame_numbers=frame_numbers,
        timestamps=timestamps,
        metadata={"fps": 30, "source": "demo"}
    )
    print(f"\n1. Exported emotions to JSON:")
    print(f"   File: {json_path}")
    print(f"   Frames: {num_frames}")

    # Export to CSV
    csv_path = output_dir / "emotions.csv"
    export_emotions_to_csv(
        emotions_list,
        csv_path,
        frame_numbers=frame_numbers,
        timestamps=timestamps
    )
    print(f"\n2. Exported emotions to CSV:")
    print(f"   File: {csv_path}")
    print(f"   Frames: {num_frames}")

    # Export landmarks
    landmarks_list = [
        np.random.rand(478, 3).astype(np.float32) for _ in range(num_frames)
    ]
    landmarks_path = output_dir / "landmarks.json"
    export_landmarks_to_json(
        landmarks_list,
        landmarks_path,
        frame_numbers=frame_numbers,
        timestamps=timestamps,
        metadata={"model": "mediapipe_face_landmarker"}
    )
    print(f"\n3. Exported landmarks to JSON:")
    print(f"   File: {landmarks_path}")
    print(f"   Frames: {num_frames}")
    print(f"   Landmarks per frame: {landmarks_list[0].shape[0]}")


def main() -> None:
    """Run all demonstrations."""
    print("\n")
    print("*" * 70)
    print("ASDRP Emotion Detector - Utilities Module Demo")
    print("*" * 70)

    demo_configuration()
    demo_geometry()
    demo_smoothing()
    demo_export()

    print("\n" + "=" * 70)
    print("Demo Complete!")
    print("=" * 70)
    print("\nCheck the 'output/demo' directory for exported files.")


if __name__ == "__main__":
    main()
