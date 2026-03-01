# ASDRP Utilities Module

Comprehensive utilities for the emotion detection pipeline.

## Quick Start

```python
from asdrp.utils import (
    PipelineConfig,
    calculate_distance_3d,
    KalmanFilter,
    export_emotions_to_json
)

# Create configuration
config = PipelineConfig.for_realtime_processing(
    model_path="model.task",
    input_path="0"  # webcam
)

# Calculate geometry
distance = calculate_distance_3d(point1, point2)

# Apply smoothing
filter = KalmanFilter()
smoothed = filter.update(noisy_value)

# Export results
export_emotions_to_json(emotions, "output.json")
```

## Modules

### config.py - Configuration Management

**Dataclasses:**
- `FaceDetectionConfig` - Face detection settings
- `EmotionAnalysisConfig` - Emotion analysis settings
- `VideoConfig` - Video input/output settings
- `VisualizationConfig` - Rendering settings
- `PipelineConfig` - Complete pipeline configuration

**Factory Methods:**
- `PipelineConfig.from_defaults()` - Standard configuration
- `PipelineConfig.for_realtime_processing()` - Real-time optimized
- `PipelineConfig.for_batch_processing()` - Batch optimized
- `PipelineConfig.for_analysis_only()` - Analysis without visualization

### geometry.py - Geometric Calculations

**Distance Functions:**
- `calculate_distance_3d(p1, p2)` - 3D Euclidean distance
- `calculate_distance_2d(p1, p2)` - 2D Euclidean distance
- `point_line_distance(point, line_p1, line_p2)` - Point to line
- `point_segment_distance(point, seg_p1, seg_p2)` - Point to segment

**Angle Functions:**
- `calculate_angle_3d(p1, vertex, p2)` - 3D angle at vertex
- `calculate_angle_2d(p1, vertex, p2)` - 2D angle at vertex

**Centroid and Normalization:**
- `calculate_centroid(points, weights=None)` - Center of mass
- `normalize_points(points, method='standard')` - Point normalization
- `denormalize_points(points, params, method)` - Reverse normalization

### smoothing.py - Temporal Filtering

**Filter Classes:**
- `MovingAverageFilter(window_size=5)` - Simple moving average
- `ExponentialMovingAverageFilter(alpha=0.3)` - Exponential weighting
- `KalmanFilter(process_var=0.01, meas_var=0.1)` - Optimal estimation
- `MedianFilter(window_size=5)` - Outlier rejection

**Common Methods:**
- `.update(value)` - Update with scalar value
- `.update_array(array)` - Update with numpy array
- `.reset()` - Reset filter state
- `.is_initialized` - Check initialization status

### export.py - Data Export

**Generic Export:**
- `export_to_json(data, path)` - Dict to JSON
- `export_to_csv(data, path)` - List of dicts to CSV

**Landmarks Export:**
- `export_landmarks_to_json(landmarks, path, ...)` - Landmarks to JSON
- `export_landmarks_to_csv(landmarks, path, ...)` - Landmarks to CSV

**Emotions Export:**
- `export_emotions_to_json(emotions, path, ...)` - Emotions to JSON
- `export_emotions_to_csv(emotions, path, ...)` - Emotions to CSV

**Combined Export:**
- `export_analysis_summary(landmarks, emotions, path, ...)` - Complete analysis

## Examples

### Configuration

```python
# Real-time processing with webcam
config = PipelineConfig.for_realtime_processing(
    model_path="face_landmarker.task",
    input_path="0"
)

# Batch processing with custom settings
config = PipelineConfig.for_batch_processing(
    model_path="face_landmarker.task",
    input_path="video.mp4",
    output_path="output.mp4",
    batch_size=16
)

# Custom configuration
config = PipelineConfig(
    face_detection=FaceDetectionConfig(
        model_path="model.task",
        num_faces=2,
        enable_smoothing=True
    ),
    emotion_analysis=EmotionAnalysisConfig(
        analyzer_type='geometric',
        emotion_classes=['happy', 'sad', 'angry', 'neutral']
    ),
    video=VideoConfig(
        input_path="input.mp4",
        skip_frames=2
    ),
    visualization=VisualizationConfig(
        draw_landmarks=True,
        show_emotion=True
    )
)
```

### Geometry

```python
import numpy as np

# Calculate distances
p1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
p2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)
distance = calculate_distance_3d(p1, p2)  # 5.0

# Calculate angles
vertex = np.array([0.0, 0.0, 0.0], dtype=np.float32)
p1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
p2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
angle = calculate_angle_3d(p1, vertex, p2)  # π/2 radians (90 degrees)

# Calculate centroid
points = np.array([[0, 0], [1, 0], [0.5, 1]], dtype=np.float32)
centroid = calculate_centroid(points)  # [0.5, 0.33]

# Normalize points
normalized, params = normalize_points(points, method='standard')
# Later: denormalize
original = denormalize_points(normalized, params, method='standard')
```

### Smoothing

```python
# Moving average filter
ma_filter = MovingAverageFilter(window_size=5)
for value in data:
    smoothed = ma_filter.update(value)

# Exponential moving average
ema_filter = ExponentialMovingAverageFilter(alpha=0.3)
for value in data:
    smoothed = ema_filter.update(value)

# Kalman filter
kalman = KalmanFilter(process_variance=0.01, measurement_variance=0.1)
for value in data:
    smoothed = kalman.update(value)

# Median filter (good for outliers)
median = MedianFilter(window_size=5)
for value in data:
    smoothed = median.update(value)

# Smoothing arrays (e.g., landmarks)
filter = KalmanFilter()
for landmarks in landmark_sequence:
    smoothed_landmarks = filter.update_array(landmarks)
```

### Export

```python
# Export emotions to JSON
emotions = [
    {"happy": 0.8, "sad": 0.1, "angry": 0.1},
    {"happy": 0.3, "sad": 0.6, "angry": 0.1}
]
export_emotions_to_json(
    emotions,
    "emotions.json",
    frame_numbers=[0, 1],
    timestamps=[0.0, 0.033],
    metadata={"fps": 30}
)

# Export to CSV
export_emotions_to_csv(
    emotions,
    "emotions.csv",
    frame_numbers=[0, 1],
    timestamps=[0.0, 0.033]
)

# Export landmarks
landmarks = [np.random.rand(478, 3).astype(np.float32) for _ in range(10)]
export_landmarks_to_json(
    landmarks,
    "landmarks.json",
    frame_numbers=list(range(10))
)

# Export complete analysis
export_analysis_summary(
    landmarks_list=landmarks,
    emotions_list=emotions,
    output_path="analysis.json",
    video_metadata={"resolution": "1920x1080", "fps": 30}
)
```

## Demo

Run the comprehensive demo:

```bash
PYTHONPATH=. python examples/utils_demo.py
```

This demonstrates all features with sample data and creates example exports.

## Testing

All modules have been validated with:
- Syntax checking (py_compile)
- Import testing
- Functional testing
- Type hint validation

## Integration

The utilities module integrates seamlessly with other ASDRP modules:
- `asdrp.face` - Face detection and landmarks
- `asdrp.video` - Video processing
- `asdrp.emotion` - Emotion analysis
- `asdrp.visualization` - Rendering

## API Reference

See individual module docstrings for detailed API documentation:
- All functions have comprehensive docstrings
- Type hints for all parameters and return values
- Usage examples in docstrings
- Error handling documentation

## File Organization

```
asdrp/utils/
├── __init__.py         # Public API exports
├── config.py           # Configuration dataclasses
├── geometry.py         # Geometric calculations
├── smoothing.py        # Temporal filters
├── export.py           # Export utilities
└── README.md           # This file
```

## License

Part of the ASDRP Emotion Detector project.
