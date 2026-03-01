# Emotion Detection Pipeline

A comprehensive emotion detection system that processes video files and real-time camera streams to detect and analyze facial emotions.

## Overview

The Emotion Detection Pipeline is the main orchestration component that brings together all the modules of the ASDRP emotion detection system:

- **Video Processing**: Read video files or camera streams
- **Face Detection**: Detect faces and extract 478 3D facial landmarks using MediaPipe
- **Emotion Analysis**: Classify emotions using geometry-based Facial Action Coding System (FACS)
- **Temporal Analysis**: Smooth predictions over time for stable results
- **Visualization**: Annotate frames with landmarks, bounding boxes, and emotion labels
- **Export**: Save results to JSON or CSV formats

## Features

- Flexible configuration system with preset configurations
- Support for video files and real-time camera streams
- Single frame, batch, and streaming processing modes
- Multiple face detection and tracking
- Temporal smoothing for stable emotion predictions
- Customizable visualization
- Progress tracking with tqdm
- Comprehensive error handling
- Context manager support for automatic resource cleanup

## Quick Start

### Installation

```bash
pip install opencv-python mediapipe numpy tqdm
```

### Download Model

Download the MediaPipe Face Landmarker model:

```bash
mkdir -p models
cd models
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Basic Usage

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

# Create configuration
config = PipelineConfig.from_defaults(
    model_path="models/face_landmarker.task",
    input_path="input.mp4",
    output_path="output.mp4"
)

# Process video
with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video()
    pipeline.save_results("results.json")
```

## Architecture

### Pipeline Components

```
┌─────────────────────────────────────────────────────┐
│         EmotionDetectionPipeline                    │
├─────────────────────────────────────────────────────┤
│                                                     │
│  ┌──────────────┐     ┌─────────────────┐         │
│  │ VideoReader  │────▶│ Face Detector   │         │
│  │  or Camera   │     │  (MediaPipe)    │         │
│  └──────────────┘     └─────────────────┘         │
│                              │                     │
│                              ▼                     │
│                    ┌──────────────────┐           │
│                    │ Emotion Analyzer │           │
│                    │   (FACS-based)   │           │
│                    └──────────────────┘           │
│                              │                     │
│                              ▼                     │
│                    ┌──────────────────┐           │
│                    │Temporal Smoother │           │
│                    │   (Optional)     │           │
│                    └──────────────────┘           │
│                              │                     │
│                              ▼                     │
│                    ┌──────────────────┐           │
│                    │   Visualizer     │           │
│                    │   (Optional)     │           │
│                    └──────────────────┘           │
│                              │                     │
│                              ▼                     │
│                    ┌──────────────────┐           │
│                    │  Results Export  │           │
│                    │  (JSON/CSV)      │           │
│                    └──────────────────┘           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### Configuration System

The pipeline uses a hierarchical configuration system:

- **PipelineConfig**: Top-level configuration
  - **FaceDetectionConfig**: Face detection settings
  - **EmotionAnalysisConfig**: Emotion analysis settings
  - **VideoConfig**: Video processing settings
  - **VisualizationConfig**: Visualization settings

## Usage Examples

### 1. Real-time Camera Processing

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

config = PipelineConfig.for_realtime_processing(
    model_path="models/face_landmarker.task",
    input_path="0"  # Default webcam
)

with EmotionDetectionPipeline(config) as pipeline:
    for result in pipeline.process_stream(max_frames=300, display=True):
        if result["faces"]:
            emotion = result["faces"][0]["emotion"]
            confidence = result["faces"][0]["confidence"]
            print(f"{emotion}: {confidence:.2f}")
```

### 2. Batch Video Processing

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

config = PipelineConfig.for_batch_processing(
    model_path="models/face_landmarker.task",
    input_path="input.mp4",
    output_path="output.mp4",
    batch_size=16
)

with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video(show_progress=True)
    pipeline.save_results("results.json")
```

### 3. Analysis Without Video Output

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

config = PipelineConfig.for_analysis_only(
    model_path="models/face_landmarker.task",
    input_path="input.mp4"
)

with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video()
    pipeline.save_results("analysis.csv")
```

### 4. Custom Configuration

```python
from asdrp import (
    EmotionDetectionPipeline,
    PipelineConfig,
    FaceDetectionConfig,
    EmotionAnalysisConfig,
    VideoConfig,
    VisualizationConfig
)

config = PipelineConfig(
    face_detection=FaceDetectionConfig(
        model_path="models/face_landmarker.task",
        min_detection_confidence=0.7,
        num_faces=2,
        running_mode="VIDEO"
    ),
    emotion_analysis=EmotionAnalysisConfig(
        analyzer_type="geometric",
        confidence_threshold=0.6,
        enable_temporal_smoothing=True
    ),
    video=VideoConfig(
        input_path="input.mp4",
        output_path="output.mp4",
        skip_frames=1  # Process every other frame
    ),
    visualization=VisualizationConfig(
        draw_landmarks=True,
        show_emotion=True,
        landmark_color=(0, 255, 0)
    )
)

with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video()
    pipeline.save_results("results.json")
```

### 5. Processing Specific Frame Range

```python
config = PipelineConfig.from_defaults(
    model_path="models/face_landmarker.task",
    input_path="input.mp4"
)

# Process frames 100-500
config.video.start_frame = 100
config.video.end_frame = 500

with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video()
    print(f"Processed {len(results)} frames")
```

### 6. Single Frame Processing

```python
from asdrp import EmotionDetectionPipeline, VideoFileReader

config = PipelineConfig.from_defaults(
    model_path="models/face_landmarker.task",
    input_path="input.mp4"
)

with EmotionDetectionPipeline(config) as pipeline:
    with VideoFileReader("input.mp4") as reader:
        frame_data = reader.get_frame_at(100)
        result = pipeline.process_frame(frame_data)

        for face in result["faces"]:
            print(f"Emotion: {face['emotion']}")
            print(f"Confidence: {face['confidence']:.2f}")
```

## Configuration Options

### Face Detection

```python
FaceDetectionConfig(
    model_path="models/face_landmarker.task",
    min_detection_confidence=0.5,      # 0.0 - 1.0
    min_tracking_confidence=0.5,       # 0.0 - 1.0
    num_faces=1,                       # Max faces to detect
    running_mode="VIDEO",              # "IMAGE" or "VIDEO"
    enable_smoothing=False,            # Temporal smoothing
    smoothing_window_size=5,           # Smoothing window
    smoothing_alpha=0.3                # Smoothing factor
)
```

### Emotion Analysis

```python
EmotionAnalysisConfig(
    analyzer_type="geometric",         # "geometric", "cnn", "hybrid"
    confidence_threshold=0.5,          # 0.0 - 1.0
    enable_temporal_smoothing=True,    # Smooth predictions
    smoothing_window_size=10           # Smoothing window
)
```

### Video Processing

```python
VideoConfig(
    input_path="input.mp4",            # Video file or camera ID
    output_path="output.mp4",          # Output video (optional)
    codec="mp4v",                      # FourCC codec
    fps=None,                          # Use input fps if None
    resolution=None,                   # Use input resolution if None
    start_frame=0,                     # Start frame
    end_frame=None,                    # End frame (None = end)
    skip_frames=0,                     # Skip N frames between processing
    max_frames=None,                   # Max frames to process
    display_realtime=False,            # Display during processing
    buffer_size=32                     # Frame buffer size
)
```

### Visualization

```python
VisualizationConfig(
    draw_landmarks=True,               # Draw facial landmarks
    draw_bounding_box=True,            # Draw face bounding box
    show_emotion=True,                 # Show emotion label
    show_confidence=True,              # Show confidence score
    show_timestamp=False,              # Show frame timestamp
    landmark_color=(0, 255, 0),        # BGR color tuple
    bbox_color=(255, 0, 0),            # BGR color tuple
    text_color=(255, 255, 255),        # BGR color tuple
    landmark_radius=2,                 # Landmark point size
    line_thickness=2,                  # Line thickness
    font_scale=0.7,                    # Text size
    background_alpha=0.5               # Text background opacity
)
```

## Output Format

### Result Structure

Each frame result contains:

```python
{
    "frame_number": 0,
    "timestamp": 0.0,
    "faces": [
        {
            "face_id": 0,
            "emotion": "happy",
            "confidence": 0.85,
            "probabilities": {
                "neutral": 0.05,
                "happy": 0.85,
                "sad": 0.02,
                "angry": 0.03,
                "surprised": 0.03,
                "fearful": 0.02
            },
            "bounding_box": {
                "x_min": 0.2,
                "y_min": 0.2,
                "width": 0.6,
                "height": 0.6
            }
        }
    ]
}
```

### JSON Export

```json
{
  "num_frames": 300,
  "emotion_classes": ["neutral", "happy", "sad", "angry", "surprised", "fearful"],
  "frames": [
    {
      "frame_number": 0,
      "timestamp": 0.0,
      "predicted_emotion": "happy",
      "confidence": 0.85,
      "emotions": {
        "neutral": 0.05,
        "happy": 0.85,
        "sad": 0.02,
        "angry": 0.03,
        "surprised": 0.03,
        "fearful": 0.02
      }
    }
  ]
}
```

### CSV Export

| frame_number | timestamp | predicted_emotion | confidence | neutral | happy | sad | angry | surprised | fearful |
|--------------|-----------|-------------------|------------|---------|-------|-----|-------|-----------|---------|
| 0            | 0.0       | happy             | 0.85       | 0.05    | 0.85  | 0.02| 0.03  | 0.03      | 0.02    |

## Performance

### Optimization Tips

1. **Frame Skipping**: Process every Nth frame
   ```python
   config.video.skip_frames = 2  # Process every 3rd frame
   ```

2. **Reduce Resolution**: Lower resolution for faster processing
   ```python
   config.video.resolution = (640, 480)
   ```

3. **Disable Visualization**: Skip rendering for faster processing
   ```python
   config.save_annotated_video = False
   config.visualization.draw_landmarks = False
   ```

4. **Batch Processing**: Use batch mode for better throughput
   ```python
   config = PipelineConfig.for_batch_processing(
       model_path="models/face_landmarker.task",
       input_path="input.mp4",
       output_path="output.mp4",
       batch_size=16
   )
   ```

### Benchmarks

Approximate processing speeds (dependent on hardware):

| Mode         | Resolution | FPS  | Config                    |
|--------------|------------|------|---------------------------|
| Real-time    | 640x480    | 25+  | for_realtime_processing() |
| Batch        | 1920x1080  | 15+  | for_batch_processing()    |
| Analysis     | 1920x1080  | 30+  | for_analysis_only()       |

## Error Handling

The pipeline provides comprehensive error handling:

```python
from asdrp import PipelineError

try:
    with EmotionDetectionPipeline(config) as pipeline:
        results = pipeline.process_video()
except PipelineError as e:
    print(f"Pipeline error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Testing

Run the test suite:

```bash
python -m pytest tests/test_pipeline.py -v
```

Or run specific tests:

```bash
python tests/test_pipeline.py
```

## Examples

See the `examples/` directory for complete examples:

- `pipeline_example.py`: Comprehensive usage examples
- `video_processing_example.py`: Video processing examples

## Documentation

For detailed documentation, see:

- `PIPELINE_USAGE.md`: Comprehensive usage guide
- API documentation in source code docstrings
- Configuration reference in `asdrp/utils/config.py`

## Supported Emotions

The pipeline detects the following emotions using FACS-based rules:

- **Neutral**: No significant emotion
- **Happy**: Joy, happiness (AU6 + AU12)
- **Sad**: Sadness (AU1 + AU4 + AU15)
- **Angry**: Anger, frustration (AU4 + AU7 + AU23)
- **Surprised**: Surprise, shock (AU1 + AU2 + AU5 + AU26)
- **Fearful**: Fear, anxiety (AU1 + AU2 + AU4 + AU5 + AU20)
- **Disgusted**: Disgust (AU9 + AU15 + AU17)

## License

MIT License

## Citation

If you use this pipeline in your research, please cite:

```
ASDRP Emotion Detection Pipeline
https://github.com/your-repo/emotion-detector
```

## Support

For issues and questions:
- Check documentation
- Review example scripts
- Open an issue on GitHub

## Roadmap

Future enhancements:

- [ ] CNN-based emotion analyzer
- [ ] Hybrid analyzer combining geometry and CNN
- [ ] Multi-threaded batch processing
- [ ] GPU acceleration
- [ ] Additional emotion categories
- [ ] Emotion intensity estimation
- [ ] Microexpression detection
- [ ] Integration with other face detectors
