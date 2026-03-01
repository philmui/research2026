# Emotion Detection Pipeline Usage Guide

This guide provides comprehensive documentation for using the `EmotionDetectionPipeline` class, the main orchestration component of the ASDRP emotion detection system.

## Table of Contents

1. [Overview](#overview)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Pipeline Configuration](#pipeline-configuration)
5. [Processing Modes](#processing-modes)
6. [Advanced Usage](#advanced-usage)
7. [API Reference](#api-reference)

## Overview

The `EmotionDetectionPipeline` class provides a high-level interface for detecting and analyzing emotions from video files and real-time camera streams. It orchestrates the entire workflow including:

- Video reading and frame extraction
- Face detection and landmark extraction (MediaPipe)
- Emotion classification (geometry-based FACS)
- Temporal smoothing and tracking
- Visualization and annotation
- Results export (JSON, CSV)
- Progress tracking and logging

## Installation

### Prerequisites

```bash
pip install opencv-python mediapipe numpy tqdm
```

### Model Download

Download the MediaPipe Face Landmarker model:

```bash
mkdir -p models
cd models
wget https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Quick Start

### Basic Video Processing

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

### Real-time Camera Processing

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

# Create configuration for real-time processing
config = PipelineConfig.for_realtime_processing(
    model_path="models/face_landmarker.task",
    input_path="0"  # Use default webcam
)

# Process camera stream
with EmotionDetectionPipeline(config) as pipeline:
    for result in pipeline.process_stream(max_frames=300, display=True):
        if result["faces"]:
            emotion = result["faces"][0]["emotion"]
            confidence = result["faces"][0]["confidence"]
            print(f"{emotion}: {confidence:.2f}")
```

## Pipeline Configuration

### Configuration Presets

The pipeline provides three configuration presets for common use cases:

#### 1. Default Configuration

```python
config = PipelineConfig.from_defaults(
    model_path="models/face_landmarker.task",
    input_path="input.mp4",
    output_path="output.mp4"
)
```

Suitable for general video processing with standard settings.

#### 2. Real-time Processing

```python
config = PipelineConfig.for_realtime_processing(
    model_path="models/face_landmarker.task",
    input_path="0"  # Camera device ID
)
```

Optimized for low-latency camera processing with:
- VIDEO mode for tracking
- Smaller smoothing windows
- Real-time display enabled
- No video output by default

#### 3. Batch Processing

```python
config = PipelineConfig.for_batch_processing(
    model_path="models/face_landmarker.task",
    input_path="input.mp4",
    output_path="output.mp4",
    batch_size=8
)
```

Optimized for throughput with:
- Larger batch sizes
- Multiple worker threads
- Both JSON and CSV output

#### 4. Analysis Only

```python
config = PipelineConfig.for_analysis_only(
    model_path="models/face_landmarker.task",
    input_path="input.mp4"
)
```

For data extraction without visualization:
- No annotated video output
- Saves landmarks and emotions
- CSV format by default

### Custom Configuration

For fine-grained control, create a custom configuration:

```python
from asdrp import (
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
        min_tracking_confidence=0.7,
        num_faces=2,  # Detect up to 2 faces
        running_mode="VIDEO",
        enable_smoothing=True,
        smoothing_window_size=5
    ),
    emotion_analysis=EmotionAnalysisConfig(
        analyzer_type="geometric",
        confidence_threshold=0.6,
        enable_temporal_smoothing=True,
        smoothing_window_size=10
    ),
    video=VideoConfig(
        input_path="input.mp4",
        output_path="output.mp4",
        start_frame=0,
        end_frame=300,  # Process first 300 frames only
        skip_frames=1,  # Process every other frame
        display_realtime=False
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
        font_scale=0.8
    ),
    output_format="both",  # Save both JSON and CSV
    save_annotated_video=True,
    save_emotions=True,
    log_level="INFO"
)
```

## Processing Modes

### 1. Video File Processing

Process an entire video file:

```python
with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video(show_progress=True)

    # Results is a list of dictionaries, one per frame
    print(f"Processed {len(results)} frames")

    # Save results
    pipeline.save_results("results.json")
```

### 2. Camera Stream Processing

Process real-time camera stream:

```python
with EmotionDetectionPipeline(config) as pipeline:
    for result in pipeline.process_stream(camera_id=0, max_frames=None, display=True):
        # Process each frame result
        for face in result["faces"]:
            print(f"Emotion: {face['emotion']}, Confidence: {face['confidence']:.2f}")

        # Break on some condition
        if some_condition:
            break
```

### 3. Single Frame Processing

Process individual frames:

```python
from asdrp import VideoFileReader

with EmotionDetectionPipeline(config) as pipeline:
    with VideoFileReader("input.mp4") as reader:
        # Get specific frame
        frame_data = reader.get_frame_at(100)

        # Process single frame
        result = pipeline.process_frame(frame_data, visualize=True)

        print(f"Detected {len(result['faces'])} faces")
```

## Advanced Usage

### Frame Range Selection

Process specific frame ranges:

```python
config.video.start_frame = 100  # Start at frame 100
config.video.end_frame = 500    # End at frame 500
config.video.skip_frames = 2    # Process every 3rd frame
config.video.max_frames = 100   # Process maximum 100 frames
```

### Multiple Faces

Detect and analyze multiple faces per frame:

```python
config.face_detection.num_faces = 3  # Detect up to 3 faces

with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video()

    for result in results:
        for face in result["faces"]:
            print(f"Face {face['face_id']}: {face['emotion']}")
```

### Temporal Smoothing

Enable temporal smoothing for more stable predictions:

```python
config.emotion_analysis.enable_temporal_smoothing = True
config.emotion_analysis.smoothing_window_size = 10  # Use 10-frame window
```

### Custom Visualization

Customize visualization appearance:

```python
config.visualization.landmark_color = (0, 255, 0)     # Green landmarks
config.visualization.bbox_color = (255, 0, 0)         # Blue bounding box
config.visualization.text_color = (255, 255, 255)     # White text
config.visualization.font_scale = 1.0                 # Larger text
config.visualization.landmark_radius = 3              # Larger landmark points
config.visualization.show_timestamp = True            # Show timestamp
```

### Output Formats

Control output format:

```python
# JSON only
config.output_format = "json"
pipeline.save_results("results.json")

# CSV only
config.output_format = "csv"
pipeline.save_results("results.csv")

# Both formats
config.output_format = "both"
pipeline.save_results("results.json")  # Creates both .json and .csv
```

### Result Analysis

Analyze accumulated results:

```python
with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video()

    # Emotion distribution
    emotion_counts = {}
    for result in results:
        for face in result["faces"]:
            emotion = face["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    # Average confidence
    confidences = [face["confidence"] for result in results for face in result["faces"]]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    print(f"Emotion distribution: {emotion_counts}")
    print(f"Average confidence: {avg_confidence:.2f}")
```

## API Reference

### EmotionDetectionPipeline

Main pipeline class for emotion detection.

#### Constructor

```python
EmotionDetectionPipeline(config: PipelineConfig)
```

**Parameters:**
- `config`: PipelineConfig object containing all settings

**Raises:**
- `PipelineError`: If initialization fails

#### Methods

##### process_video()

```python
process_video(show_progress: bool = True) -> List[Dict[str, Any]]
```

Process entire video file.

**Parameters:**
- `show_progress`: Whether to display progress bar

**Returns:**
- List of result dictionaries, one per processed frame

**Raises:**
- `PipelineError`: If processing fails

##### process_stream()

```python
process_stream(
    camera_id: int = 0,
    max_frames: Optional[int] = None,
    display: bool = True
) -> Iterator[Dict[str, Any]]
```

Process real-time camera stream.

**Parameters:**
- `camera_id`: Camera device ID (0 for default)
- `max_frames`: Maximum frames to process (None for unlimited)
- `display`: Whether to display frames in real-time

**Yields:**
- Result dictionary for each processed frame

**Raises:**
- `PipelineError`: If processing fails

##### process_frame()

```python
process_frame(
    frame_data: FrameData,
    visualize: bool = True
) -> Dict[str, Any]
```

Process a single frame.

**Parameters:**
- `frame_data`: FrameData containing the frame
- `visualize`: Whether to apply visualization overlay

**Returns:**
- Result dictionary containing:
  - `frame_number`: Frame number
  - `timestamp`: Frame timestamp
  - `faces`: List of face detection results
  - `annotated_frame`: Optional annotated frame (if visualize=True)

**Raises:**
- `PipelineError`: If processing fails

##### save_results()

```python
save_results(output_path: str | Path) -> None
```

Save processing results to file.

**Parameters:**
- `output_path`: Path to output file (.json or .csv)

**Raises:**
- `PipelineError`: If saving fails or results are empty

##### get_results()

```python
get_results() -> List[Dict[str, Any]]
```

Get accumulated processing results.

**Returns:**
- List of result dictionaries from all processed frames

##### close()

```python
close() -> None
```

Close pipeline and release all resources.

### Result Structure

Each result dictionary contains:

```python
{
    "frame_number": int,        # Frame number in sequence
    "timestamp": float,         # Timestamp in seconds
    "faces": [                  # List of detected faces
        {
            "face_id": int,                    # Face identifier
            "emotion": str,                    # Detected emotion
            "confidence": float,               # Confidence score (0-1)
            "probabilities": {                 # All emotion probabilities
                "neutral": float,
                "happy": float,
                "sad": float,
                "angry": float,
                "surprised": float,
                "fearful": float
            },
            "bounding_box": {                  # Face bounding box (optional)
                "x_min": float,                # Normalized coordinates
                "y_min": float,
                "width": float,
                "height": float
            }
        }
    ]
}
```

### Supported Emotions

The pipeline detects the following emotions:

- `neutral`: Neutral/no emotion
- `happy`: Happiness, joy
- `sad`: Sadness
- `angry`: Anger, frustration
- `surprised`: Surprise, shock
- `fearful`: Fear, anxiety
- `disgusted`: Disgust

## Error Handling

The pipeline provides comprehensive error handling:

```python
from asdrp import EmotionDetectionPipeline, PipelineError

try:
    config = PipelineConfig.from_defaults(
        model_path="models/face_landmarker.task",
        input_path="input.mp4"
    )

    with EmotionDetectionPipeline(config) as pipeline:
        results = pipeline.process_video()
        pipeline.save_results("results.json")

except PipelineError as e:
    print(f"Pipeline error: {e}")
except FileNotFoundError as e:
    print(f"File not found: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

## Performance Tips

1. **Frame Skipping**: Skip frames for faster processing
   ```python
   config.video.skip_frames = 2  # Process every 3rd frame
   ```

2. **Reduce Resolution**: Process at lower resolution
   ```python
   config.video.resolution = (640, 480)
   ```

3. **Disable Visualization**: Skip visualization for faster processing
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

## Logging

The pipeline provides configurable logging:

```python
config.log_level = "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

Log messages include:
- Component initialization
- Processing progress
- Error details
- Resource cleanup

## Examples

See `examples/pipeline_example.py` for comprehensive examples of:
- Basic video processing
- Real-time camera processing
- Batch processing
- Analysis-only mode
- Custom configuration
- Single frame processing

## Troubleshooting

### Common Issues

1. **Model not found**
   - Download the MediaPipe model from the URL above
   - Ensure model path is correct in configuration

2. **Camera not opening**
   - Check camera device ID (usually 0 for default camera)
   - Ensure camera permissions are granted
   - Try different camera IDs if multiple cameras are available

3. **Out of memory**
   - Reduce batch size
   - Enable frame skipping
   - Disable video output
   - Process video in chunks

4. **Slow processing**
   - Skip frames
   - Reduce resolution
   - Disable visualization
   - Use batch processing mode

## Support

For issues and questions, please refer to:
- Project documentation
- Example scripts
- Source code comments
