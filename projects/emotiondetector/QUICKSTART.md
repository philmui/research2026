# Quick Start Guide - Emotion Detection Pipeline

Get started with the Emotion Detection Pipeline in 5 minutes.

## Prerequisites

```bash
# Install dependencies
pip install opencv-python mediapipe numpy tqdm

# Download model (25MB)
mkdir -p models
wget -P models/ https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## 30-Second Demo

Process a video file with 3 lines of code:

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

config = PipelineConfig.from_defaults(
    model_path="models/face_landmarker.task",
    input_path="your_video.mp4",
    output_path="output.mp4"
)

with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video()
    pipeline.save_results("results.json")
```

## Common Use Cases

### 1. Process a Video File

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

config = PipelineConfig.from_defaults(
    model_path="models/face_landmarker.task",
    input_path="input.mp4",
    output_path="output.mp4"
)

with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video(show_progress=True)
    pipeline.save_results("results.json")

print(f"Processed {len(results)} frames")
```

### 2. Use Your Webcam

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

config = PipelineConfig.for_realtime_processing(
    model_path="models/face_landmarker.task",
    input_path="0"  # 0 = default webcam
)

with EmotionDetectionPipeline(config) as pipeline:
    for result in pipeline.process_stream(max_frames=300, display=True):
        if result["faces"]:
            print(f"Emotion: {result['faces'][0]['emotion']}")
```

Press 'q' to quit the camera view.

### 3. Fast Processing (Skip Frames)

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

config = PipelineConfig.from_defaults(
    model_path="models/face_landmarker.task",
    input_path="long_video.mp4"
)

# Process every 3rd frame for 3x speedup
config.video.skip_frames = 2
config.save_annotated_video = False  # Skip video output

with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video()
    pipeline.save_results("results.csv")
```

### 4. Extract Data Only (No Video)

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

### 5. Detect Multiple Faces

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

config = PipelineConfig.from_defaults(
    model_path="models/face_landmarker.task",
    input_path="group_video.mp4"
)

# Detect up to 3 faces per frame
config.face_detection.num_faces = 3

with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video()

    # Print results for each face
    for result in results:
        print(f"\nFrame {result['frame_number']}:")
        for face in result["faces"]:
            print(f"  Face {face['face_id']}: {face['emotion']} ({face['confidence']:.2f})")
```

### 6. Custom Visualization

```python
from asdrp import EmotionDetectionPipeline, PipelineConfig

config = PipelineConfig.from_defaults(
    model_path="models/face_landmarker.task",
    input_path="input.mp4",
    output_path="output.mp4"
)

# Customize appearance
config.visualization.landmark_color = (0, 255, 0)     # Green
config.visualization.bbox_color = (255, 0, 0)         # Blue
config.visualization.text_color = (255, 255, 255)     # White
config.visualization.show_timestamp = True
config.visualization.font_scale = 1.0

with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video()
```

## Understanding Results

Each result contains:

```python
{
    "frame_number": 42,
    "timestamp": 1.4,
    "faces": [
        {
            "face_id": 0,
            "emotion": "happy",              # Detected emotion
            "confidence": 0.85,              # Confidence (0-1)
            "probabilities": {               # All emotion scores
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

## Supported Emotions

- `neutral`: No strong emotion
- `happy`: Joy, happiness, smiling
- `sad`: Sadness, sorrow
- `angry`: Anger, frustration
- `surprised`: Surprise, shock
- `fearful`: Fear, anxiety

## Output Files

### JSON Format

Detailed results with all emotion probabilities:

```bash
pipeline.save_results("results.json")
```

### CSV Format

Tabular format for analysis:

```bash
pipeline.save_results("results.csv")
```

### Both Formats

Save both JSON and CSV:

```python
config.output_format = "both"
pipeline.save_results("results.json")  # Creates results.json AND results.csv
```

## Configuration Presets

### Default Configuration
General-purpose video processing:
```python
config = PipelineConfig.from_defaults(model_path, input_path, output_path)
```

### Real-time Processing
Optimized for webcam with low latency:
```python
config = PipelineConfig.for_realtime_processing(model_path, camera_id)
```

### Batch Processing
High throughput with parallel processing:
```python
config = PipelineConfig.for_batch_processing(model_path, input_path, output_path, batch_size=16)
```

### Analysis Only
Data extraction without video output:
```python
config = PipelineConfig.for_analysis_only(model_path, input_path)
```

## Troubleshooting

### Problem: Model not found
```
FileNotFoundError: Model file not found
```

**Solution**: Download the model:
```bash
mkdir -p models
wget -P models/ https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Problem: Camera not opening
```
CameraCaptureError: Failed to open camera
```

**Solutions**:
1. Check camera permissions
2. Try different camera ID: `input_path="1"` instead of `"0"`
3. Make sure no other app is using the camera

### Problem: Processing too slow
**Solutions**:
1. Skip frames: `config.video.skip_frames = 2`
2. Lower resolution: `config.video.resolution = (640, 480)`
3. Disable video output: `config.save_annotated_video = False`
4. Disable visualization: `config.visualization.draw_landmarks = False`

### Problem: Out of memory
**Solutions**:
1. Skip frames to process fewer frames
2. Disable annotated video output
3. Process video in chunks using `start_frame` and `end_frame`

## Next Steps

1. Check out `examples/pipeline_example.py` for more examples
2. Read `PIPELINE_USAGE.md` for comprehensive documentation
3. See `README_PIPELINE.md` for architecture details
4. Customize configuration for your use case

## Help

For more information:
- Full documentation: `PIPELINE_USAGE.md`
- API reference: Docstrings in source code
- Examples: `examples/` directory
- Tests: `tests/test_pipeline.py`

## Simple Analysis Script

Save this as `analyze_video.py`:

```python
#!/usr/bin/env python3
"""Simple script to analyze a video file."""
import sys
from asdrp import EmotionDetectionPipeline, PipelineConfig

if len(sys.argv) != 2:
    print("Usage: python analyze_video.py <video_file>")
    sys.exit(1)

video_file = sys.argv[1]
output_file = video_file.replace('.mp4', '_output.mp4')
results_file = video_file.replace('.mp4', '_results.json')

config = PipelineConfig.from_defaults(
    model_path="models/face_landmarker.task",
    input_path=video_file,
    output_path=output_file
)

print(f"Processing: {video_file}")
with EmotionDetectionPipeline(config) as pipeline:
    results = pipeline.process_video(show_progress=True)
    pipeline.save_results(results_file)

print(f"\nDone!")
print(f"Output video: {output_file}")
print(f"Results: {results_file}")
print(f"Processed {len(results)} frames")

# Print emotion summary
emotions = {}
for r in results:
    for face in r["faces"]:
        e = face["emotion"]
        emotions[e] = emotions.get(e, 0) + 1

print("\nEmotion Distribution:")
for emotion, count in sorted(emotions.items(), key=lambda x: x[1], reverse=True):
    pct = count / len(results) * 100
    print(f"  {emotion}: {count} frames ({pct:.1f}%)")
```

Run it:
```bash
python analyze_video.py my_video.mp4
```
