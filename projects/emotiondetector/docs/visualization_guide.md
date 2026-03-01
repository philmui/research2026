# Visualization Module Guide

This guide provides comprehensive documentation for the visualization module of the ASDRP Emotion Detector project.

## Overview

The visualization module provides tools for displaying emotion detection results in various formats:

- **Face Overlay**: Draw facial landmarks, connections, and bounding boxes on video frames
- **Emotion Display**: Show emotion labels, confidence scores, and probability distributions
- **Statistical Plots**: Create charts and graphs for analyzing emotion data
- **Heatmaps**: Generate temporal heatmaps and transition matrices

## Module Structure

```
asdrp/visualization/
├── __init__.py           # Module exports
├── overlay.py            # Face landmark overlay visualization
├── emotion_display.py    # Emotion information display
├── plots.py              # Statistical plotting functions
└── heatmap.py            # Temporal heatmap generation
```

## 1. Face Overlay (`overlay.py`)

### FaceOverlay Class

The `FaceOverlay` class provides methods to draw facial landmarks and annotations on images.

#### Basic Usage

```python
from asdrp.visualization import FaceOverlay, OverlayStyle
import cv2

# Create overlay with default style
overlay = FaceOverlay()

# Or with custom style
style = OverlayStyle(
    landmark_color=(0, 255, 0),      # Green landmarks
    landmark_radius=3,
    connection_color=(255, 255, 255), # White connections
    connection_thickness=1,
    bbox_color=(255, 0, 0),          # Blue bounding box
    bbox_thickness=2
)
overlay = FaceOverlay(style=style)

# Read image
image = cv2.imread("frame.jpg")

# Draw complete face visualization
overlay.draw_complete_face(image, face_landmarks)

# Save result
cv2.imwrite("output.jpg", image)
```

#### Available Methods

- **`draw_landmarks()`**: Draw all facial landmarks as points
- **`draw_landmark_group()`**: Draw specific landmark groups (eyes, eyebrows, mouth, etc.)
- **`draw_connections()`**: Draw lines connecting landmarks
- **`draw_eyes()`**: Draw eye landmarks and connections
- **`draw_eyebrows()`**: Draw eyebrow landmarks and connections
- **`draw_mouth()`**: Draw mouth landmarks and connections
- **`draw_face_oval()`**: Draw face oval outline
- **`draw_bounding_box()`**: Draw bounding box with optional label
- **`draw_complete_face()`**: Draw all face features at once

#### Landmark Groups

The following landmark groups are available:
- `left_eye`: Left eye contour (6 landmarks)
- `right_eye`: Right eye contour (6 landmarks)
- `left_eyebrow`: Left eyebrow (5 landmarks)
- `right_eyebrow`: Right eyebrow (5 landmarks)
- `nose`: Nose landmarks (7 landmarks)
- `mouth_outer`: Outer mouth contour (8 landmarks)
- `mouth_inner`: Inner mouth contour (6 landmarks)
- `face_oval`: Face outline (12 landmarks)

#### OverlayStyle Configuration

```python
@dataclass
class OverlayStyle:
    landmark_color: tuple[int, int, int] = (0, 255, 0)
    landmark_radius: int = 2
    connection_color: tuple[int, int, int] = (255, 255, 255)
    connection_thickness: int = 1
    bbox_color: tuple[int, int, int] = (255, 0, 0)
    bbox_thickness: int = 2
    fill_landmarks: bool = True
    draw_indices: bool = False
    text_color: tuple[int, int, int] = (0, 255, 255)
    text_scale: float = 0.3
    text_thickness: int = 1
```

## 2. Emotion Display (`emotion_display.py`)

### EmotionDisplay Class

The `EmotionDisplay` class visualizes emotion predictions on video frames.

#### Basic Usage

```python
from asdrp.visualization import EmotionDisplay, DisplayStyle

# Create display with default style
display = EmotionDisplay()

# Or with custom style
style = DisplayStyle(
    text_scale=0.8,
    show_probabilities=True,
    show_confidence=True,
    show_action_units=True,
    position="top_left"  # Options: top_left, top_right, bottom_left, bottom_right
)
display = EmotionDisplay(style=style)

# Draw emotion information
display.draw_complete_display(image, emotion_prediction, top_n_emotions=3)
```

#### Available Methods

- **`draw_emotion_label()`**: Draw primary emotion label with confidence
- **`draw_probability_bars()`**: Draw probability bars for all/top-N emotions
- **`draw_action_units()`**: Draw detected action units with intensities
- **`draw_complete_display()`**: Draw all emotion information
- **`draw_timeline_marker()`**: Draw emotion timeline at bottom/top of frame
- **`create_emotion_indicator()`**: Create colored square indicator for an emotion

#### Emotion Color Scheme

The module uses a consistent color scheme (BGR format):
- **Neutral**: Gray (200, 200, 200)
- **Happy**: Yellow (0, 255, 255)
- **Sad**: Blue (255, 0, 0)
- **Angry**: Red (0, 0, 255)
- **Surprised**: Orange (255, 128, 0)
- **Fearful**: Purple (128, 0, 128)
- **Disgusted**: Dark Green (0, 128, 0)

#### DisplayStyle Configuration

```python
@dataclass
class DisplayStyle:
    text_color: tuple[int, int, int] = (255, 255, 255)
    text_scale: float = 0.7
    text_thickness: int = 2
    bar_height: int = 20
    bar_width: int = 200
    bar_spacing: int = 5
    background_alpha: float = 0.6
    show_probabilities: bool = True
    show_confidence: bool = True
    show_action_units: bool = False
    position: str = "top_left"
```

## 3. Statistical Plots (`plots.py`)

### Plotting Functions

The module provides several functions for creating statistical visualizations using matplotlib.

#### plot_emotion_distribution()

Create bar chart or pie chart showing emotion distribution.

```python
from asdrp.visualization import plot_emotion_distribution

fig = plot_emotion_distribution(
    emotion_predictions,
    plot_type="bar",  # or "pie"
    title="Emotion Distribution",
    figsize=(10, 6),
    save_path="emotion_dist.png"
)
```

#### plot_emotion_timeline()

Plot emotion changes over time as a timeline.

```python
from asdrp.visualization import plot_emotion_timeline

fig = plot_emotion_timeline(
    emotion_predictions,
    title="Emotion Timeline",
    figsize=(14, 6),
    save_path="timeline.png",
    use_frame_numbers=True
)
```

#### plot_confidence_over_time()

Plot confidence scores over time with optional moving average.

```python
from asdrp.visualization import plot_confidence_over_time

fig = plot_confidence_over_time(
    emotion_predictions,
    show_moving_average=True,
    window_size=10,
    save_path="confidence.png"
)
```

#### plot_action_units()

Plot action unit activations over time.

```python
from asdrp.visualization import plot_action_units

fig = plot_action_units(
    emotion_predictions,
    au_types=[ActionUnitType.AU12, ActionUnitType.AU6],  # Optional: specific AUs
    save_path="action_units.png"
)
```

#### plot_emotion_transitions()

Create transition matrix heatmap showing emotion transitions.

```python
from asdrp.visualization import plot_emotion_transitions

fig = plot_emotion_transitions(
    emotion_predictions,
    normalize=True,  # Convert counts to probabilities
    save_path="transitions.png"
)
```

#### plot_emotion_probabilities_over_time()

Plot probability distributions over time.

```python
from asdrp.visualization import plot_emotion_probabilities_over_time

fig = plot_emotion_probabilities_over_time(
    emotion_predictions,
    emotions_to_plot=[EmotionType.HAPPY, EmotionType.SAD],  # Optional
    stacked=False,  # Set True for stacked area plot
    save_path="probabilities.png"
)
```

#### plot_emotion_summary()

Create comprehensive summary with multiple subplots.

```python
from asdrp.visualization import plot_emotion_summary

fig = plot_emotion_summary(
    emotion_predictions,
    title="Emotion Analysis Summary",
    figsize=(16, 10),
    save_path="summary.png"
)
```

## 4. Heatmaps (`heatmap.py`)

### EmotionHeatmap Class

The `EmotionHeatmap` class creates heatmap visualizations for temporal analysis.

#### Basic Usage

```python
from asdrp.visualization import EmotionHeatmap

# Create heatmap generator
heatmap = EmotionHeatmap(cmap="YlOrRd")

# Create temporal heatmap
fig = heatmap.create_temporal_heatmap(
    emotion_predictions,
    window_size=10,
    save_path="temporal_heatmap.png"
)
```

#### Available Methods

##### create_temporal_heatmap()

Create heatmap showing emotion intensities over time windows.

```python
fig = heatmap.create_temporal_heatmap(
    emotion_predictions,
    window_size=10,  # Frames per window
    title="Emotion Intensity Over Time",
    figsize=(14, 6),
    save_path="temporal.png"
)
```

##### create_transition_heatmap()

Create heatmap showing emotion transitions.

```python
fig = heatmap.create_transition_heatmap(
    emotion_predictions,
    normalize=True,  # Show probabilities instead of counts
    save_path="transitions.png"
)
```

##### create_correlation_heatmap()

Show correlations between emotion probabilities.

```python
fig = heatmap.create_correlation_heatmap(
    emotion_predictions,
    save_path="correlations.png"
)
```

##### create_intensity_matrix()

Compare emotion intensities across multiple sessions.

```python
fig = heatmap.create_intensity_matrix(
    [session1_predictions, session2_predictions, session3_predictions],
    session_labels=["Session 1", "Session 2", "Session 3"],
    save_path="intensity_matrix.png"
)
```

##### create_sliding_window_heatmap()

Create heatmap using sliding window analysis.

```python
fig = heatmap.create_sliding_window_heatmap(
    emotion_predictions,
    window_size=30,
    stride=10,
    save_path="sliding_window.png"
)
```

## Complete Example

Here's a complete example combining multiple visualization components:

```python
import cv2
from pathlib import Path
from asdrp.face.landmarker import FaceLandmarker
from asdrp.emotion.geometry_analyzer import GeometryBasedEmotionAnalyzer
from asdrp.visualization import (
    FaceOverlay,
    EmotionDisplay,
    plot_emotion_summary,
    EmotionHeatmap
)

# Initialize components
landmarker = FaceLandmarker(model_path="models/face_landmarker.task")
analyzer = GeometryBasedEmotionAnalyzer()
overlay = FaceOverlay()
display = EmotionDisplay()

# Process video
cap = cv2.VideoCapture("video.mp4")
predictions = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect face and landmarks
    results = landmarker.detect(frame)
    if not results:
        continue

    face_landmarks = results[0]

    # Analyze emotion
    prediction = analyzer.analyze(face_landmarks)
    predictions.append(prediction)

    # Visualize on frame
    overlay.draw_complete_face(frame, face_landmarks)
    display.draw_complete_display(frame, prediction)

    cv2.imshow("Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Create statistical plots
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# Summary plot
plot_emotion_summary(predictions, save_path=output_dir / "summary.png")

# Temporal heatmap
heatmap = EmotionHeatmap()
heatmap.create_temporal_heatmap(predictions, save_path=output_dir / "heatmap.png")
```

## Best Practices

### Performance Optimization

1. **Batch Processing**: Process multiple frames before visualization
2. **Selective Drawing**: Only draw necessary components for your use case
3. **Resolution Management**: Scale down images for real-time processing

### Visual Quality

1. **Color Consistency**: Use the provided emotion color scheme for consistency
2. **Text Readability**: Adjust text scale and thickness based on image resolution
3. **Overlay Transparency**: Use semi-transparent backgrounds for better readability

### Analysis Workflow

1. **Real-time Visualization**: Use overlay and display for live feedback
2. **Post-processing Analysis**: Use plots and heatmaps for detailed analysis
3. **Report Generation**: Combine multiple visualizations in summary plots

## Customization

### Custom Color Schemes

```python
from asdrp.visualization import EMOTION_COLORS
from asdrp.emotion.base import EmotionType

# Override default colors
EMOTION_COLORS[EmotionType.HAPPY] = (0, 200, 255)  # Custom orange
```

### Custom Styles

```python
# Create reusable style presets
minimal_style = OverlayStyle(
    landmark_radius=1,
    connection_thickness=1,
    draw_indices=False
)

detailed_style = OverlayStyle(
    landmark_radius=3,
    connection_thickness=2,
    draw_indices=True
)
```

### Custom Plots

You can extend the plotting functions or create custom visualizations:

```python
import matplotlib.pyplot as plt

def custom_plot(predictions):
    fig, ax = plt.subplots()
    # Your custom plotting code here
    return fig
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed (opencv-python, matplotlib, seaborn)
2. **Color Issues**: OpenCV uses BGR format, matplotlib uses RGB
3. **Memory Usage**: For large datasets, process in batches or use lower resolution

### Performance Tips

- Use `cv2.resize()` to scale images before processing
- Disable unnecessary visualizations in production
- Save plots as PNG for better quality, JPG for smaller size

## API Reference

For detailed API documentation, see the docstrings in each module file:

- `overlay.py`: Face landmark overlay visualization
- `emotion_display.py`: Emotion information display
- `plots.py`: Statistical plotting functions
- `heatmap.py`: Temporal heatmap generation

## Examples

See `examples/visualization_demo.py` for comprehensive demonstrations of all visualization features.
