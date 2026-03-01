# Visualization Module Quick Reference

## Quick Start

```python
from asdrp.visualization import FaceOverlay, EmotionDisplay, EmotionHeatmap
from asdrp.visualization import plot_emotion_summary
```

## FaceOverlay - Draw Landmarks

```python
# Initialize
overlay = FaceOverlay()

# Draw everything
overlay.draw_complete_face(image, face_landmarks)

# Draw specific features
overlay.draw_eyes(image, face_landmarks)
overlay.draw_mouth(image, face_landmarks)
overlay.draw_bounding_box(image, bbox, label="Face")

# Custom style
from asdrp.visualization import OverlayStyle
style = OverlayStyle(
    landmark_color=(0, 255, 0),  # Green
    landmark_radius=3,
    connection_color=(255, 255, 255),
    bbox_color=(255, 0, 0)
)
overlay = FaceOverlay(style=style)
```

## EmotionDisplay - Show Emotions

```python
# Initialize
display = EmotionDisplay()

# Draw everything
display.draw_complete_display(image, emotion_prediction)

# Draw components separately
display.draw_emotion_label(image, emotion_prediction)
display.draw_probability_bars(image, emotion_prediction, top_n=3)
display.draw_action_units(image, emotion_prediction)

# Custom style
from asdrp.visualization import DisplayStyle
style = DisplayStyle(
    text_scale=0.8,
    show_probabilities=True,
    show_confidence=True,
    position="top_left"  # or top_right, bottom_left, bottom_right
)
display = EmotionDisplay(style=style)
```

## Statistical Plots

```python
from asdrp.visualization import (
    plot_emotion_distribution,
    plot_emotion_timeline,
    plot_confidence_over_time,
    plot_emotion_summary
)

# Emotion distribution (bar or pie)
plot_emotion_distribution(predictions, plot_type="bar", save_path="dist.png")

# Timeline
plot_emotion_timeline(predictions, save_path="timeline.png")

# Confidence over time
plot_confidence_over_time(predictions, save_path="confidence.png")

# Complete summary
plot_emotion_summary(predictions, save_path="summary.png")
```

## Heatmaps

```python
heatmap = EmotionHeatmap()

# Temporal heatmap
heatmap.create_temporal_heatmap(predictions, window_size=10, save_path="temporal.png")

# Transition matrix
heatmap.create_transition_heatmap(predictions, normalize=True, save_path="transitions.png")

# Correlation matrix
heatmap.create_correlation_heatmap(predictions, save_path="correlation.png")

# Sliding window
heatmap.create_sliding_window_heatmap(
    predictions,
    window_size=30,
    stride=10,
    save_path="sliding.png"
)
```

## Complete Pipeline Example

```python
import cv2
from asdrp.face.landmarker import FaceLandmarker
from asdrp.emotion.geometry_analyzer import GeometryBasedEmotionAnalyzer
from asdrp.visualization import FaceOverlay, EmotionDisplay, plot_emotion_summary

# Initialize
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

    # Detect and analyze
    results = landmarker.detect(frame)
    if results:
        face_landmarks = results[0]
        prediction = analyzer.analyze(face_landmarks)
        predictions.append(prediction)

        # Visualize
        overlay.draw_complete_face(frame, face_landmarks)
        display.draw_complete_display(frame, prediction)

        cv2.imshow("Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Generate summary
plot_emotion_summary(predictions, save_path="summary.png")
```

## Emotion Colors (BGR)

```python
from asdrp.visualization import EMOTION_COLORS
from asdrp.emotion.base import EmotionType

EMOTION_COLORS = {
    EmotionType.NEUTRAL: (200, 200, 200),    # Gray
    EmotionType.HAPPY: (0, 255, 255),        # Yellow
    EmotionType.SAD: (255, 0, 0),            # Blue
    EmotionType.ANGRY: (0, 0, 255),          # Red
    EmotionType.SURPRISED: (255, 128, 0),    # Orange
    EmotionType.FEARFUL: (128, 0, 128),      # Purple
    EmotionType.DISGUSTED: (0, 128, 0),      # Dark Green
}
```

## Common Patterns

### Real-Time Processing
```python
# Minimal overhead for real-time
overlay.draw_eyes(frame, landmarks)
overlay.draw_mouth(frame, landmarks)
display.draw_emotion_label(frame, prediction)
```

### High-Quality Output
```python
# All features for saved output
overlay.draw_complete_face(frame, landmarks,
                          draw_all_landmarks=True,
                          draw_all_connections=True)
display.draw_complete_display(frame, prediction,
                              show_bars=True,
                              show_aus=True)
```

### Batch Analysis
```python
# Process all frames first, then analyze
predictions = []
for frame in frames:
    # ... detection and analysis ...
    predictions.append(prediction)

# Generate all plots
plot_emotion_distribution(predictions, save_path="dist.png")
plot_emotion_timeline(predictions, save_path="timeline.png")
plot_confidence_over_time(predictions, save_path="confidence.png")

heatmap = EmotionHeatmap()
heatmap.create_temporal_heatmap(predictions, save_path="temporal.png")
heatmap.create_transition_heatmap(predictions, save_path="transitions.png")
```

## Tips

1. **Performance**: Draw only what you need for real-time processing
2. **Quality**: Use 300 DPI for publication-quality plots
3. **Colors**: OpenCV uses BGR, matplotlib uses RGB
4. **Memory**: Process large videos in batches
5. **Customization**: Create reusable style objects for consistency

## Common Issues

**Import Error**: Install dependencies
```bash
pip install opencv-python matplotlib seaborn numpy
```

**Color Mismatch**: Remember OpenCV uses BGR
```python
# Convert if needed
rgb_color = (b, g, r)  # to BGR
bgr_color = (r, g, b)  # to RGB
```

**Memory Issues**: Process in batches
```python
for batch in batches:
    predictions = process_batch(batch)
    plot_emotion_summary(predictions)
```

## More Information

- Full Documentation: `docs/visualization_guide.md`
- Demo Script: `examples/visualization_demo.py`
- Unit Tests: `tests/test_visualization.py`
- Implementation Summary: `VISUALIZATION_MODULE_SUMMARY.md`
