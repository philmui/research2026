# Emotion Module - Quick Reference

## Import Classes

```python
from asdrp.emotion import (
    # Core
    EmotionType, ActionUnitType, EmotionPrediction, ActionUnit,

    # Analysis
    GeometryBasedEmotionAnalyzer, FeatureExtractor,

    # Temporal
    TemporalEmotionAnalyzer, TemporalFilter,

    # Metrics
    compute_emotion_metrics, compute_emotion_distribution,
)
```

## Emotion Types

```python
EmotionType.NEUTRAL
EmotionType.HAPPY
EmotionType.SAD
EmotionType.ANGRY
EmotionType.SURPRISED
EmotionType.FEARFUL
EmotionType.DISGUSTED
```

## Action Units

| AU | Name | Description |
|----|------|-------------|
| AU1 | Inner Brow Raiser | Inner eyebrows raised |
| AU2 | Outer Brow Raiser | Outer eyebrows raised |
| AU4 | Brow Lowerer | Eyebrows lowered (frown) |
| AU5 | Upper Lid Raiser | Eyes widened |
| AU6 | Cheek Raiser | Cheeks raised (part of smile) |
| AU7 | Lid Tightener | Eyes narrowed/tightened |
| AU9 | Nose Wrinkler | Nose wrinkled (disgust) |
| AU10 | Upper Lip Raiser | Upper lip raised |
| AU12 | Lip Corner Puller | Smile |
| AU15 | Lip Corner Depressor | Frown |
| AU17 | Chin Raiser | Chin raised/pushed up |
| AU20 | Lip Stretcher | Lips stretched horizontally |
| AU23 | Lip Tightener | Lips tightened/pressed |
| AU25 | Lips Part | Lips separated |
| AU26 | Jaw Drop | Mouth opened wide |

## Emotion Rules

| Emotion | Action Units | Weights |
|---------|-------------|---------|
| Happy | AU6 + AU12 | 50% + 50% |
| Sad | AU1 + AU4 + AU15 | 30% + 40% + 30% |
| Angry | AU4 + AU7 + AU23 | 40% + 30% + 30% |
| Surprised | AU1 + AU2 + AU5 + AU26 | 20% + 20% + 30% + 30% |
| Fearful | AU1 + AU2 + AU4 + AU5 + AU20 | 25% + 20% + 15% + 20% + 20% |
| Disgusted | AU9 + AU15 + AU17 | 40% + 30% + 30% |

## Basic Usage

### Single Frame Analysis

```python
# Initialize
analyzer = GeometryBasedEmotionAnalyzer()

# Analyze
prediction = analyzer.analyze(face_landmarks)

# Results
print(prediction.emotion)              # EmotionType
print(prediction.confidence)           # 0.0 - 1.0
print(prediction.probabilities)        # Dict[EmotionType, float]
print(prediction.action_units)         # Dict[ActionUnitType, ActionUnit]
```

### Video Analysis with Smoothing

```python
# Initialize
analyzer = GeometryBasedEmotionAnalyzer()
temporal = TemporalEmotionAnalyzer(window_size=5)

# Process frames
for landmarks in video_landmarks:
    prediction = analyzer.analyze(landmarks)
    smoothed = temporal.smooth_prediction(prediction)
    # Use smoothed prediction
```

### Batch Processing

```python
analyzer = GeometryBasedEmotionAnalyzer()
predictions = analyzer.analyze_batch(landmarks_list)
```

## Configuration

### Analyzer Thresholds

```python
analyzer = GeometryBasedEmotionAnalyzer(
    emotion_threshold=0.3,    # Min confidence for classification
    au_threshold=0.3,         # Min intensity for AU presence
    neutral_threshold=0.2     # Max score for neutral classification
)

# Or adjust later
analyzer.set_emotion_threshold(0.4)
analyzer.set_au_threshold(0.35)
```

### Temporal Settings

```python
temporal = TemporalEmotionAnalyzer(
    window_size=5,                  # Moving average window
    hysteresis_threshold=0.15,      # Confidence diff to change emotion
    min_state_duration=0.5,         # Min duration for stable state (sec)
    microexpression_duration=0.5    # Max duration for microexpression (sec)
)
```

## EmotionPrediction Methods

```python
# Get top N emotions
top_3 = prediction.get_top_emotions(n=3)
for emotion, prob in top_3:
    print(f"{emotion}: {prob:.2f}")

# Get active action units
active = prediction.get_active_action_units(threshold=0.3)
for au in active:
    print(f"{au.au_type}: {au.intensity:.2f}")

# Serialize
pred_dict = prediction.to_dict()
```

## Metrics

### Compute Metrics

```python
metrics = compute_emotion_metrics(predictions)

print(metrics.total_frames)
print(metrics.dominant_emotion)
print(metrics.duration)
print(metrics.fps)
print(metrics.emotion_distribution)
print(metrics.average_confidences)
```

### Other Metrics

```python
# Distribution
dist = compute_emotion_distribution(predictions, normalize=True)

# Transitions
transitions = detect_emotion_transitions(predictions, min_duration=0.5)

# Stability (0-1, higher = more stable)
stability = compute_emotion_stability(predictions)

# Confidence stats
conf_stats = compute_confidence_statistics(predictions)
# Returns: mean, std, min, max, median

# AU statistics
au_stats = compute_au_statistics(predictions)
# Returns: mean, std, max, presence_rate per AU

# Peak emotions
peaks = find_peak_emotions(predictions, EmotionType.HAPPY, top_n=5)

# Timeline
timeline = compute_emotion_timeline(predictions, window_size=1.0)
```

## Temporal Analysis

### Emotion States

```python
# Get detected stable states
states = temporal.get_emotion_states(min_duration=1.0)

for state in states:
    print(f"{state.emotion}: {state.duration:.2f}s")
    print(f"  Frames: {state.start_frame}-{state.end_frame}")
    print(f"  Confidence: {state.average_confidence:.2f}")
```

### Microexpressions

```python
# Detect microexpressions
microexp = temporal.detect_microexpression(prediction)

if microexp:
    print(f"Microexpression: {microexp.emotion}")
    print(f"Intensity: {microexp.intensity:.2f}")
    print(f"Duration: {microexp.duration:.2f}s")

# Get all microexpressions
all_microexps = temporal.get_microexpressions()
angry_microexps = temporal.get_microexpressions(emotion=EmotionType.ANGRY)
```

### Filtering

```python
# Median filter
filtered = TemporalFilter.median_filter(predictions, window_size=5)

# Exponential smoothing
smoothed = TemporalFilter.exponential_smoothing(predictions, alpha=0.3)

# Remove transients
cleaned = TemporalFilter.remove_transients(predictions, min_duration=0.3)
```

## Custom Rules

```python
from asdrp.emotion import EmotionRuleBuilder

builder = EmotionRuleBuilder()

# Add rules
builder.add_rule(EmotionType.HAPPY, ActionUnitType.AU12, weight=0.6)
builder.add_rule(EmotionType.HAPPY, ActionUnitType.AU6, weight=0.4)

# Normalize
builder.normalize_weights(EmotionType.HAPPY)

# Apply
analyzer.EMOTION_AU_RULES = builder.build()
```

## Feature Extraction

```python
extractor = FeatureExtractor(au_threshold=0.3)

# Extract features
features = extractor.extract_features(landmarks)
# Returns dict with:
#   - eye_aspect_ratio
#   - eyebrow_height
#   - mouth_aspect_ratio
#   - mouth_corner_height
#   - etc.

# Detect action units
action_units = extractor.detect_action_units(landmarks)
```

## Complete Example

```python
from asdrp.emotion import (
    GeometryBasedEmotionAnalyzer,
    TemporalEmotionAnalyzer,
    compute_emotion_metrics,
)
from asdrp.face import MediaPipeFaceLandmarker
from asdrp.video import VideoReader

# Initialize
landmarker = MediaPipeFaceLandmarker(model_path="models/face_landmarker.task")
analyzer = GeometryBasedEmotionAnalyzer()
temporal = TemporalEmotionAnalyzer(window_size=5)

# Process video
predictions = []
reader = VideoReader("video.mp4")

for frame in reader:
    # Detect face
    landmarks_list = landmarker.detect(frame.image, timestamp_ms=frame.timestamp)

    if not landmarks_list:
        continue

    # Analyze emotion
    prediction = analyzer.analyze(landmarks_list[0])
    smoothed = temporal.smooth_prediction(prediction)
    predictions.append(smoothed)

    # Display
    print(f"Frame {frame.frame_number}: {smoothed.emotion.value} ({smoothed.confidence:.2f})")

reader.close()

# Analyze results
metrics = compute_emotion_metrics(predictions)
print(f"\nResults:")
print(f"Dominant emotion: {metrics.dominant_emotion}")
print(f"Duration: {metrics.duration:.2f}s")
print(f"\nDistribution:")
for emotion, prob in sorted(metrics.emotion_distribution.items(),
                           key=lambda x: x[1], reverse=True):
    print(f"  {emotion.value}: {prob:.1%}")
```

## Tips

1. **Use temporal smoothing for video** - Reduces jitter and false positives
2. **Adjust thresholds for your use case** - Higher thresholds = more conservative
3. **Check confidence scores** - Low confidence may indicate ambiguous expression
4. **Use batch processing for efficiency** - Process multiple faces at once
5. **Monitor microexpressions** - Can reveal concealed emotions
6. **Analyze emotion transitions** - Understand emotional dynamics
7. **Use metrics for sequence analysis** - Get overall patterns and statistics

## Files Location

```
asdrp/emotion/
├── __init__.py              # Module exports
├── base.py                  # Core classes
├── features.py              # Feature extraction
├── geometry_analyzer.py     # Rule-based analyzer
├── metrics.py               # Statistics
└── temporal.py              # Temporal analysis
```
