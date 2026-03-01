# Emotion Detection Module - Usage Guide

## Overview

The emotion detection module provides a comprehensive system for analyzing facial emotions from video or images. It includes:

1. **Base Classes** - Core data structures and abstract interfaces
2. **Feature Extraction** - Geometric features and Action Unit detection based on FACS
3. **Geometry Analyzer** - Rule-based emotion classification
4. **Metrics** - Statistics and analysis of emotion sequences
5. **Temporal Analysis** - Smoothing and microexpression detection

## Quick Start

```python
from asdrp.emotion import GeometryBasedEmotionAnalyzer, TemporalEmotionAnalyzer
from asdrp.face import MediaPipeFaceLandmarker
import cv2

# Initialize components
landmarker = MediaPipeFaceLandmarker(
    model_path="models/face_landmarker.task"
)
emotion_analyzer = GeometryBasedEmotionAnalyzer()
temporal_analyzer = TemporalEmotionAnalyzer(window_size=5)

# Process a frame
frame = cv2.imread("image.jpg")
landmarks_list = landmarker.detect(frame)

if landmarks_list:
    landmarks = landmarks_list[0]

    # Analyze emotion
    prediction = emotion_analyzer.analyze(landmarks)

    # Apply temporal smoothing (for video)
    smoothed = temporal_analyzer.smooth_prediction(prediction)

    print(f"Emotion: {smoothed.emotion}")
    print(f"Confidence: {smoothed.confidence:.2f}")
```

## Module Components

### 1. Base Classes (`asdrp.emotion.base`)

#### EmotionType
Enumeration of the seven basic emotions:
- `NEUTRAL`
- `HAPPY`
- `SAD`
- `ANGRY`
- `SURPRISED`
- `FEARFUL`
- `DISGUSTED`

#### ActionUnitType
Facial Action Units from FACS:
- Upper face: AU1, AU2, AU4, AU5, AU6, AU7
- Lower face: AU9, AU10, AU12, AU15, AU17, AU20, AU23, AU25, AU26

#### EmotionPrediction
Contains emotion analysis results:
```python
prediction = EmotionPrediction(
    emotion=EmotionType.HAPPY,
    confidence=0.85,
    probabilities={...},
    action_units={...},
    features={...},
    timestamp=1000.0,
    frame_number=1
)

# Get top emotions
top_emotions = prediction.get_top_emotions(n=3)

# Get active action units
active_aus = prediction.get_active_action_units(threshold=0.3)

# Serialize to dict
pred_dict = prediction.to_dict()
```

### 2. Feature Extraction (`asdrp.emotion.features`)

#### FeatureExtractor
Extracts geometric features and detects action units:

```python
from asdrp.emotion import FeatureExtractor

extractor = FeatureExtractor(au_threshold=0.3)

# Extract all features
features = extractor.extract_features(landmarks)
# Returns dict with features like:
# - eye_aspect_ratio
# - eyebrow_height
# - mouth_aspect_ratio
# - mouth_corner_height
# etc.

# Detect action units
action_units = extractor.detect_action_units(landmarks)
# Returns dict mapping ActionUnitType to ActionUnit objects

for au_type, au in action_units.items():
    if au.present:
        print(f"{au_type}: intensity={au.intensity:.2f}")
```

### 3. Geometry-Based Analyzer (`asdrp.emotion.geometry_analyzer`)

#### GeometryBasedEmotionAnalyzer
Rule-based emotion classification using action units:

```python
from asdrp.emotion import GeometryBasedEmotionAnalyzer

analyzer = GeometryBasedEmotionAnalyzer(
    emotion_threshold=0.3,
    au_threshold=0.3,
    neutral_threshold=0.2
)

# Analyze single face
prediction = analyzer.analyze(landmarks)

# Analyze multiple faces
predictions = analyzer.analyze_batch(landmarks_list)

# Get emotion rule descriptions
desc = analyzer.get_emotion_description(EmotionType.HAPPY)
print(desc)  # "Happy: lip corner pull (smile) (50%), cheek raise (50%)"
```

**Emotion Detection Rules:**
- **Happy**: AU6 (cheek raiser) + AU12 (smile)
- **Sad**: AU1 (inner brow raise) + AU4 (brow lower) + AU15 (lip corner down)
- **Angry**: AU4 (brow lower) + AU7 (lid tighten) + AU23 (lip tighten)
- **Surprised**: AU1 + AU2 (brow raise) + AU5 (eye widen) + AU26 (jaw drop)
- **Fearful**: AU1 + AU2 + AU4 + AU5 + AU20 (lip stretch)
- **Disgusted**: AU9 (nose wrinkle) + AU15 + AU17 (chin raise)

#### Custom Rules with EmotionRuleBuilder
```python
from asdrp.emotion import EmotionRuleBuilder, ActionUnitType, EmotionType

builder = EmotionRuleBuilder()
builder.add_rule(EmotionType.HAPPY, ActionUnitType.AU12, weight=0.6)
builder.add_rule(EmotionType.HAPPY, ActionUnitType.AU6, weight=0.4)
builder.normalize_weights(EmotionType.HAPPY)

# Apply custom rules
analyzer.EMOTION_AU_RULES = builder.build()
```

### 4. Metrics (`asdrp.emotion.metrics`)

#### Computing Metrics
```python
from asdrp.emotion import compute_emotion_metrics

# Analyze sequence of predictions
metrics = compute_emotion_metrics(predictions)

print(f"Total frames: {metrics.total_frames}")
print(f"Duration: {metrics.duration:.2f}s")
print(f"Dominant emotion: {metrics.dominant_emotion}")
print(f"FPS: {metrics.fps:.1f}")

# Emotion distribution
for emotion, prob in metrics.emotion_distribution.items():
    print(f"{emotion}: {prob:.1%}")

# Transition probabilities
prob = metrics.get_transition_probability(
    EmotionType.NEUTRAL,
    EmotionType.HAPPY
)
```

#### Other Metrics Functions
```python
from asdrp.emotion import (
    compute_emotion_distribution,
    detect_emotion_transitions,
    compute_emotion_stability,
    compute_confidence_statistics,
    compute_au_statistics,
    find_peak_emotions,
    compute_emotion_timeline
)

# Distribution
distribution = compute_emotion_distribution(predictions, normalize=True)

# Transitions
transitions = detect_emotion_transitions(predictions, min_duration=0.5)
for from_emotion, to_emotion, timestamp in transitions:
    print(f"{from_emotion} -> {to_emotion} at {timestamp:.1f}ms")

# Stability score (0-1, higher = more stable)
stability = compute_emotion_stability(predictions)

# Confidence statistics
conf_stats = compute_confidence_statistics(predictions)
print(f"Mean confidence: {conf_stats['mean']:.2f}")

# Action unit statistics
au_stats = compute_au_statistics(predictions)
for au_type, stats in au_stats.items():
    print(f"{au_type}: mean={stats['mean']:.2f}, presence={stats['presence_rate']:.1%}")

# Find peak emotions
peaks = find_peak_emotions(predictions, EmotionType.HAPPY, top_n=5)

# Timeline with sliding window
timeline = compute_emotion_timeline(predictions, window_size=1.0)
```

### 5. Temporal Analysis (`asdrp.emotion.temporal`)

#### TemporalEmotionAnalyzer
Smooth predictions and detect patterns over time:

```python
from asdrp.emotion import TemporalEmotionAnalyzer

temporal = TemporalEmotionAnalyzer(
    window_size=5,              # Moving average window
    hysteresis_threshold=0.15,  # Confidence diff to change emotion
    min_state_duration=0.5,     # Min duration for stable state
    microexpression_duration=0.5 # Max duration for microexpression
)

# Process video frames
for frame in video_frames:
    landmarks = landmarker.detect(frame)[0]
    prediction = emotion_analyzer.analyze(landmarks)

    # Apply temporal smoothing
    smoothed = temporal.smooth_prediction(prediction)

    # Check for microexpression
    microexp = temporal.detect_microexpression(prediction)
    if microexp:
        print(f"Microexpression: {microexp.emotion} "
              f"(intensity={microexp.intensity:.2f})")

# Get detected emotion states
states = temporal.get_emotion_states(min_duration=0.5)
for state in states:
    print(f"{state.emotion}: {state.duration:.2f}s "
          f"(confidence={state.average_confidence:.2f})")

# Get microexpressions
microexps = temporal.get_microexpressions(emotion=EmotionType.ANGRY)
```

#### Temporal Filters
```python
from asdrp.emotion import TemporalFilter

# Median filter (reduces noise)
filtered = TemporalFilter.median_filter(predictions, window_size=5)

# Exponential smoothing
smoothed = TemporalFilter.exponential_smoothing(predictions, alpha=0.3)

# Remove transient spikes
cleaned = TemporalFilter.remove_transients(predictions, min_duration=0.3)
```

## Complete Example: Video Analysis

```python
from asdrp.emotion import (
    GeometryBasedEmotionAnalyzer,
    TemporalEmotionAnalyzer,
    compute_emotion_metrics,
    TemporalFilter
)
from asdrp.face import MediaPipeFaceLandmarker
from asdrp.video import VideoReader
import cv2

# Initialize
landmarker = MediaPipeFaceLandmarker(model_path="models/face_landmarker.task")
emotion_analyzer = GeometryBasedEmotionAnalyzer()
temporal_analyzer = TemporalEmotionAnalyzer(window_size=5)

# Process video
predictions = []
reader = VideoReader("video.mp4")

for frame in reader:
    # Detect landmarks
    landmarks_list = landmarker.detect(frame.image, timestamp_ms=frame.timestamp)

    if not landmarks_list:
        continue

    # Analyze emotion
    landmarks = landmarks_list[0]
    prediction = emotion_analyzer.analyze(landmarks)

    # Apply temporal smoothing
    smoothed = temporal_analyzer.smooth_prediction(prediction)
    predictions.append(smoothed)

    # Display
    text = f"{smoothed.emotion.value}: {smoothed.confidence:.2f}"
    cv2.putText(frame.image, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Emotion Detection", frame.image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

reader.close()
cv2.destroyAllWindows()

# Analyze results
metrics = compute_emotion_metrics(predictions)
print(f"\nVideo Analysis Results:")
print(f"Total frames: {metrics.total_frames}")
print(f"Duration: {metrics.duration:.2f}s")
print(f"Dominant emotion: {metrics.dominant_emotion}")
print(f"\nEmotion Distribution:")
for emotion, prob in sorted(metrics.emotion_distribution.items(),
                           key=lambda x: x[1], reverse=True):
    print(f"  {emotion.value}: {prob:.1%}")

# Get emotion states
states = temporal_analyzer.get_emotion_states(min_duration=1.0)
print(f"\nDetected {len(states)} emotion states:")
for state in states:
    print(f"  {state.emotion.value}: {state.duration:.2f}s "
          f"(frames {state.start_frame}-{state.end_frame})")

# Get microexpressions
microexps = temporal_analyzer.get_microexpressions()
print(f"\nDetected {len(microexps)} microexpressions:")
for me in microexps[:5]:  # Show first 5
    print(f"  Frame {me.frame_number}: {me.emotion.value} "
          f"(intensity={me.intensity:.2f})")
```

## Configuration and Tuning

### Adjusting Thresholds

```python
# Emotion classification threshold
analyzer.set_emotion_threshold(0.4)  # Higher = more conservative

# Action unit detection threshold
analyzer.set_au_threshold(0.35)  # Higher = require stronger AU presence

# Temporal smoothing
temporal = TemporalEmotionAnalyzer(
    window_size=7,              # Larger = smoother but more lag
    hysteresis_threshold=0.2,   # Larger = more stable but less responsive
    min_state_duration=1.0,     # Longer states only
)
```

### Customizing Detection Rules

```python
from asdrp.emotion import EmotionRuleBuilder

# Build custom rules
builder = EmotionRuleBuilder()

# Define happy emotion
builder.add_rule(EmotionType.HAPPY, ActionUnitType.AU12, 0.7)  # Smile is most important
builder.add_rule(EmotionType.HAPPY, ActionUnitType.AU6, 0.3)   # Cheek raise secondary

# Define sad emotion
builder.add_rule(EmotionType.SAD, ActionUnitType.AU1, 0.3)
builder.add_rule(EmotionType.SAD, ActionUnitType.AU4, 0.4)
builder.add_rule(EmotionType.SAD, ActionUnitType.AU15, 0.3)

# Apply
analyzer.EMOTION_AU_RULES = builder.build()
```

## References

1. Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System (FACS)*. Consulting Psychologists Press.

2. Ekman, P. (1992). An argument for basic emotions. *Cognition & Emotion*, 6(3-4), 169-200.

3. Ekman, P., Friesen, W. V., & Hager, J. C. (2002). *Facial Action Coding System: The Manual*. Research Nexus.

4. Ekman, P., & Friesen, W. V. (2003). *Unmasking the face: A guide to recognizing emotions from facial clues*. Malor Books.
