# Emotion Detection Methodology

## Overview

This document explains the emotion detection methodology used in the Emotion Detector system. Our approach combines **facial landmark detection** using MediaPipe with **geometry-based emotion classification** inspired by the Facial Action Coding System (FACS).

## MediaPipe Face Landmarker

### What is MediaPipe Face Landmarker?

MediaPipe Face Landmarker is a machine learning solution developed by Google that detects facial landmarks in real-time. It identifies 478 distinct 3D landmarks across the human face, providing precise locations for eyes, eyebrows, nose, mouth, and facial contours.

### Key Features

- **478 Landmarks**: High-resolution facial mapping
- **3D Coordinates**: x, y, and depth (z) information
- **Real-time Performance**: Optimized for video streams (30+ FPS)
- **Multi-face Support**: Can detect multiple faces simultaneously
- **Robust Tracking**: Handles various lighting conditions and head poses

### Landmark Groups

The 478 landmarks are organized into functional regions:

```
Face Oval (Contour):      Landmarks 0-16, 234-454
Left Eye:                 Landmarks 33, 133, 157, 158, 159, 160, 161, 173, 246
Right Eye:                Landmarks 263, 362, 384, 385, 386, 387, 388, 398, 466
Left Eyebrow:             Landmarks 70, 63, 105, 66, 107, 55, 65, 52, 53, 46
Right Eyebrow:            Landmarks 300, 293, 334, 296, 336, 285, 295, 282, 283, 276
Nose:                     Landmarks 1, 2, 98, 327, 4, 5, 6, 168, 197, 195
Mouth Outer:              Landmarks 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
Mouth Inner:              Landmarks 78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
Lips:                     Upper and lower lip details
```

### Coordinate System

- **X-axis**: Horizontal position (0 = left, 1 = right)
- **Y-axis**: Vertical position (0 = top, 1 = bottom)
- **Z-axis**: Depth relative to face center (negative = closer to camera)

Coordinates are normalized relative to image dimensions, making them scale-invariant.

## Facial Action Coding System (FACS)

### Introduction to FACS

The Facial Action Coding System (FACS) is a comprehensive, anatomically-based system for describing all observable facial movement. Developed by Paul Ekman and Wallace Friesen in 1978, FACS is the gold standard for measuring facial expressions.

### Action Units (AUs)

FACS decomposes facial expressions into individual components called **Action Units (AUs)**. Each AU represents the contraction or relaxation of one or more facial muscles.

#### Key Action Units for Emotion Detection

| Action Unit | Name | Facial Movement | Primary Muscles |
|------------|------|-----------------|-----------------|
| AU1 | Inner Brow Raiser | Raises inner portion of eyebrows | Frontalis (inner) |
| AU2 | Outer Brow Raiser | Raises outer portion of eyebrows | Frontalis (outer) |
| AU4 | Brow Lowerer | Lowers and draws eyebrows together | Corrugator supercilii |
| AU5 | Upper Lid Raiser | Raises upper eyelid, widens eye | Levator palpebrae |
| AU6 | Cheek Raiser | Raises cheeks, creates crows feet | Orbicularis oculi |
| AU7 | Lid Tightener | Tightens eyelids | Orbicularis oculi |
| AU9 | Nose Wrinkler | Wrinkles nose bridge | Levator labii superioris |
| AU10 | Upper Lip Raiser | Raises upper lip | Levator labii superioris |
| AU12 | Lip Corner Puller | Pulls lip corners up (smile) | Zygomaticus major |
| AU15 | Lip Corner Depressor | Pulls lip corners down | Depressor anguli oris |
| AU17 | Chin Raiser | Raises chin, pushes lower lip up | Mentalis |
| AU20 | Lip Stretcher | Stretches lips horizontally | Risorius |
| AU23 | Lip Tightener | Tightens and narrows lips | Orbicularis oris |
| AU25 | Lips Part | Separates lips slightly | Relaxation |
| AU26 | Jaw Drop | Lowers jaw, opens mouth | Masseter relaxation |
| AU27 | Mouth Stretch | Opens mouth wide | Pterygoids, digastric |

### Emotion-AU Relationships

FACS research has identified consistent AU patterns for basic emotions:

**Happy (Joy)**
- AU6 (Cheek Raiser) + AU12 (Lip Corner Puller)
- Often accompanied by AU25 (Lips Part)

**Sad (Sadness)**
- AU1 (Inner Brow Raiser) + AU4 (Brow Lowerer) + AU15 (Lip Corner Depressor)
- May include AU17 (Chin Raiser)

**Angry (Anger)**
- AU4 (Brow Lowerer) + AU5 (Upper Lid Raiser) + AU7 (Lid Tightener)
- Often with AU23 (Lip Tightener) or AU24 (Lip Presser)

**Surprised (Surprise)**
- AU1 (Inner Brow Raiser) + AU2 (Outer Brow Raiser) + AU5 (Upper Lid Raiser)
- Combined with AU26 (Jaw Drop) or AU27 (Mouth Stretch)

**Fearful (Fear)**
- AU1 (Inner Brow Raiser) + AU2 (Outer Brow Raiser) + AU4 (Brow Lowerer)
- With AU5 (Upper Lid Raiser) + AU20 (Lip Stretcher) + AU25 (Lips Part)

**Disgusted (Disgust)**
- AU9 (Nose Wrinkler) + AU10 (Upper Lip Raiser)
- Sometimes with AU17 (Chin Raiser)

## Geometry-Based Emotion Detection

Our system translates MediaPipe landmarks into FACS action units using geometric analysis.

### Landmark to Action Unit Mapping

#### AU1 & AU2: Brow Raising

```python
def detect_brow_raise(landmarks):
    """
    Compare vertical distance between brows and eyes
    to a neutral baseline.
    """
    # Inner brow (AU1)
    inner_brow_left = landmarks[55]  # Left inner brow
    inner_brow_right = landmarks[285]  # Right inner brow

    # Outer brow (AU2)
    outer_brow_left = landmarks[46]  # Left outer brow
    outer_brow_right = landmarks[276]  # Right outer brow

    # Eye references
    left_eye_top = landmarks[159]
    right_eye_top = landmarks[386]

    # Calculate vertical distances
    inner_distance_left = inner_brow_left.y - left_eye_top.y
    outer_distance_left = outer_brow_left.y - left_eye_top.y

    # Normalize by face height
    face_height = get_face_height(landmarks)

    au1_left = inner_distance_left / face_height
    au2_left = outer_distance_left / face_height

    # Similar for right side...
    # Average both sides for final AU values
```

#### AU4: Brow Lowering

```python
def detect_brow_lower(landmarks):
    """
    Measure horizontal distance between inner brow points.
    Closer = more furrowed = higher AU4 activation.
    """
    left_inner_brow = landmarks[55]
    right_inner_brow = landmarks[285]

    # Horizontal distance
    brow_distance = abs(left_inner_brow.x - right_inner_brow.x)

    # Normalize by face width
    face_width = get_face_width(landmarks)

    # Invert: smaller distance = higher activation
    au4_value = 1.0 - (brow_distance / (face_width * 0.15))
    return clip(au4_value, 0, 1)
```

#### AU5: Upper Lid Raiser

```python
def detect_eye_opening(landmarks):
    """
    Vertical distance between upper and lower eyelids.
    Larger distance = wider eyes = higher AU5.
    """
    # Left eye
    left_eye_top = landmarks[159]
    left_eye_bottom = landmarks[145]
    left_eye_height = abs(left_eye_top.y - left_eye_bottom.y)

    # Right eye
    right_eye_top = landmarks[386]
    right_eye_bottom = landmarks[374]
    right_eye_height = abs(right_eye_top.y - right_eye_bottom.y)

    # Average both eyes, normalize by face height
    avg_eye_height = (left_eye_height + right_eye_height) / 2
    face_height = get_face_height(landmarks)

    au5_value = avg_eye_height / (face_height * 0.04)  # Baseline ratio
    return clip(au5_value, 0, 1)
```

#### AU6: Cheek Raiser

```python
def detect_cheek_raise(landmarks):
    """
    Measure upward movement of cheek landmarks.
    Also correlates with narrowing of eye height.
    """
    # Cheek points
    left_cheek = landmarks[50]
    right_cheek = landmarks[280]

    # Reference: mouth corners
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]

    # Vertical distance from mouth to cheek
    left_distance = left_mouth.y - left_cheek.y
    right_distance = right_mouth.y - right_cheek.y

    # Normalize
    face_height = get_face_height(landmarks)
    avg_distance = (left_distance + right_distance) / 2

    au6_value = avg_distance / (face_height * 0.08)
    return clip(au6_value, 0, 1)
```

#### AU9 & AU10: Nose and Upper Lip

```python
def detect_nose_wrinkle_lip_raise(landmarks):
    """
    Vertical movement of nose bridge and upper lip.
    """
    nose_bridge = landmarks[168]
    upper_lip_center = landmarks[0]

    # Distance between nose and upper lip (gets smaller when raised)
    distance = abs(upper_lip_center.y - nose_bridge.y)

    face_height = get_face_height(landmarks)
    normalized = distance / face_height

    # Invert: smaller distance = higher activation
    au9_au10_value = 1.0 - (normalized / 0.15)
    return clip(au9_au10_value, 0, 1)
```

#### AU12: Lip Corner Puller (Smile)

```python
def detect_smile(landmarks):
    """
    Measure upward and outward movement of mouth corners.
    Classic smile indicator.
    """
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    mouth_center = landmarks[0]

    # Vertical position of corners relative to center
    left_raise = mouth_center.y - left_corner.y
    right_raise = mouth_center.y - right_corner.y
    avg_raise = (left_raise + right_raise) / 2

    # Horizontal width (smile stretches mouth)
    mouth_width = abs(left_corner.x - right_corner.x)

    # Combine vertical and horizontal components
    face_width = get_face_width(landmarks)
    face_height = get_face_height(landmarks)

    vertical_component = avg_raise / (face_height * 0.05)
    horizontal_component = mouth_width / (face_width * 0.35)

    au12_value = (vertical_component + horizontal_component) / 2
    return clip(au12_value, 0, 1)
```

#### AU15: Lip Corner Depressor

```python
def detect_frown(landmarks):
    """
    Downward movement of mouth corners.
    Opposite of smile.
    """
    left_corner = landmarks[61]
    right_corner = landmarks[291]
    mouth_center = landmarks[0]

    # Vertical position (negative = below center)
    left_drop = left_corner.y - mouth_center.y
    right_drop = right_corner.y - mouth_center.y
    avg_drop = (left_drop + right_drop) / 2

    # Only positive values (below center) count
    if avg_drop <= 0:
        return 0.0

    face_height = get_face_height(landmarks)
    au15_value = avg_drop / (face_height * 0.05)
    return clip(au15_value, 0, 1)
```

#### AU25 & AU26: Mouth Opening

```python
def detect_mouth_opening(landmarks):
    """
    Vertical distance between upper and lower lips.
    """
    upper_lip = landmarks[13]  # Upper lip center
    lower_lip = landmarks[14]  # Lower lip center

    mouth_height = abs(lower_lip.y - upper_lip.y)

    face_height = get_face_height(landmarks)
    normalized = mouth_height / face_height

    # AU25: Slight opening (0.02-0.08)
    # AU26: Large opening (>0.08)
    au25_value = clip(normalized / 0.08, 0, 1)
    au26_value = clip((normalized - 0.08) / 0.12, 0, 1)

    return au25_value, au26_value
```

### Helper Functions

```python
def get_face_height(landmarks):
    """
    Vertical distance from forehead to chin.
    """
    forehead = landmarks[10]  # Top of face
    chin = landmarks[152]  # Bottom of chin
    return abs(chin.y - forehead.y)

def get_face_width(landmarks):
    """
    Horizontal distance from left to right cheek.
    """
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    return abs(right_cheek.x - left_cheek.x)

def clip(value, min_val, max_val):
    """Constrain value to range."""
    return max(min_val, min(value, max_val))
```

## Emotion Classification Rules

Once action units are computed, we apply rule-based classification:

### Rule Structure

```python
class EmotionRule:
    def __init__(self, name, required_aus, threshold=0.5):
        self.name = name
        self.required_aus = required_aus  # Dict of AU: min_value
        self.threshold = threshold

    def evaluate(self, action_units):
        """
        Check if AU activations match the rule.
        Returns confidence score (0-1).
        """
        matches = []
        for au, min_value in self.required_aus.items():
            au_value = getattr(action_units, au)
            if au_value >= min_value:
                matches.append(au_value)
            else:
                matches.append(0)

        # Average of matching AUs
        confidence = sum(matches) / len(self.required_aus)
        return confidence if confidence >= self.threshold else 0.0
```

### Emotion-Specific Rules

```python
# Happy: Smile + Cheek Raise
HAPPY_RULE = EmotionRule(
    name="happy",
    required_aus={
        "au6_cheek_raiser": 0.3,
        "au12_lip_corner_puller": 0.4
    },
    threshold=0.35
)

# Sad: Inner Brow Raise + Lip Corner Depressor
SAD_RULE = EmotionRule(
    name="sad",
    required_aus={
        "au1_inner_brow_raiser": 0.3,
        "au4_brow_lowerer": 0.2,
        "au15_lip_corner_depressor": 0.3
    },
    threshold=0.3
)

# Angry: Brow Lower + Lid Tighten + Lip Tighten
ANGRY_RULE = EmotionRule(
    name="angry",
    required_aus={
        "au4_brow_lowerer": 0.5,
        "au5_upper_lid_raiser": 0.3,
        "au7_lid_tightener": 0.3,
        "au23_lip_tightener": 0.3
    },
    threshold=0.4
)

# Surprised: Brow Raise + Eye Widen + Jaw Drop
SURPRISED_RULE = EmotionRule(
    name="surprised",
    required_aus={
        "au1_inner_brow_raiser": 0.5,
        "au2_outer_brow_raiser": 0.5,
        "au5_upper_lid_raiser": 0.5,
        "au26_jaw_drop": 0.4
    },
    threshold=0.5
)

# Fearful: Similar to surprise but with lip stretch
FEARFUL_RULE = EmotionRule(
    name="fearful",
    required_aus={
        "au1_inner_brow_raiser": 0.4,
        "au2_outer_brow_raiser": 0.4,
        "au4_brow_lowerer": 0.3,
        "au5_upper_lid_raiser": 0.4,
        "au20_lip_stretcher": 0.3
    },
    threshold=0.4
)

# Disgusted: Nose Wrinkle + Upper Lip Raise
DISGUSTED_RULE = EmotionRule(
    name="disgusted",
    required_aus={
        "au9_nose_wrinkler": 0.4,
        "au10_upper_lip_raiser": 0.4
    },
    threshold=0.4
)
```

### Classification Process

```python
def classify_emotion(action_units):
    """
    Apply all rules and return highest confidence emotion.
    """
    rules = [
        HAPPY_RULE,
        SAD_RULE,
        ANGRY_RULE,
        SURPRISED_RULE,
        FEARFUL_RULE,
        DISGUSTED_RULE
    ]

    scores = {}
    for rule in rules:
        confidence = rule.evaluate(action_units)
        if confidence > 0:
            scores[rule.name] = confidence

    if not scores:
        return Emotion(label="neutral", confidence=1.0)

    # Primary emotion: highest score
    primary = max(scores.items(), key=lambda x: x[1])

    # Secondary emotion: second highest (if close)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    secondary = None
    if len(sorted_scores) > 1 and sorted_scores[1][1] > 0.3:
        if sorted_scores[0][1] - sorted_scores[1][1] < 0.2:
            secondary = sorted_scores[1][0]

    return Emotion(
        label=primary[0],
        confidence=primary[1],
        action_units=action_units,
        secondary_emotion=secondary
    )
```

## Temporal Smoothing

To reduce jitter and noise in real-time detection:

### Moving Average Filter

```python
class EmotionTracker:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.history = []

    def update(self, emotion):
        self.history.append(emotion)
        if len(self.history) > self.window_size:
            self.history.pop(0)

        # Count occurrences of each emotion
        emotion_counts = {}
        for e in self.history:
            emotion_counts[e.label] = emotion_counts.get(e.label, 0) + 1

        # Most frequent emotion in window
        smoothed_label = max(emotion_counts.items(), key=lambda x: x[1])[0]

        # Average confidence for that emotion
        confidences = [e.confidence for e in self.history if e.label == smoothed_label]
        smoothed_confidence = sum(confidences) / len(confidences)

        return Emotion(label=smoothed_label, confidence=smoothed_confidence)
```

### Hysteresis

Prevent rapid switching between emotions:

```python
class HysteresisTracker:
    def __init__(self, switch_threshold=0.15):
        self.current_emotion = "neutral"
        self.switch_threshold = switch_threshold

    def update(self, new_emotion):
        if new_emotion.label == self.current_emotion:
            # Stay with current emotion
            return new_emotion

        # Require significant confidence difference to switch
        if new_emotion.confidence > self.switch_threshold:
            self.current_emotion = new_emotion.label
            return new_emotion
        else:
            # Keep previous emotion
            return Emotion(label=self.current_emotion, confidence=0.5)
```

## Validation and Calibration

### Baseline Calibration

Capture neutral expression to establish personal baseline:

```python
def calibrate_baseline(landmarks_sequence):
    """
    Average landmarks over neutral expression frames.
    Use as reference for detecting deviations.
    """
    avg_landmarks = average_landmarks(landmarks_sequence)

    baseline = {
        'brow_height': measure_brow_height(avg_landmarks),
        'eye_opening': measure_eye_opening(avg_landmarks),
        'mouth_width': measure_mouth_width(avg_landmarks),
        'mouth_height': measure_mouth_height(avg_landmarks)
    }

    return baseline
```

### Confidence Scoring

Factors affecting confidence:

1. **Landmark Detection Quality**: MediaPipe confidence score
2. **AU Activation Strength**: How strongly AUs are activated
3. **Rule Match Certainty**: How well the pattern matches known rules
4. **Temporal Consistency**: Stability over recent frames
5. **Face Quality**: Lighting, occlusion, head pose

## Limitations and Future Improvements

### Current Limitations

1. **Rule-Based**: Cannot detect subtle or complex emotions
2. **Cultural Variations**: Rules based on Western FACS research
3. **Individual Differences**: Some people have less expressive faces
4. **Lighting Sensitivity**: Poor lighting affects landmark detection
5. **Head Pose**: Large angles reduce landmark accuracy

### Planned Improvements

1. **Machine Learning Integration**: Train neural networks on AU features
2. **Person-Specific Calibration**: Adapt rules to individual expressions
3. **Micro-Expression Detection**: Capture brief, subtle emotions
4. **Context Awareness**: Consider environmental and social context
5. **Multi-Modal Fusion**: Combine facial, vocal, and body language

## References

- Ekman, P., & Friesen, W. V. (1978). *Facial Action Coding System: A Technique for the Measurement of Facial Movement*. Consulting Psychologists Press.
- Ekman, P. (1992). "An Argument for Basic Emotions." *Cognition & Emotion*, 6(3-4), 169-200.
- MediaPipe Face Landmarker Guide: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- FACS Manual: https://www.paulekman.com/facial-action-coding-system/
- Li, S., & Deng, W. (2020). "Deep Facial Expression Recognition: A Survey." *IEEE Transactions on Affective Computing*.
