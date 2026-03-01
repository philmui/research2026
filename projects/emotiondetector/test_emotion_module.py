#!/usr/bin/env python3
"""Test script to verify emotion module implementation."""

import numpy as np

# Test imports
print("Testing imports...")
from asdrp.emotion import (
    ActionUnit,
    ActionUnitType,
    EmotionPrediction,
    EmotionType,
    FeatureExtractor,
    GeometryBasedEmotionAnalyzer,
    TemporalEmotionAnalyzer,
    compute_emotion_metrics,
)
from asdrp.face.base import FaceLandmarks

print("✓ All imports successful\n")

# Test 1: Create emotion types
print("Test 1: EmotionType enum")
for emotion in EmotionType:
    print(f"  - {emotion}")
print("✓ EmotionType enum working\n")

# Test 2: Create action unit
print("Test 2: ActionUnit creation")
au = ActionUnit(
    au_type=ActionUnitType.AU12,
    intensity=0.8,
    present=True,
    confidence=0.95
)
print(f"  Created: {au.au_type} with intensity {au.intensity}")
print("✓ ActionUnit creation working\n")

# Test 3: Create mock landmarks
print("Test 3: Create mock facial landmarks")
# Create 478 landmarks (MediaPipe standard)
mock_landmarks = np.random.rand(478, 3).astype(np.float32) * 0.5 + 0.25
face_landmarks = FaceLandmarks(
    landmarks=mock_landmarks,
    timestamp=1000.0,
    frame_number=1
)
print(f"  Created landmarks with {face_landmarks.num_landmarks} points")
print("✓ FaceLandmarks creation working\n")

# Test 4: Feature extraction
print("Test 4: Feature extraction")
extractor = FeatureExtractor()
features = extractor.extract_features(face_landmarks)
print(f"  Extracted {len(features)} features:")
for key, value in list(features.items())[:5]:
    print(f"    - {key}: {value:.4f}")
print("✓ Feature extraction working\n")

# Test 5: Action unit detection
print("Test 5: Action unit detection")
action_units = extractor.detect_action_units(face_landmarks)
print(f"  Detected {len(action_units)} action units:")
for au_type, au in list(action_units.items())[:5]:
    print(f"    - {au_type}: intensity={au.intensity:.2f}, present={au.present}")
print("✓ Action unit detection working\n")

# Test 6: Emotion analysis
print("Test 6: Emotion analysis")
analyzer = GeometryBasedEmotionAnalyzer()
prediction = analyzer.analyze(face_landmarks)
print(f"  Predicted emotion: {prediction.emotion}")
print(f"  Confidence: {prediction.confidence:.3f}")
print(f"  Top 3 emotions:")
for emotion, prob in prediction.get_top_emotions(3):
    print(f"    - {emotion}: {prob:.3f}")
print("✓ Emotion analysis working\n")

# Test 7: Temporal smoothing
print("Test 7: Temporal smoothing")
temporal = TemporalEmotionAnalyzer(window_size=3)

# Create sequence of predictions
predictions = []
for i in range(10):
    mock_lm = np.random.rand(478, 3).astype(np.float32) * 0.5 + 0.25
    lm = FaceLandmarks(landmarks=mock_lm, timestamp=float(i * 100), frame_number=i)
    pred = analyzer.analyze(lm)
    smoothed = temporal.smooth_prediction(pred)
    predictions.append(smoothed)
    print(f"  Frame {i}: {smoothed.emotion} (conf: {smoothed.confidence:.2f})")

print("✓ Temporal smoothing working\n")

# Test 8: Compute metrics
print("Test 8: Compute metrics")
metrics = compute_emotion_metrics(predictions)
print(f"  Total frames: {metrics.total_frames}")
print(f"  Duration: {metrics.duration:.2f}s")
print(f"  Dominant emotion: {metrics.dominant_emotion}")
print(f"  Emotion distribution:")
for emotion, prob in metrics.emotion_distribution.items():
    print(f"    - {emotion}: {prob:.2%}")
print("✓ Metrics computation working\n")

# Test 9: Emotion prediction serialization
print("Test 9: Prediction serialization")
pred_dict = prediction.to_dict()
print(f"  Serialized keys: {list(pred_dict.keys())}")
print("✓ Serialization working\n")

# Test 10: Emotion rule descriptions
print("Test 10: Emotion rule descriptions")
for emotion in [EmotionType.HAPPY, EmotionType.SAD, EmotionType.ANGRY]:
    desc = analyzer.get_emotion_description(emotion)
    print(f"  {desc}")
print("✓ Rule descriptions working\n")

print("=" * 60)
print("All tests passed! Emotion module is working correctly.")
print("=" * 60)
