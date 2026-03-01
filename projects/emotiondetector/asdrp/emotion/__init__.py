"""Emotion detection and classification module.

This module provides comprehensive tools for detecting and analyzing emotions
from facial landmarks. It includes:

- Base classes and data structures for emotion analysis
- Feature extraction and Action Unit detection based on FACS
- Rule-based geometry analyzer for emotion classification
- Temporal analysis with smoothing and microexpression detection
- Metrics and statistics for emotion sequences

Example:
    >>> from asdrp.emotion import GeometryBasedEmotionAnalyzer, TemporalEmotionAnalyzer
    >>> from asdrp.face import MediaPipeFaceLandmarker
    >>>
    >>> # Initialize analyzers
    >>> landmarker = MediaPipeFaceLandmarker()
    >>> emotion_analyzer = GeometryBasedEmotionAnalyzer()
    >>> temporal_analyzer = TemporalEmotionAnalyzer()
    >>>
    >>> # Analyze a frame
    >>> landmarks = landmarker.detect(frame)[0]
    >>> prediction = emotion_analyzer.analyze(landmarks)
    >>> smoothed = temporal_analyzer.smooth_prediction(prediction)
    >>> print(f"Detected emotion: {smoothed.emotion} (confidence: {smoothed.confidence:.2f})")
"""

# Base classes and enums
from asdrp.emotion.base import (
    ActionUnit,
    ActionUnitType,
    BaseEmotionAnalyzer,
    EmotionPrediction,
    EmotionType,
)

# Feature extraction
from asdrp.emotion.features import FeatureExtractor

# Geometry-based analyzer
from asdrp.emotion.geometry_analyzer import (
    EmotionRuleBuilder,
    GeometryBasedEmotionAnalyzer,
)

# Metrics and statistics
from asdrp.emotion.metrics import (
    EmotionMetrics,
    compute_au_statistics,
    compute_confidence_statistics,
    compute_emotion_distribution,
    compute_emotion_metrics,
    compute_emotion_stability,
    compute_emotion_timeline,
    detect_emotion_transitions,
    find_peak_emotions,
)

# Temporal analysis
from asdrp.emotion.temporal import (
    EmotionState,
    Microexpression,
    TemporalEmotionAnalyzer,
    TemporalFilter,
)

__all__ = [
    # Base classes and enums
    "ActionUnit",
    "ActionUnitType",
    "BaseEmotionAnalyzer",
    "EmotionPrediction",
    "EmotionType",
    # Feature extraction
    "FeatureExtractor",
    # Geometry-based analyzer
    "GeometryBasedEmotionAnalyzer",
    "EmotionRuleBuilder",
    # Metrics
    "EmotionMetrics",
    "compute_emotion_metrics",
    "compute_emotion_distribution",
    "detect_emotion_transitions",
    "compute_emotion_stability",
    "compute_confidence_statistics",
    "compute_au_statistics",
    "find_peak_emotions",
    "compute_emotion_timeline",
    # Temporal analysis
    "TemporalEmotionAnalyzer",
    "TemporalFilter",
    "EmotionState",
    "Microexpression",
]
