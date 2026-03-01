"""Visualization and output generation module.

This module provides tools for visualizing emotion detection results, including:
- Face landmark overlays on video frames
- Emotion labels and probability displays
- Statistical plots and charts
- Temporal heatmaps and transition matrices
"""

from .emotion_display import EMOTION_COLORS, DisplayStyle, EmotionDisplay
from .heatmap import EmotionHeatmap
from .overlay import (
    ALL_CONNECTIONS,
    FACE_OVAL_CONNECTIONS,
    LEFT_EYE_CONNECTIONS,
    LEFT_EYEBROW_CONNECTIONS,
    MOUTH_CONNECTIONS,
    NOSE_CONNECTIONS,
    RIGHT_EYE_CONNECTIONS,
    RIGHT_EYEBROW_CONNECTIONS,
    FaceOverlay,
    OverlayStyle,
)
from .plots import (
    plot_action_units,
    plot_confidence_over_time,
    plot_emotion_distribution,
    plot_emotion_probabilities_over_time,
    plot_emotion_summary,
    plot_emotion_timeline,
    plot_emotion_transitions,
)

__all__ = [
    # Overlay classes and styles
    "FaceOverlay",
    "OverlayStyle",
    # Connection definitions
    "ALL_CONNECTIONS",
    "LEFT_EYE_CONNECTIONS",
    "RIGHT_EYE_CONNECTIONS",
    "LEFT_EYEBROW_CONNECTIONS",
    "RIGHT_EYEBROW_CONNECTIONS",
    "NOSE_CONNECTIONS",
    "MOUTH_CONNECTIONS",
    "FACE_OVAL_CONNECTIONS",
    # Emotion display
    "EmotionDisplay",
    "DisplayStyle",
    "EMOTION_COLORS",
    # Plotting functions
    "plot_emotion_distribution",
    "plot_emotion_timeline",
    "plot_confidence_over_time",
    "plot_action_units",
    "plot_emotion_transitions",
    "plot_emotion_probabilities_over_time",
    "plot_emotion_summary",
    # Heatmap
    "EmotionHeatmap",
]
