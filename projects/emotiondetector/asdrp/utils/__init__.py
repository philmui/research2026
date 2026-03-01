"""Utility functions and helpers for the emotion detection pipeline.

This package provides various utility modules including:
- config: Configuration dataclasses for pipeline components
- geometry: Geometric calculations for facial landmarks
- smoothing: Temporal filtering for noise reduction
- export: Functions for exporting results to various formats
"""

# Configuration classes
from .config import (
    EmotionAnalysisConfig,
    FaceDetectionConfig,
    PipelineConfig,
    VideoConfig,
    VisualizationConfig,
)

# Geometry utilities
from .geometry import (
    calculate_angle_2d,
    calculate_angle_3d,
    calculate_centroid,
    calculate_distance_2d,
    calculate_distance_3d,
    denormalize_points,
    normalize_points,
    point_line_distance,
    point_segment_distance,
)

# Smoothing filters
from .smoothing import (
    ExponentialMovingAverageFilter,
    KalmanFilter,
    MedianFilter,
    MovingAverageFilter,
)

# Export utilities
from .export import (
    export_analysis_summary,
    export_emotions_to_csv,
    export_emotions_to_json,
    export_landmarks_to_csv,
    export_landmarks_to_json,
    export_to_csv,
    export_to_json,
)

__all__ = [
    # Config
    "FaceDetectionConfig",
    "EmotionAnalysisConfig",
    "VideoConfig",
    "VisualizationConfig",
    "PipelineConfig",
    # Geometry
    "calculate_distance_3d",
    "calculate_distance_2d",
    "calculate_angle_3d",
    "calculate_angle_2d",
    "calculate_centroid",
    "point_line_distance",
    "point_segment_distance",
    "normalize_points",
    "denormalize_points",
    # Smoothing
    "MovingAverageFilter",
    "ExponentialMovingAverageFilter",
    "KalmanFilter",
    "MedianFilter",
    # Export
    "export_to_json",
    "export_to_csv",
    "export_landmarks_to_json",
    "export_landmarks_to_csv",
    "export_emotions_to_json",
    "export_emotions_to_csv",
    "export_analysis_summary",
]
