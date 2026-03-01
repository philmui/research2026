"""ASDRP Emotion Detection Package.

A comprehensive package for emotion detection from video using facial analysis.

This package provides a complete pipeline for detecting and analyzing emotions
from video files and real-time camera streams. It includes:

- Video reading and camera capture utilities
- MediaPipe-based face detection and landmark extraction
- Geometry-based emotion classification using Facial Action Coding System (FACS)
- Temporal analysis with smoothing and microexpression detection
- Flexible configuration system
- Multiple export formats (JSON, CSV)
- Visualization and annotation capabilities

Quick Start:
    >>> from asdrp import EmotionDetectionPipeline, PipelineConfig
    >>>
    >>> # Create configuration
    >>> config = PipelineConfig.from_defaults(
    ...     model_path="models/face_landmarker.task",
    ...     input_path="input.mp4",
    ...     output_path="output.mp4"
    ... )
    >>>
    >>> # Run pipeline
    >>> with EmotionDetectionPipeline(config) as pipeline:
    ...     results = pipeline.process_video()
    ...     pipeline.save_results("results.json")

For real-time processing:
    >>> config = PipelineConfig.for_realtime_processing(
    ...     model_path="models/face_landmarker.task",
    ...     input_path="0"  # Use default webcam
    ... )
    >>> with EmotionDetectionPipeline(config) as pipeline:
    ...     for result in pipeline.process_stream(max_frames=100):
    ...         print(f"Emotion: {result['faces'][0]['emotion']}")
"""

__version__ = "0.1.0"
__author__ = "ASDRP Research Team"
__license__ = "MIT"

# Core Pipeline
from asdrp.pipeline import EmotionDetectionPipeline, PipelineError

# Configuration
from asdrp.utils.config import (
    EmotionAnalysisConfig,
    FaceDetectionConfig,
    PipelineConfig,
    VideoConfig,
    VisualizationConfig,
)

# Emotion Analysis
from asdrp.emotion import (
    EmotionPrediction,
    EmotionType,
    GeometryBasedEmotionAnalyzer,
    TemporalEmotionAnalyzer,
)

# Face Detection
from asdrp.face.base import FaceLandmarks
from asdrp.face.detector import MediaPipeFaceDetector

# Video Processing
from asdrp.video.camera import CameraCapture
from asdrp.video.frame import FrameData, VideoMetadata
from asdrp.video.reader import VideoFileReader

# Export Utilities
from asdrp.utils.export import (
    export_emotions_to_csv,
    export_emotions_to_json,
    export_to_csv,
    export_to_json,
)

# Public API
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core Pipeline
    "EmotionDetectionPipeline",
    "PipelineError",
    # Configuration
    "PipelineConfig",
    "FaceDetectionConfig",
    "EmotionAnalysisConfig",
    "VideoConfig",
    "VisualizationConfig",
    # Emotion Analysis
    "EmotionPrediction",
    "EmotionType",
    "GeometryBasedEmotionAnalyzer",
    "TemporalEmotionAnalyzer",
    # Face Detection
    "FaceLandmarks",
    "MediaPipeFaceDetector",
    # Video Processing
    "VideoFileReader",
    "CameraCapture",
    "FrameData",
    "VideoMetadata",
    # Export Utilities
    "export_to_json",
    "export_to_csv",
    "export_emotions_to_json",
    "export_emotions_to_csv",
]
