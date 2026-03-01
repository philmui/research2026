"""Configuration dataclasses for the emotion detection pipeline.

This module provides structured configuration classes for all components of the
emotion detection system, including face detection, emotion analysis, video
processing, and visualization settings.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass
class FaceDetectionConfig:
    """Configuration for face detection and landmark extraction.

    Attributes:
        model_path: Path to the MediaPipe Face Landmarker model file (.task).
        min_detection_confidence: Minimum confidence for face detection (0.0 to 1.0).
        min_tracking_confidence: Minimum confidence for face tracking (0.0 to 1.0).
        num_faces: Maximum number of faces to detect per frame.
        running_mode: Processing mode ('IMAGE' or 'VIDEO').
        enable_smoothing: Whether to apply temporal smoothing to landmarks.
        smoothing_window_size: Window size for moving average smoothing.
        smoothing_alpha: Alpha parameter for exponential moving average (0.0 to 1.0).

    Example:
        >>> config = FaceDetectionConfig(
        ...     model_path="models/face_landmarker.task",
        ...     num_faces=2,
        ...     enable_smoothing=True
        ... )
    """

    model_path: str | Path
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    num_faces: int = 1
    running_mode: str = "VIDEO"
    enable_smoothing: bool = False
    smoothing_window_size: int = 5
    smoothing_alpha: float = 0.3

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.model_path = Path(self.model_path)

        if not 0.0 <= self.min_detection_confidence <= 1.0:
            raise ValueError(
                f"min_detection_confidence must be in [0.0, 1.0], "
                f"got {self.min_detection_confidence}"
            )
        if not 0.0 <= self.min_tracking_confidence <= 1.0:
            raise ValueError(
                f"min_tracking_confidence must be in [0.0, 1.0], "
                f"got {self.min_tracking_confidence}"
            )
        if self.num_faces < 1:
            raise ValueError(f"num_faces must be at least 1, got {self.num_faces}")
        if self.running_mode not in ("IMAGE", "VIDEO"):
            raise ValueError(
                f"running_mode must be 'IMAGE' or 'VIDEO', got {self.running_mode}"
            )
        if not 0.0 <= self.smoothing_alpha <= 1.0:
            raise ValueError(
                f"smoothing_alpha must be in [0.0, 1.0], got {self.smoothing_alpha}"
            )
        if self.smoothing_window_size < 1:
            raise ValueError(
                f"smoothing_window_size must be at least 1, got {self.smoothing_window_size}"
            )


@dataclass
class EmotionAnalysisConfig:
    """Configuration for emotion analysis.

    Attributes:
        analyzer_type: Type of emotion analyzer ('geometric', 'cnn', 'hybrid').
        model_path: Optional path to trained emotion detection model.
        confidence_threshold: Minimum confidence threshold for emotion prediction.
        enable_temporal_smoothing: Whether to apply temporal smoothing to emotions.
        smoothing_window_size: Window size for emotion smoothing.
        emotion_classes: List of emotion class names to detect.
        use_action_units: Whether to compute facial action units.
        normalize_features: Whether to normalize feature vectors.

    Example:
        >>> config = EmotionAnalysisConfig(
        ...     analyzer_type='geometric',
        ...     confidence_threshold=0.6,
        ...     emotion_classes=['happy', 'sad', 'angry', 'neutral']
        ... )
    """

    analyzer_type: str = "geometric"
    model_path: Optional[str | Path] = None
    confidence_threshold: float = 0.5
    enable_temporal_smoothing: bool = True
    smoothing_window_size: int = 10
    emotion_classes: list[str] = field(
        default_factory=lambda: ["neutral", "happy", "sad", "angry", "surprised", "fearful"]
    )
    use_action_units: bool = False
    normalize_features: bool = True

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.model_path is not None:
            self.model_path = Path(self.model_path)

        if self.analyzer_type not in ("geometric", "cnn", "hybrid"):
            raise ValueError(
                f"analyzer_type must be 'geometric', 'cnn', or 'hybrid', "
                f"got {self.analyzer_type}"
            )
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be in [0.0, 1.0], got {self.confidence_threshold}"
            )
        if self.smoothing_window_size < 1:
            raise ValueError(
                f"smoothing_window_size must be at least 1, got {self.smoothing_window_size}"
            )
        if not self.emotion_classes:
            raise ValueError("emotion_classes must not be empty")


@dataclass
class VideoConfig:
    """Configuration for video processing.

    Attributes:
        input_path: Path to input video file or camera device ID.
        output_path: Optional path to output video file.
        codec: FourCC codec code for output video (e.g., 'mp4v', 'avc1', 'H264').
        fps: Frames per second for output video (None uses input video fps).
        resolution: Output video resolution as (width, height) (None uses input resolution).
        start_frame: Starting frame number for processing (0-indexed).
        end_frame: Ending frame number for processing (None processes to end).
        skip_frames: Number of frames to skip between processed frames.
        max_frames: Maximum number of frames to process (None processes all).
        display_realtime: Whether to display video in real-time during processing.
        buffer_size: Size of frame buffer for video reading.

    Example:
        >>> config = VideoConfig(
        ...     input_path="input.mp4",
        ...     output_path="output.mp4",
        ...     codec='mp4v',
        ...     skip_frames=2
        ... )
    """

    input_path: str | Path
    output_path: Optional[str | Path] = None
    codec: str = "mp4v"
    fps: Optional[float] = None
    resolution: Optional[Tuple[int, int]] = None
    start_frame: int = 0
    end_frame: Optional[int] = None
    skip_frames: int = 0
    max_frames: Optional[int] = None
    display_realtime: bool = False
    buffer_size: int = 32

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.input_path = Path(self.input_path)
        if self.output_path is not None:
            self.output_path = Path(self.output_path)

        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.resolution is not None:
            if len(self.resolution) != 2:
                raise ValueError(
                    f"resolution must be (width, height), got {self.resolution}"
                )
            if self.resolution[0] <= 0 or self.resolution[1] <= 0:
                raise ValueError(f"resolution dimensions must be positive, got {self.resolution}")
        if self.start_frame < 0:
            raise ValueError(f"start_frame must be non-negative, got {self.start_frame}")
        if self.end_frame is not None and self.end_frame < self.start_frame:
            raise ValueError(
                f"end_frame ({self.end_frame}) must be >= start_frame ({self.start_frame})"
            )
        if self.skip_frames < 0:
            raise ValueError(f"skip_frames must be non-negative, got {self.skip_frames}")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError(f"max_frames must be at least 1, got {self.max_frames}")
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be at least 1, got {self.buffer_size}")


@dataclass
class VisualizationConfig:
    """Configuration for visualization and rendering.

    Attributes:
        draw_landmarks: Whether to draw facial landmarks on frames.
        draw_bounding_box: Whether to draw bounding box around detected faces.
        show_emotion: Whether to display emotion labels on frames.
        show_confidence: Whether to display confidence scores.
        show_timestamp: Whether to display frame timestamp.
        landmark_color: BGR color tuple for landmarks (B, G, R).
        bbox_color: BGR color tuple for bounding box.
        text_color: BGR color tuple for text labels.
        landmark_radius: Radius of landmark points in pixels.
        line_thickness: Thickness of lines in pixels.
        font_scale: Scale factor for text font.
        background_alpha: Alpha transparency for text background (0.0 to 1.0).
        landmark_connections: Whether to draw connections between landmarks.
        connection_color: BGR color tuple for landmark connections.
        connection_thickness: Thickness of landmark connections.

    Example:
        >>> config = VisualizationConfig(
        ...     draw_landmarks=True,
        ...     show_emotion=True,
        ...     landmark_color=(0, 255, 0),  # Green
        ...     text_color=(255, 255, 255)   # White
        ... )
    """

    draw_landmarks: bool = True
    draw_bounding_box: bool = True
    show_emotion: bool = True
    show_confidence: bool = True
    show_timestamp: bool = False
    landmark_color: Tuple[int, int, int] = (0, 255, 0)  # Green
    bbox_color: Tuple[int, int, int] = (255, 0, 0)  # Blue
    text_color: Tuple[int, int, int] = (255, 255, 255)  # White
    landmark_radius: int = 2
    line_thickness: int = 2
    font_scale: float = 0.7
    background_alpha: float = 0.5
    landmark_connections: bool = False
    connection_color: Tuple[int, int, int] = (100, 100, 100)  # Gray
    connection_thickness: int = 1

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not 0.0 <= self.background_alpha <= 1.0:
            raise ValueError(
                f"background_alpha must be in [0.0, 1.0], got {self.background_alpha}"
            )
        if self.landmark_radius < 1:
            raise ValueError(f"landmark_radius must be at least 1, got {self.landmark_radius}")
        if self.line_thickness < 1:
            raise ValueError(f"line_thickness must be at least 1, got {self.line_thickness}")
        if self.font_scale <= 0:
            raise ValueError(f"font_scale must be positive, got {self.font_scale}")


@dataclass
class PipelineConfig:
    """Composite configuration for the complete emotion detection pipeline.

    This class combines all component configurations into a single unified
    configuration object for the entire processing pipeline.

    Attributes:
        face_detection: Configuration for face detection.
        emotion_analysis: Configuration for emotion analysis.
        video: Configuration for video processing.
        visualization: Configuration for visualization.
        output_format: Format for saving results ('json', 'csv', 'both').
        save_annotated_video: Whether to save video with visualizations.
        save_landmarks: Whether to save facial landmarks to file.
        save_emotions: Whether to save emotion predictions to file.
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
        device: Device for computation ('cpu', 'cuda', 'mps').
        num_workers: Number of parallel workers for processing.
        batch_size: Batch size for processing multiple frames.
        enable_profiling: Whether to enable performance profiling.
        metadata: Additional custom metadata.

    Example:
        >>> config = PipelineConfig(
        ...     face_detection=FaceDetectionConfig(model_path="model.task"),
        ...     emotion_analysis=EmotionAnalysisConfig(analyzer_type='geometric'),
        ...     video=VideoConfig(input_path="input.mp4"),
        ...     visualization=VisualizationConfig(show_emotion=True)
        ... )
    """

    face_detection: FaceDetectionConfig
    emotion_analysis: EmotionAnalysisConfig
    video: VideoConfig
    visualization: VisualizationConfig
    output_format: str = "json"
    save_annotated_video: bool = True
    save_landmarks: bool = False
    save_emotions: bool = True
    log_level: str = "INFO"
    device: str = "cpu"
    num_workers: int = 1
    batch_size: int = 1
    enable_profiling: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if self.output_format not in ("json", "csv", "both"):
            raise ValueError(
                f"output_format must be 'json', 'csv', or 'both', got {self.output_format}"
            )
        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
            raise ValueError(
                f"log_level must be 'DEBUG', 'INFO', 'WARNING', or 'ERROR', "
                f"got {self.log_level}"
            )
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError(f"device must be 'cpu', 'cuda', or 'mps', got {self.device}")
        if self.num_workers < 1:
            raise ValueError(f"num_workers must be at least 1, got {self.num_workers}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.batch_size}")

    @classmethod
    def from_defaults(
        cls,
        model_path: str | Path,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> "PipelineConfig":
        """Create a pipeline configuration with sensible defaults.

        Args:
            model_path: Path to face detection model.
            input_path: Path to input video file.
            output_path: Optional path to output video file.

        Returns:
            PipelineConfig with default settings.

        Example:
            >>> config = PipelineConfig.from_defaults(
            ...     model_path="model.task",
            ...     input_path="input.mp4",
            ...     output_path="output.mp4"
            ... )
        """
        return cls(
            face_detection=FaceDetectionConfig(model_path=model_path),
            emotion_analysis=EmotionAnalysisConfig(),
            video=VideoConfig(input_path=input_path, output_path=output_path),
            visualization=VisualizationConfig(),
        )

    @classmethod
    def for_realtime_processing(
        cls,
        model_path: str | Path,
        input_path: str | Path = "0",
    ) -> "PipelineConfig":
        """Create a pipeline configuration optimized for real-time processing.

        Args:
            model_path: Path to face detection model.
            input_path: Path to input video file or camera device ID (default "0").

        Returns:
            PipelineConfig optimized for real-time performance.

        Example:
            >>> config = PipelineConfig.for_realtime_processing(
            ...     model_path="model.task",
            ...     input_path="0"  # Use webcam
            ... )
        """
        return cls(
            face_detection=FaceDetectionConfig(
                model_path=model_path,
                running_mode="VIDEO",
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                enable_smoothing=True,
                smoothing_window_size=3,
            ),
            emotion_analysis=EmotionAnalysisConfig(
                analyzer_type="geometric",
                enable_temporal_smoothing=True,
                smoothing_window_size=5,
            ),
            video=VideoConfig(
                input_path=input_path,
                display_realtime=True,
                skip_frames=0,
            ),
            visualization=VisualizationConfig(
                draw_landmarks=True,
                show_emotion=True,
                show_confidence=False,
                landmark_connections=False,
            ),
            save_annotated_video=False,
            save_emotions=False,
        )

    @classmethod
    def for_batch_processing(
        cls,
        model_path: str | Path,
        input_path: str | Path,
        output_path: str | Path,
        batch_size: int = 8,
    ) -> "PipelineConfig":
        """Create a pipeline configuration optimized for batch processing.

        Args:
            model_path: Path to face detection model.
            input_path: Path to input video file.
            output_path: Path to output video file.
            batch_size: Number of frames to process in parallel.

        Returns:
            PipelineConfig optimized for batch processing.

        Example:
            >>> config = PipelineConfig.for_batch_processing(
            ...     model_path="model.task",
            ...     input_path="input.mp4",
            ...     output_path="output.mp4",
            ...     batch_size=16
            ... )
        """
        return cls(
            face_detection=FaceDetectionConfig(
                model_path=model_path,
                running_mode="IMAGE",
                enable_smoothing=False,
            ),
            emotion_analysis=EmotionAnalysisConfig(
                analyzer_type="geometric",
                enable_temporal_smoothing=False,
            ),
            video=VideoConfig(
                input_path=input_path,
                output_path=output_path,
                display_realtime=False,
                buffer_size=batch_size * 2,
            ),
            visualization=VisualizationConfig(
                draw_landmarks=True,
                show_emotion=True,
                landmark_connections=True,
            ),
            batch_size=batch_size,
            num_workers=4,
            save_annotated_video=True,
            save_emotions=True,
            output_format="both",
        )

    @classmethod
    def for_analysis_only(
        cls,
        model_path: str | Path,
        input_path: str | Path,
    ) -> "PipelineConfig":
        """Create a pipeline configuration for analysis without visualization.

        Args:
            model_path: Path to face detection model.
            input_path: Path to input video file.

        Returns:
            PipelineConfig for data extraction only.

        Example:
            >>> config = PipelineConfig.for_analysis_only(
            ...     model_path="model.task",
            ...     input_path="input.mp4"
            ... )
        """
        return cls(
            face_detection=FaceDetectionConfig(
                model_path=model_path,
                enable_smoothing=False,
            ),
            emotion_analysis=EmotionAnalysisConfig(
                analyzer_type="geometric",
                use_action_units=True,
            ),
            video=VideoConfig(
                input_path=input_path,
                display_realtime=False,
            ),
            visualization=VisualizationConfig(
                draw_landmarks=False,
                draw_bounding_box=False,
                show_emotion=False,
            ),
            save_annotated_video=False,
            save_landmarks=True,
            save_emotions=True,
            output_format="csv",
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Returns:
            Dictionary representation of the configuration.
        """
        return {
            "face_detection": self.face_detection.__dict__,
            "emotion_analysis": self.emotion_analysis.__dict__,
            "video": self.video.__dict__,
            "visualization": self.visualization.__dict__,
            "output_format": self.output_format,
            "save_annotated_video": self.save_annotated_video,
            "save_landmarks": self.save_landmarks,
            "save_emotions": self.save_emotions,
            "log_level": self.log_level,
            "device": self.device,
            "num_workers": self.num_workers,
            "batch_size": self.batch_size,
            "enable_profiling": self.enable_profiling,
            "metadata": self.metadata,
        }
