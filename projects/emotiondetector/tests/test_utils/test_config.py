"""Tests for configuration classes.

This module tests the configuration dataclasses used throughout the project.
"""

from pathlib import Path

import pytest

from asdrp.utils.config import (
    EmotionAnalysisConfig,
    FaceDetectionConfig,
    PipelineConfig,
    VideoConfig,
    VisualizationConfig,
)


class TestFaceDetectionConfig:
    """Test suite for FaceDetectionConfig."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        config = FaceDetectionConfig(
            model_path="model.task",
            min_detection_confidence=0.6,
            num_faces=2,
        )

        assert config.model_path == Path("model.task")
        assert config.min_detection_confidence == 0.6
        assert config.num_faces == 2

    def test_initialization_defaults(self) -> None:
        """Test initialization with default values."""
        config = FaceDetectionConfig(model_path="model.task")

        assert config.min_detection_confidence == 0.5
        assert config.min_tracking_confidence == 0.5
        assert config.num_faces == 1
        assert config.running_mode == "VIDEO"

    def test_invalid_detection_confidence(self) -> None:
        """Test that invalid detection confidence raises ValueError."""
        with pytest.raises(ValueError, match="min_detection_confidence"):
            FaceDetectionConfig(
                model_path="model.task",
                min_detection_confidence=1.5,
            )

        with pytest.raises(ValueError, match="min_detection_confidence"):
            FaceDetectionConfig(
                model_path="model.task",
                min_detection_confidence=-0.1,
            )

    def test_invalid_tracking_confidence(self) -> None:
        """Test that invalid tracking confidence raises ValueError."""
        with pytest.raises(ValueError, match="min_tracking_confidence"):
            FaceDetectionConfig(
                model_path="model.task",
                min_tracking_confidence=2.0,
            )

    def test_invalid_num_faces(self) -> None:
        """Test that invalid num_faces raises ValueError."""
        with pytest.raises(ValueError, match="num_faces"):
            FaceDetectionConfig(
                model_path="model.task",
                num_faces=0,
            )

    def test_invalid_running_mode(self) -> None:
        """Test that invalid running mode raises ValueError."""
        with pytest.raises(ValueError, match="running_mode"):
            FaceDetectionConfig(
                model_path="model.task",
                running_mode="INVALID",
            )

    def test_invalid_smoothing_alpha(self) -> None:
        """Test that invalid smoothing alpha raises ValueError."""
        with pytest.raises(ValueError, match="smoothing_alpha"):
            FaceDetectionConfig(
                model_path="model.task",
                smoothing_alpha=1.5,
            )

    def test_path_conversion(self) -> None:
        """Test that model_path is converted to Path object."""
        config = FaceDetectionConfig(model_path="model.task")
        assert isinstance(config.model_path, Path)


class TestEmotionAnalysisConfig:
    """Test suite for EmotionAnalysisConfig."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        config = EmotionAnalysisConfig(
            analyzer_type="geometric",
            confidence_threshold=0.7,
        )

        assert config.analyzer_type == "geometric"
        assert config.confidence_threshold == 0.7

    def test_initialization_defaults(self) -> None:
        """Test initialization with defaults."""
        config = EmotionAnalysisConfig()

        assert config.analyzer_type == "geometric"
        assert config.confidence_threshold == 0.5
        assert config.enable_temporal_smoothing is True

    def test_invalid_confidence_threshold(self) -> None:
        """Test that invalid confidence threshold raises ValueError."""
        with pytest.raises(ValueError, match="confidence_threshold"):
            EmotionAnalysisConfig(confidence_threshold=1.5)

    def test_invalid_smoothing_window(self) -> None:
        """Test that invalid smoothing window raises ValueError."""
        with pytest.raises(ValueError, match="smoothing_window_size"):
            EmotionAnalysisConfig(smoothing_window_size=0)


class TestVideoConfig:
    """Test suite for VideoConfig."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        config = VideoConfig(
            input_path="input.mp4",
            output_path="output.mp4",
            fps=30.0,
        )

        assert config.input_path == Path("input.mp4")
        assert config.output_path == Path("output.mp4")
        assert config.fps == 30.0

    def test_initialization_defaults(self) -> None:
        """Test initialization with defaults."""
        config = VideoConfig(input_path="input.mp4")

        assert config.output_path is None
        assert config.fps is None
        assert config.skip_frames == 0
        assert config.max_frames is None

    def test_invalid_skip_frames(self) -> None:
        """Test that negative skip_frames raises ValueError."""
        with pytest.raises(ValueError, match="skip_frames"):
            VideoConfig(input_path="input.mp4", skip_frames=-1)

    def test_invalid_max_frames(self) -> None:
        """Test that invalid max_frames raises ValueError."""
        with pytest.raises(ValueError, match="max_frames"):
            VideoConfig(input_path="input.mp4", max_frames=0)

    def test_invalid_frame_range(self) -> None:
        """Test that invalid frame range raises ValueError."""
        with pytest.raises(ValueError, match="start_frame must be less than end_frame"):
            VideoConfig(
                input_path="input.mp4",
                start_frame=100,
                end_frame=50,
            )

    def test_path_conversion(self) -> None:
        """Test that paths are converted to Path objects."""
        config = VideoConfig(
            input_path="input.mp4",
            output_path="output.mp4",
        )

        assert isinstance(config.input_path, Path)
        assert isinstance(config.output_path, Path)


class TestVisualizationConfig:
    """Test suite for VisualizationConfig."""

    def test_initialization_valid(self) -> None:
        """Test valid initialization."""
        config = VisualizationConfig(
            draw_landmarks=True,
            draw_bounding_box=True,
            show_emotion=True,
        )

        assert config.draw_landmarks is True
        assert config.draw_bounding_box is True
        assert config.show_emotion is True

    def test_initialization_defaults(self) -> None:
        """Test initialization with defaults."""
        config = VisualizationConfig()

        assert config.draw_landmarks is True
        assert config.show_emotion is True

    def test_invalid_line_thickness(self) -> None:
        """Test that invalid line thickness raises ValueError."""
        with pytest.raises(ValueError, match="line_thickness"):
            VisualizationConfig(line_thickness=0)

    def test_invalid_font_scale(self) -> None:
        """Test that invalid font scale raises ValueError."""
        with pytest.raises(ValueError, match="font_scale"):
            VisualizationConfig(font_scale=0.0)

    def test_invalid_alpha(self) -> None:
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="background_alpha"):
            VisualizationConfig(background_alpha=1.5)

    def test_color_tuples(self) -> None:
        """Test color tuple values."""
        config = VisualizationConfig(
            bbox_color=(255, 0, 0),
            landmark_color=(0, 255, 0),
            text_color=(0, 0, 255),
        )

        assert config.bbox_color == (255, 0, 0)
        assert config.landmark_color == (0, 255, 0)
        assert config.text_color == (0, 0, 255)


class TestPipelineConfig:
    """Test suite for PipelineConfig."""

    def test_initialization_all_configs(self) -> None:
        """Test initialization with all sub-configs."""
        face_config = FaceDetectionConfig(model_path="model.task")
        emotion_config = EmotionAnalysisConfig()
        video_config = VideoConfig(input_path="input.mp4")
        viz_config = VisualizationConfig()

        pipeline_config = PipelineConfig(
            face_detection=face_config,
            emotion_analysis=emotion_config,
            video=video_config,
            visualization=viz_config,
        )

        assert pipeline_config.face_detection == face_config
        assert pipeline_config.emotion_analysis == emotion_config
        assert pipeline_config.video == video_config
        assert pipeline_config.visualization == viz_config

    def test_from_defaults_class_method(self) -> None:
        """Test creating config from defaults."""
        config = PipelineConfig.from_defaults(
            model_path="model.task",
            input_path="input.mp4",
        )

        assert config.face_detection.model_path == Path("model.task")
        assert config.video.input_path == Path("input.mp4")

    def test_for_realtime_processing_class_method(self) -> None:
        """Test creating config for realtime processing."""
        config = PipelineConfig.for_realtime_processing(
            model_path="model.task",
            input_path="0",
        )

        assert config.face_detection.running_mode == "VIDEO"
        assert config.emotion_analysis.enable_temporal_smoothing is True

    def test_to_dict_method(self) -> None:
        """Test conversion to dictionary."""
        config = PipelineConfig.from_defaults(
            model_path="model.task",
            input_path="input.mp4",
        )

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "face_detection" in config_dict
        assert "emotion_analysis" in config_dict
        assert "video" in config_dict
        assert "visualization" in config_dict


class TestConfigValidation:
    """Test suite for configuration validation."""

    def test_consistent_running_mode(self) -> None:
        """Test that configs are internally consistent."""
        config = PipelineConfig.from_defaults(
            model_path="model.task",
            input_path="input.mp4",
        )

        # Running mode should be consistent
        assert config.face_detection.running_mode in ("IMAGE", "VIDEO")

    def test_output_format_validation(self) -> None:
        """Test output format validation."""
        config = PipelineConfig.from_defaults(
            model_path="model.task",
            input_path="input.mp4",
        )

        # Should have valid output format
        assert config.output_format in ("json", "csv", "both")
