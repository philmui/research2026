"""Comprehensive tests for the emotion detection pipeline.

This module provides extensive end-to-end testing of the EmotionDetectionPipeline
with mocking to avoid requiring actual model files and videos.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from asdrp.emotion.base import EmotionPrediction, EmotionType
from asdrp.face.base import BoundingBox, FaceLandmarks
from asdrp.pipeline import EmotionDetectionPipeline, PipelineError
from asdrp.utils.config import PipelineConfig
from asdrp.video.frame import FrameData


class TestEmotionDetectionPipelineInitialization:
    """Test suite for pipeline initialization."""

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_initialization_valid(
        self, mock_emotion_analyzer: Mock, mock_face_detector: Mock
    ) -> None:
        """Test valid pipeline initialization."""
        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            pipeline = EmotionDetectionPipeline(config)

            assert pipeline.config == config
            assert pipeline.face_detector is not None
            assert pipeline.emotion_analyzer is not None
            assert pipeline.results == []

            pipeline.close()

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    def test_initialization_failure(self, mock_face_detector: Mock) -> None:
        """Test pipeline initialization failure."""
        mock_face_detector.side_effect = RuntimeError("Initialization failed")

        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            with pytest.raises(PipelineError, match="initialization failed"):
                EmotionDetectionPipeline(config)

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    @patch("asdrp.pipeline.TemporalEmotionAnalyzer")
    def test_initialization_with_temporal_smoothing(
        self,
        mock_temporal: Mock,
        mock_emotion: Mock,
        mock_face: Mock,
    ) -> None:
        """Test initialization with temporal smoothing enabled."""
        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )
        config.emotion_analysis.enable_temporal_smoothing = True

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            pipeline = EmotionDetectionPipeline(config)

            assert pipeline.temporal_analyzer is not None
            pipeline.close()


class TestPipelineFrameProcessing:
    """Test suite for single frame processing."""

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_process_frame_no_faces(
        self,
        mock_emotion_analyzer: Mock,
        mock_face_detector: Mock,
        sample_face_image: np.ndarray,
    ) -> None:
        """Test processing frame with no faces detected."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = []  # No faces
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion_analyzer.return_value = mock_analyzer_instance

        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            pipeline = EmotionDetectionPipeline(config)

            frame_data = FrameData(
                frame=sample_face_image,
                frame_number=0,
                timestamp=0.0,
            )

            result = pipeline.process_frame(frame_data, visualize=False)

            assert result["frame_number"] == 0
            assert result["timestamp"] == 0.0
            assert len(result["faces"]) == 0

            pipeline.close()

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_process_frame_with_face(
        self,
        mock_emotion_analyzer: Mock,
        mock_face_detector: Mock,
        sample_face_image: np.ndarray,
        sample_face_landmarks: FaceLandmarks,
        sample_emotion_probabilities: dict,
    ) -> None:
        """Test processing frame with one face detected."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = [sample_face_landmarks]
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.85,
            probabilities=sample_emotion_probabilities,
        )
        mock_analyzer_instance.analyze.return_value = mock_prediction
        mock_emotion_analyzer.return_value = mock_analyzer_instance

        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            pipeline = EmotionDetectionPipeline(config)

            frame_data = FrameData(
                frame=sample_face_image,
                frame_number=0,
                timestamp=0.0,
            )

            result = pipeline.process_frame(frame_data, visualize=False)

            assert result["frame_number"] == 0
            assert len(result["faces"]) == 1
            assert result["faces"][0]["emotion"] == "happy"
            assert result["faces"][0]["confidence"] == 0.85

            pipeline.close()

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_process_frame_with_visualization(
        self,
        mock_emotion_analyzer: Mock,
        mock_face_detector: Mock,
        sample_face_image: np.ndarray,
        sample_face_landmarks: FaceLandmarks,
        sample_emotion_probabilities: dict,
    ) -> None:
        """Test processing frame with visualization enabled."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = [sample_face_landmarks]
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.85,
            probabilities=sample_emotion_probabilities,
        )
        mock_analyzer_instance.analyze.return_value = mock_prediction
        mock_emotion_analyzer.return_value = mock_analyzer_instance

        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )
        config.save_annotated_video = True

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            pipeline = EmotionDetectionPipeline(config)

            frame_data = FrameData(
                frame=sample_face_image,
                frame_number=0,
                timestamp=0.0,
            )

            result = pipeline.process_frame(frame_data, visualize=True)

            assert "annotated_frame" in result
            assert result["annotated_frame"].shape == sample_face_image.shape

            pipeline.close()


class TestPipelineVideoProcessing:
    """Test suite for video processing."""

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    @patch("asdrp.pipeline.VideoFileReader")
    def test_process_video_basic(
        self,
        mock_reader_class: Mock,
        mock_emotion_analyzer: Mock,
        mock_face_detector: Mock,
        sample_face_image: np.ndarray,
    ) -> None:
        """Test basic video processing."""
        # Setup mock reader
        mock_reader = Mock()
        mock_metadata = Mock()
        mock_metadata.total_frames = 10
        mock_metadata.fps = 30.0
        mock_metadata.width = 640
        mock_metadata.height = 480
        mock_reader.get_metadata.return_value = mock_metadata

        # Mock frame iteration
        frames = [
            FrameData(frame=sample_face_image, frame_number=i, timestamp=i * 0.033)
            for i in range(3)
        ]
        mock_reader.__iter__.return_value = iter(frames)
        mock_reader.__enter__.return_value = mock_reader
        mock_reader.__exit__.return_value = None
        mock_reader_class.return_value = mock_reader

        # Setup face detector
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = []
        mock_face_detector.return_value = mock_detector_instance

        # Setup emotion analyzer
        mock_analyzer_instance = Mock()
        mock_emotion_analyzer.return_value = mock_analyzer_instance

        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            pipeline = EmotionDetectionPipeline(config)
            results = pipeline.process_video(show_progress=False)

            assert len(results) == 3
            assert all("frame_number" in r for r in results)

            pipeline.close()

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    @patch("asdrp.pipeline.VideoFileReader")
    def test_process_video_with_max_frames(
        self,
        mock_reader_class: Mock,
        mock_emotion_analyzer: Mock,
        mock_face_detector: Mock,
        sample_face_image: np.ndarray,
    ) -> None:
        """Test video processing with max_frames limit."""
        # Setup mocks similar to above
        mock_reader = Mock()
        mock_metadata = Mock()
        mock_metadata.total_frames = 100
        mock_metadata.fps = 30.0
        mock_metadata.width = 640
        mock_metadata.height = 480
        mock_reader.get_metadata.return_value = mock_metadata

        # Create more frames than max_frames
        frames = [
            FrameData(frame=sample_face_image, frame_number=i, timestamp=i * 0.033)
            for i in range(20)
        ]
        mock_reader.__iter__.return_value = iter(frames)
        mock_reader.__enter__.return_value = mock_reader
        mock_reader.__exit__.return_value = None
        mock_reader_class.return_value = mock_reader

        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = []
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion_analyzer.return_value = mock_analyzer_instance

        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )
        config.video.max_frames = 5

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            pipeline = EmotionDetectionPipeline(config)
            results = pipeline.process_video(show_progress=False)

            # Should only process max_frames
            assert len(results) <= 5

            pipeline.close()


class TestPipelineResultsSaving:
    """Test suite for saving pipeline results."""

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_save_results_json(
        self,
        mock_emotion_analyzer: Mock,
        mock_face_detector: Mock,
        temp_output_dir: Path,
    ) -> None:
        """Test saving results to JSON format."""
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion_analyzer.return_value = mock_analyzer_instance

        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )
        config.output_format = "json"

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            pipeline = EmotionDetectionPipeline(config)

            # Add some mock results
            pipeline.results = [
                {
                    "frame_number": 0,
                    "timestamp": 0.0,
                    "faces": [
                        {
                            "emotion": "happy",
                            "confidence": 0.8,
                            "probabilities": {"happy": 0.8, "neutral": 0.2},
                        }
                    ],
                }
            ]

            output_path = temp_output_dir / "results.json"
            pipeline.save_results(output_path)

            assert output_path.exists()

            pipeline.close()

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_save_results_empty(
        self,
        mock_emotion_analyzer: Mock,
        mock_face_detector: Mock,
        temp_output_dir: Path,
    ) -> None:
        """Test that saving empty results raises error."""
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion_analyzer.return_value = mock_analyzer_instance

        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            pipeline = EmotionDetectionPipeline(config)

            output_path = temp_output_dir / "results.json"

            with pytest.raises(PipelineError, match="No results to save"):
                pipeline.save_results(output_path)

            pipeline.close()


class TestPipelineContextManager:
    """Test suite for pipeline context manager functionality."""

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_context_manager(
        self, mock_emotion_analyzer: Mock, mock_face_detector: Mock
    ) -> None:
        """Test using pipeline as context manager."""
        mock_detector_instance = Mock()
        mock_detector_instance.close = Mock()
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion_analyzer.return_value = mock_analyzer_instance

        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            with EmotionDetectionPipeline(config) as pipeline:
                assert pipeline is not None

            # Close should be called automatically
            mock_detector_instance.close.assert_called_once()

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_repr(self, mock_emotion_analyzer: Mock, mock_face_detector: Mock) -> None:
        """Test string representation of pipeline."""
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion_analyzer.return_value = mock_analyzer_instance

        config = PipelineConfig.from_defaults(
            model_path="fake_model.task",
            input_path="fake_video.mp4",
        )

        with patch("asdrp.utils.config.Path.exists", return_value=True):
            pipeline = EmotionDetectionPipeline(config)

            repr_str = repr(pipeline)
            assert "EmotionDetectionPipeline" in repr_str
            assert "fake_video.mp4" in repr_str

            pipeline.close()
