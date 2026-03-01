"""Unit tests for the EmotionDetectionPipeline.

This module contains tests for the pipeline orchestration functionality.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np

from asdrp.emotion.base import EmotionPrediction, EmotionType
from asdrp.face.base import BoundingBox, FaceLandmarks
from asdrp.pipeline import EmotionDetectionPipeline, PipelineError
from asdrp.utils.config import PipelineConfig
from asdrp.video.frame import FrameData


class TestPipelineConfiguration(unittest.TestCase):
    """Tests for pipeline configuration."""

    def test_from_defaults(self):
        """Test creating configuration with defaults."""
        config = PipelineConfig.from_defaults(
            model_path="test_model.task",
            input_path="test_input.mp4",
            output_path="test_output.mp4",
        )

        self.assertEqual(config.face_detection.model_path, Path("test_model.task"))
        self.assertEqual(config.video.input_path, Path("test_input.mp4"))
        self.assertEqual(config.video.output_path, Path("test_output.mp4"))

    def test_for_realtime_processing(self):
        """Test creating configuration for real-time processing."""
        config = PipelineConfig.for_realtime_processing(
            model_path="test_model.task", input_path="0"
        )

        self.assertEqual(config.face_detection.running_mode, "VIDEO")
        self.assertTrue(config.face_detection.enable_smoothing)
        self.assertTrue(config.video.display_realtime)
        self.assertFalse(config.save_annotated_video)

    def test_for_batch_processing(self):
        """Test creating configuration for batch processing."""
        config = PipelineConfig.for_batch_processing(
            model_path="test_model.task",
            input_path="test_input.mp4",
            output_path="test_output.mp4",
            batch_size=16,
        )

        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.face_detection.running_mode, "IMAGE")
        self.assertTrue(config.save_annotated_video)
        self.assertEqual(config.output_format, "both")

    def test_for_analysis_only(self):
        """Test creating configuration for analysis only."""
        config = PipelineConfig.for_analysis_only(
            model_path="test_model.task", input_path="test_input.mp4"
        )

        self.assertFalse(config.save_annotated_video)
        self.assertTrue(config.save_landmarks)
        self.assertTrue(config.save_emotions)
        self.assertEqual(config.output_format, "csv")


class TestPipelineInitialization(unittest.TestCase):
    """Tests for pipeline initialization."""

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    @patch("asdrp.pipeline.TemporalEmotionAnalyzer")
    def test_initialization_success(
        self, mock_temporal, mock_emotion, mock_face_detector
    ):
        """Test successful pipeline initialization."""
        config = PipelineConfig.from_defaults(
            model_path="test_model.task",
            input_path="test_input.mp4",
            output_path="test_output.mp4",
        )

        pipeline = EmotionDetectionPipeline(config)

        # Verify components were initialized
        mock_face_detector.assert_called_once()
        mock_emotion.assert_called_once()

        self.assertIsNotNone(pipeline.config)
        self.assertEqual(len(pipeline.results), 0)
        self.assertEqual(pipeline._frame_count, 0)

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    def test_initialization_failure(self, mock_face_detector):
        """Test pipeline initialization failure handling."""
        mock_face_detector.side_effect = Exception("Model load failed")

        config = PipelineConfig.from_defaults(
            model_path="test_model.task",
            input_path="test_input.mp4",
        )

        with self.assertRaises(PipelineError):
            EmotionDetectionPipeline(config)


class TestFrameProcessing(unittest.TestCase):
    """Tests for single frame processing."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PipelineConfig.from_defaults(
            model_path="test_model.task",
            input_path="test_input.mp4",
        )

        # Create mock frame data
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.frame_data = FrameData(
            frame=self.frame, frame_number=0, timestamp=0.0, metadata={}
        )

        # Create mock landmarks
        self.mock_landmarks = FaceLandmarks(
            landmarks=np.random.rand(478, 3).astype(np.float32),
            bounding_box=BoundingBox(x_min=0.2, y_min=0.2, width=0.6, height=0.6),
            timestamp=0.0,
            frame_number=0,
        )

        # Create mock emotion prediction
        self.mock_prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.85,
            probabilities={
                EmotionType.NEUTRAL: 0.05,
                EmotionType.HAPPY: 0.85,
                EmotionType.SAD: 0.02,
                EmotionType.ANGRY: 0.03,
                EmotionType.SURPRISED: 0.03,
                EmotionType.FEARFUL: 0.02,
            },
            action_units={},
            features={},
            timestamp=0.0,
            frame_number=0,
            face_landmarks=self.mock_landmarks,
        )

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    @patch("asdrp.pipeline.TemporalEmotionAnalyzer")
    def test_process_frame_success(
        self, mock_temporal, mock_emotion, mock_face_detector
    ):
        """Test successful frame processing."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = [self.mock_landmarks]
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze.return_value = self.mock_prediction
        mock_emotion.return_value = mock_analyzer_instance

        mock_temporal_instance = Mock()
        mock_temporal_instance.smooth_prediction.return_value = self.mock_prediction
        mock_temporal.return_value = mock_temporal_instance

        # Create pipeline
        pipeline = EmotionDetectionPipeline(self.config)

        # Process frame
        result = pipeline.process_frame(self.frame_data, visualize=False)

        # Verify result structure
        self.assertEqual(result["frame_number"], 0)
        self.assertEqual(result["timestamp"], 0.0)
        self.assertEqual(len(result["faces"]), 1)

        face_result = result["faces"][0]
        self.assertEqual(face_result["emotion"], "happy")
        self.assertAlmostEqual(face_result["confidence"], 0.85)
        self.assertIn("probabilities", face_result)
        self.assertIn("bounding_box", face_result)

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_process_frame_no_faces(self, mock_emotion, mock_face_detector):
        """Test frame processing with no faces detected."""
        # Setup mocks
        mock_detector_instance = Mock()
        mock_detector_instance.detect.return_value = []  # No faces
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion.return_value = mock_analyzer_instance

        # Create pipeline
        pipeline = EmotionDetectionPipeline(self.config)

        # Process frame
        result = pipeline.process_frame(self.frame_data, visualize=False)

        # Verify result structure
        self.assertEqual(result["frame_number"], 0)
        self.assertEqual(len(result["faces"]), 0)


class TestResultStorage(unittest.TestCase):
    """Tests for result storage and retrieval."""

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_get_results(self, mock_emotion, mock_face_detector):
        """Test getting accumulated results."""
        config = PipelineConfig.from_defaults(
            model_path="test_model.task",
            input_path="test_input.mp4",
        )

        # Setup mocks
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion.return_value = mock_analyzer_instance

        pipeline = EmotionDetectionPipeline(config)

        # Initially empty
        results = pipeline.get_results()
        self.assertEqual(len(results), 0)

        # Add some mock results
        pipeline.results.append({"frame_number": 0, "faces": []})
        pipeline.results.append({"frame_number": 1, "faces": []})

        results = pipeline.get_results()
        self.assertEqual(len(results), 2)


class TestResourceManagement(unittest.TestCase):
    """Tests for resource management and cleanup."""

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_context_manager(self, mock_emotion, mock_face_detector):
        """Test pipeline as context manager."""
        config = PipelineConfig.from_defaults(
            model_path="test_model.task",
            input_path="test_input.mp4",
        )

        # Setup mocks
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion.return_value = mock_analyzer_instance

        with EmotionDetectionPipeline(config) as pipeline:
            self.assertIsNotNone(pipeline.face_detector)

        # Verify cleanup was called
        mock_detector_instance.close.assert_called_once()

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_manual_close(self, mock_emotion, mock_face_detector):
        """Test manual pipeline closure."""
        config = PipelineConfig.from_defaults(
            model_path="test_model.task",
            input_path="test_input.mp4",
        )

        # Setup mocks
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion.return_value = mock_analyzer_instance

        pipeline = EmotionDetectionPipeline(config)
        pipeline.close()

        # Verify cleanup was called
        mock_detector_instance.close.assert_called_once()
        self.assertIsNone(pipeline.face_detector)


class TestResultExport(unittest.TestCase):
    """Tests for result export functionality."""

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    @patch("asdrp.pipeline.export_emotions_to_json")
    def test_save_results_json(
        self, mock_export_json, mock_emotion, mock_face_detector
    ):
        """Test saving results to JSON."""
        config = PipelineConfig.from_defaults(
            model_path="test_model.task",
            input_path="test_input.mp4",
        )
        config.output_format = "json"

        # Setup mocks
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion.return_value = mock_analyzer_instance

        pipeline = EmotionDetectionPipeline(config)

        # Add mock results
        pipeline.results.append(
            {
                "frame_number": 0,
                "timestamp": 0.0,
                "faces": [
                    {
                        "emotion": "happy",
                        "confidence": 0.85,
                        "probabilities": {"happy": 0.85, "sad": 0.15},
                    }
                ],
            }
        )

        # Save results
        pipeline.save_results("test_results.json")

        # Verify export was called
        mock_export_json.assert_called_once()

    @patch("asdrp.pipeline.MediaPipeFaceDetector")
    @patch("asdrp.pipeline.GeometryBasedEmotionAnalyzer")
    def test_save_results_empty(self, mock_emotion, mock_face_detector):
        """Test saving empty results raises error."""
        config = PipelineConfig.from_defaults(
            model_path="test_model.task",
            input_path="test_input.mp4",
        )

        # Setup mocks
        mock_detector_instance = Mock()
        mock_face_detector.return_value = mock_detector_instance

        mock_analyzer_instance = Mock()
        mock_emotion.return_value = mock_analyzer_instance

        pipeline = EmotionDetectionPipeline(config)

        # Try to save empty results
        with self.assertRaises(PipelineError):
            pipeline.save_results("test_results.json")


def run_tests():
    """Run all tests."""
    unittest.main()


if __name__ == "__main__":
    run_tests()
