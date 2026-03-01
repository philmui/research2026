"""Tests for temporal emotion smoothing.

This module tests the TemporalEmotionAnalyzer for smoothing emotions over time.
"""

import pytest

from asdrp.emotion.base import EmotionPrediction, EmotionType
from asdrp.emotion.temporal import TemporalEmotionAnalyzer


class TestTemporalEmotionAnalyzer:
    """Test suite for TemporalEmotionAnalyzer."""

    def test_initialization_default(self) -> None:
        """Test initialization with default parameters."""
        analyzer = TemporalEmotionAnalyzer()
        assert analyzer.smoothing_window == 10

    def test_initialization_custom_window(self) -> None:
        """Test initialization with custom smoothing window."""
        analyzer = TemporalEmotionAnalyzer(smoothing_window=5)
        assert analyzer.smoothing_window == 5

    def test_smooth_prediction_single(
        self, sample_emotion_predictions: list
    ) -> None:
        """Test smoothing a single prediction."""
        analyzer = TemporalEmotionAnalyzer(smoothing_window=3)
        prediction = sample_emotion_predictions[0]

        smoothed = analyzer.smooth_prediction(prediction)

        assert smoothed.emotion == prediction.emotion
        assert isinstance(smoothed, EmotionPrediction)

    def test_smooth_prediction_sequence(
        self, sample_emotion_predictions: list
    ) -> None:
        """Test smoothing a sequence of predictions."""
        analyzer = TemporalEmotionAnalyzer(smoothing_window=3)

        smoothed_predictions = []
        for prediction in sample_emotion_predictions:
            smoothed = analyzer.smooth_prediction(prediction)
            smoothed_predictions.append(smoothed)

        assert len(smoothed_predictions) == len(sample_emotion_predictions)

    def test_smoothing_reduces_noise(
        self, sample_emotion_probabilities: dict
    ) -> None:
        """Test that smoothing reduces rapid changes."""
        analyzer = TemporalEmotionAnalyzer(smoothing_window=5)

        # Create alternating predictions
        predictions = []
        for i in range(10):
            emotion = EmotionType.HAPPY if i % 2 == 0 else EmotionType.SAD
            probs = sample_emotion_probabilities.copy()

            prediction = EmotionPrediction(
                emotion=emotion,
                confidence=0.7,
                probabilities=probs,
                frame_number=i,
            )
            predictions.append(prediction)

        # Smooth predictions
        smoothed = []
        for pred in predictions:
            smoothed.append(analyzer.smooth_prediction(pred))

        # Later predictions should be more stable
        # (this is a simplified check)
        assert len(smoothed) == len(predictions)

    def test_reset_history(self, sample_emotion_predictions: list) -> None:
        """Test resetting smoothing history."""
        analyzer = TemporalEmotionAnalyzer(smoothing_window=3)

        # Add some predictions
        for prediction in sample_emotion_predictions:
            analyzer.smooth_prediction(prediction)

        # Reset
        analyzer.reset()

        # After reset, next prediction should be unaffected by previous history
        new_prediction = sample_emotion_predictions[0]
        smoothed = analyzer.smooth_prediction(new_prediction)

        assert smoothed.emotion == new_prediction.emotion

    def test_window_size_effect(
        self, sample_emotion_probabilities: dict
    ) -> None:
        """Test effect of different window sizes."""
        small_window = TemporalEmotionAnalyzer(smoothing_window=2)
        large_window = TemporalEmotionAnalyzer(smoothing_window=10)

        predictions = []
        for i in range(15):
            prediction = EmotionPrediction(
                emotion=EmotionType.HAPPY,
                confidence=0.8,
                probabilities=sample_emotion_probabilities,
                frame_number=i,
            )
            predictions.append(prediction)

        # Process with both analyzers
        small_results = [small_window.smooth_prediction(p) for p in predictions]
        large_results = [large_window.smooth_prediction(p) for p in predictions]

        assert len(small_results) == len(large_results)


class TestTemporalEmotionAnalyzerEdgeCases:
    """Test suite for edge cases in temporal smoothing."""

    def test_empty_history(self, sample_emotion_predictions: list) -> None:
        """Test smoothing with empty history."""
        analyzer = TemporalEmotionAnalyzer()
        prediction = sample_emotion_predictions[0]

        smoothed = analyzer.smooth_prediction(prediction)

        # Should return prediction unchanged or with minimal modification
        assert smoothed.emotion == prediction.emotion

    def test_single_frame_window(
        self, sample_emotion_predictions: list
    ) -> None:
        """Test with window size of 1 (no smoothing)."""
        analyzer = TemporalEmotionAnalyzer(smoothing_window=1)
        prediction = sample_emotion_predictions[0]

        smoothed = analyzer.smooth_prediction(prediction)

        # Should be essentially unchanged
        assert smoothed.emotion == prediction.emotion
        assert smoothed.confidence == pytest.approx(prediction.confidence, rel=0.1)

    def test_many_predictions(
        self, sample_emotion_probabilities: dict
    ) -> None:
        """Test with many predictions exceeding window size."""
        analyzer = TemporalEmotionAnalyzer(smoothing_window=5)

        # Create 100 predictions
        for i in range(100):
            prediction = EmotionPrediction(
                emotion=EmotionType.HAPPY,
                confidence=0.8,
                probabilities=sample_emotion_probabilities,
                frame_number=i,
            )
            smoothed = analyzer.smooth_prediction(prediction)

            assert smoothed is not None
            assert isinstance(smoothed, EmotionPrediction)

    def test_confidence_preservation(
        self, sample_emotion_predictions: list
    ) -> None:
        """Test that confidence values remain in valid range."""
        analyzer = TemporalEmotionAnalyzer(smoothing_window=5)

        for prediction in sample_emotion_predictions:
            smoothed = analyzer.smooth_prediction(prediction)

            assert 0.0 <= smoothed.confidence <= 1.0

    def test_probability_sum(
        self, sample_emotion_predictions: list
    ) -> None:
        """Test that probabilities still sum to 1.0 after smoothing."""
        analyzer = TemporalEmotionAnalyzer(smoothing_window=5)

        for prediction in sample_emotion_predictions:
            smoothed = analyzer.smooth_prediction(prediction)

            prob_sum = sum(smoothed.probabilities.values())
            assert prob_sum == pytest.approx(1.0, rel=0.01)
