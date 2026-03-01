"""Unit tests for the visualization module."""

import numpy as np
import pytest

from asdrp.emotion.base import (
    ActionUnit,
    ActionUnitType,
    EmotionPrediction,
    EmotionType,
)
from asdrp.face.base import BoundingBox, FaceLandmarks
from asdrp.visualization import (
    DisplayStyle,
    EmotionDisplay,
    EmotionHeatmap,
    FaceOverlay,
    OverlayStyle,
)


class TestFaceOverlay:
    """Tests for FaceOverlay class."""

    def test_initialization_default(self):
        """Test FaceOverlay initialization with default style."""
        overlay = FaceOverlay()
        assert overlay.style is not None
        assert isinstance(overlay.style, OverlayStyle)

    def test_initialization_custom_style(self):
        """Test FaceOverlay initialization with custom style."""
        style = OverlayStyle(landmark_color=(255, 0, 0), landmark_radius=5)
        overlay = FaceOverlay(style=style)
        assert overlay.style.landmark_color == (255, 0, 0)
        assert overlay.style.landmark_radius == 5

    def test_draw_landmarks(self):
        """Test drawing landmarks on image."""
        # Create test data
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        landmarks = np.random.rand(478, 3).astype(np.float32)
        face_landmarks = FaceLandmarks(landmarks=landmarks)

        overlay = FaceOverlay()
        result = overlay.draw_landmarks(image, face_landmarks)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_draw_bounding_box(self):
        """Test drawing bounding box."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = BoundingBox(x_min=0.2, y_min=0.2, width=0.6, height=0.6)

        overlay = FaceOverlay()
        result = overlay.draw_bounding_box(image, bbox, label="Face")

        assert result.shape == image.shape


class TestEmotionDisplay:
    """Tests for EmotionDisplay class."""

    def test_initialization_default(self):
        """Test EmotionDisplay initialization with default style."""
        display = EmotionDisplay()
        assert display.style is not None
        assert isinstance(display.style, DisplayStyle)

    def test_initialization_custom_style(self):
        """Test EmotionDisplay initialization with custom style."""
        style = DisplayStyle(text_scale=1.0, show_probabilities=False)
        display = EmotionDisplay(style=style)
        assert display.style.text_scale == 1.0
        assert display.style.show_probabilities is False

    def test_draw_emotion_label(self):
        """Test drawing emotion label."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.85,
            probabilities={
                EmotionType.HAPPY: 0.85,
                EmotionType.NEUTRAL: 0.10,
                EmotionType.SAD: 0.05,
            },
        )

        display = EmotionDisplay()
        result = display.draw_emotion_label(image, prediction)

        assert result.shape == image.shape
        assert result.dtype == np.uint8

    def test_draw_probability_bars(self):
        """Test drawing probability bars."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.85,
            probabilities={
                EmotionType.HAPPY: 0.50,
                EmotionType.NEUTRAL: 0.30,
                EmotionType.SAD: 0.20,
            },
        )

        display = EmotionDisplay()
        result = display.draw_probability_bars(image, prediction, top_n=3)

        assert result.shape == image.shape

    def test_draw_action_units(self):
        """Test drawing action units."""
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        action_units = {
            ActionUnitType.AU12: ActionUnit(
                au_type=ActionUnitType.AU12,
                intensity=0.7,
                present=True,
                confidence=0.9,
            ),
        }

        prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.85,
            probabilities={EmotionType.HAPPY: 0.85, EmotionType.NEUTRAL: 0.15},
            action_units=action_units,
        )

        display = EmotionDisplay()
        result = display.draw_action_units(image, prediction)

        assert result.shape == image.shape

    def test_create_emotion_indicator(self):
        """Test creating emotion indicator."""
        display = EmotionDisplay()
        indicator = display.create_emotion_indicator(EmotionType.HAPPY, size=(100, 100))

        assert indicator.shape == (100, 100, 3)
        assert indicator.dtype == np.uint8


class TestEmotionHeatmap:
    """Tests for EmotionHeatmap class."""

    def test_initialization(self):
        """Test EmotionHeatmap initialization."""
        heatmap = EmotionHeatmap(cmap="YlOrRd")
        assert heatmap.cmap == "YlOrRd"

    def test_create_temporal_heatmap(self):
        """Test creating temporal heatmap."""
        # Create test predictions
        predictions = []
        for i in range(50):
            pred = EmotionPrediction(
                emotion=EmotionType.HAPPY,
                confidence=0.8,
                probabilities={
                    EmotionType.HAPPY: 0.50,
                    EmotionType.NEUTRAL: 0.30,
                    EmotionType.SAD: 0.20,
                },
                frame_number=i,
            )
            predictions.append(pred)

        heatmap = EmotionHeatmap()
        fig = heatmap.create_temporal_heatmap(predictions, window_size=10)

        assert fig is not None

    def test_create_transition_heatmap(self):
        """Test creating transition heatmap."""
        # Create test predictions with transitions
        predictions = []
        emotions = [EmotionType.HAPPY, EmotionType.NEUTRAL, EmotionType.SAD]

        for i in range(30):
            emotion = emotions[i % len(emotions)]
            pred = EmotionPrediction(
                emotion=emotion,
                confidence=0.8,
                probabilities={emotion: 0.8, EmotionType.NEUTRAL: 0.2},
                frame_number=i,
            )
            predictions.append(pred)

        heatmap = EmotionHeatmap()
        fig = heatmap.create_transition_heatmap(predictions, normalize=True)

        assert fig is not None

    def test_create_correlation_heatmap(self):
        """Test creating correlation heatmap."""
        predictions = []
        for i in range(50):
            pred = EmotionPrediction(
                emotion=EmotionType.HAPPY,
                confidence=0.8,
                probabilities={
                    EmotionType.HAPPY: 0.50,
                    EmotionType.NEUTRAL: 0.30,
                    EmotionType.SAD: 0.20,
                },
                frame_number=i,
            )
            predictions.append(pred)

        heatmap = EmotionHeatmap()
        fig = heatmap.create_correlation_heatmap(predictions)

        assert fig is not None


class TestPlottingFunctions:
    """Tests for plotting functions."""

    def test_imports(self):
        """Test that all plotting functions can be imported."""
        from asdrp.visualization import (
            plot_action_units,
            plot_confidence_over_time,
            plot_emotion_distribution,
            plot_emotion_probabilities_over_time,
            plot_emotion_summary,
            plot_emotion_timeline,
            plot_emotion_transitions,
        )

        assert plot_emotion_distribution is not None
        assert plot_emotion_timeline is not None
        assert plot_confidence_over_time is not None
        assert plot_action_units is not None
        assert plot_emotion_transitions is not None
        assert plot_emotion_probabilities_over_time is not None
        assert plot_emotion_summary is not None


class TestOverlayStyle:
    """Tests for OverlayStyle configuration."""

    def test_default_values(self):
        """Test OverlayStyle default values."""
        style = OverlayStyle()
        assert style.landmark_color == (0, 255, 0)
        assert style.landmark_radius == 2
        assert style.connection_color == (255, 255, 255)
        assert style.bbox_color == (255, 0, 0)
        assert style.fill_landmarks is True

    def test_custom_values(self):
        """Test OverlayStyle with custom values."""
        style = OverlayStyle(
            landmark_color=(255, 0, 0),
            landmark_radius=5,
            draw_indices=True,
        )
        assert style.landmark_color == (255, 0, 0)
        assert style.landmark_radius == 5
        assert style.draw_indices is True


class TestDisplayStyle:
    """Tests for DisplayStyle configuration."""

    def test_default_values(self):
        """Test DisplayStyle default values."""
        style = DisplayStyle()
        assert style.text_color == (255, 255, 255)
        assert style.text_scale == 0.7
        assert style.show_probabilities is True
        assert style.show_confidence is True
        assert style.position == "top_left"

    def test_custom_values(self):
        """Test DisplayStyle with custom values."""
        style = DisplayStyle(
            text_scale=1.0, show_probabilities=False, position="bottom_right"
        )
        assert style.text_scale == 1.0
        assert style.show_probabilities is False
        assert style.position == "bottom_right"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
