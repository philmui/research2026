"""Tests for emotion detection base classes and data structures.

This module tests EmotionType, EmotionPrediction, ActionUnit, and related
base classes in the emotion.base module.
"""

import pytest

from asdrp.emotion.base import (
    ActionUnit,
    ActionUnitType,
    EmotionPrediction,
    EmotionType,
)


class TestEmotionType:
    """Test suite for EmotionType enum."""

    def test_emotion_values(self) -> None:
        """Test that emotion values are correct strings."""
        assert EmotionType.NEUTRAL.value == "neutral"
        assert EmotionType.HAPPY.value == "happy"
        assert EmotionType.SAD.value == "sad"
        assert EmotionType.ANGRY.value == "angry"
        assert EmotionType.SURPRISED.value == "surprised"
        assert EmotionType.FEARFUL.value == "fearful"
        assert EmotionType.DISGUSTED.value == "disgusted"

    def test_emotion_str_conversion(self) -> None:
        """Test string conversion of EmotionType."""
        assert str(EmotionType.HAPPY) == "happy"
        assert str(EmotionType.SAD) == "sad"

    def test_all_basic_emotions_defined(self) -> None:
        """Test that all six basic emotions plus neutral are defined."""
        emotions = list(EmotionType)
        assert len(emotions) == 7

        expected = {"neutral", "happy", "sad", "angry", "surprised", "fearful", "disgusted"}
        actual = {e.value for e in emotions}
        assert actual == expected


class TestActionUnitType:
    """Test suite for ActionUnitType enum."""

    def test_au_values(self) -> None:
        """Test that action unit values are correct integers."""
        assert ActionUnitType.AU1.value == 1
        assert ActionUnitType.AU2.value == 2
        assert ActionUnitType.AU4.value == 4
        assert ActionUnitType.AU6.value == 6
        assert ActionUnitType.AU12.value == 12

    def test_au_str_conversion(self) -> None:
        """Test string conversion of ActionUnitType."""
        assert str(ActionUnitType.AU1) == "AU1"
        assert str(ActionUnitType.AU12) == "AU12"

    def test_key_action_units_defined(self) -> None:
        """Test that key action units are defined."""
        # Upper face
        assert hasattr(ActionUnitType, "AU1")  # Inner Brow Raiser
        assert hasattr(ActionUnitType, "AU2")  # Outer Brow Raiser
        assert hasattr(ActionUnitType, "AU4")  # Brow Lowerer
        assert hasattr(ActionUnitType, "AU5")  # Upper Lid Raiser
        assert hasattr(ActionUnitType, "AU6")  # Cheek Raiser
        assert hasattr(ActionUnitType, "AU7")  # Lid Tightener

        # Lower face
        assert hasattr(ActionUnitType, "AU12")  # Lip Corner Puller
        assert hasattr(ActionUnitType, "AU15")  # Lip Corner Depressor
        assert hasattr(ActionUnitType, "AU25")  # Lips Part
        assert hasattr(ActionUnitType, "AU26")  # Jaw Drop


class TestActionUnit:
    """Test suite for ActionUnit dataclass."""

    def test_initialization_valid(self) -> None:
        """Test valid ActionUnit initialization."""
        au = ActionUnit(
            au_type=ActionUnitType.AU6,
            intensity=0.7,
            present=True,
            confidence=0.9,
        )

        assert au.au_type == ActionUnitType.AU6
        assert au.intensity == 0.7
        assert au.present is True
        assert au.confidence == 0.9

    def test_initialization_minimal(self) -> None:
        """Test ActionUnit initialization with default confidence."""
        au = ActionUnit(au_type=ActionUnitType.AU12, intensity=0.5, present=True)

        assert au.confidence == 1.0  # Default value

    def test_initialization_invalid_intensity(self) -> None:
        """Test that invalid intensity raises ValueError."""
        with pytest.raises(ValueError, match="intensity must be between"):
            ActionUnit(
                au_type=ActionUnitType.AU6,
                intensity=1.5,  # Invalid
                present=True,
            )

        with pytest.raises(ValueError, match="intensity must be between"):
            ActionUnit(
                au_type=ActionUnitType.AU6,
                intensity=-0.1,  # Invalid
                present=True,
            )

    def test_initialization_invalid_confidence(self) -> None:
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between"):
            ActionUnit(
                au_type=ActionUnitType.AU6,
                intensity=0.5,
                present=True,
                confidence=2.0,  # Invalid
            )


class TestEmotionPrediction:
    """Test suite for EmotionPrediction dataclass."""

    def test_initialization_valid(
        self, sample_emotion_probabilities: dict
    ) -> None:
        """Test valid EmotionPrediction initialization."""
        prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.85,
            probabilities=sample_emotion_probabilities,
            timestamp=1000.0,
            frame_number=10,
        )

        assert prediction.emotion == EmotionType.HAPPY
        assert prediction.confidence == 0.85
        assert prediction.timestamp == 1000.0
        assert prediction.frame_number == 10

    def test_initialization_minimal(
        self, sample_emotion_probabilities: dict
    ) -> None:
        """Test EmotionPrediction with minimal parameters."""
        prediction = EmotionPrediction(
            emotion=EmotionType.NEUTRAL,
            confidence=0.5,
            probabilities=sample_emotion_probabilities,
        )

        assert prediction.timestamp == 0.0
        assert prediction.frame_number == 0
        assert len(prediction.action_units) == 0
        assert len(prediction.features) == 0

    def test_initialization_invalid_confidence(
        self, sample_emotion_probabilities: dict
    ) -> None:
        """Test that invalid confidence raises ValueError."""
        with pytest.raises(ValueError, match="confidence must be between"):
            EmotionPrediction(
                emotion=EmotionType.HAPPY,
                confidence=1.5,
                probabilities=sample_emotion_probabilities,
            )

    def test_initialization_invalid_probabilities(self) -> None:
        """Test that probabilities not summing to 1.0 raises ValueError."""
        # Probabilities don't sum to 1.0
        invalid_probs = {
            EmotionType.HAPPY: 0.5,
            EmotionType.SAD: 0.2,
            EmotionType.ANGRY: 0.1,
        }

        with pytest.raises(ValueError, match="probabilities must sum to 1.0"):
            EmotionPrediction(
                emotion=EmotionType.HAPPY,
                confidence=0.8,
                probabilities=invalid_probs,
            )

    def test_get_top_emotions(
        self, sample_emotion_predictions: list
    ) -> None:
        """Test getting top N emotions by probability."""
        prediction = sample_emotion_predictions[0]
        top_3 = prediction.get_top_emotions(n=3)

        assert len(top_3) == 3
        assert all(isinstance(item, tuple) for item in top_3)
        assert all(len(item) == 2 for item in top_3)

        # Should be sorted by probability descending
        for i in range(len(top_3) - 1):
            assert top_3[i][1] >= top_3[i + 1][1]

    def test_get_top_emotions_all(
        self, sample_emotion_predictions: list
    ) -> None:
        """Test getting all emotions."""
        prediction = sample_emotion_predictions[0]
        all_emotions = prediction.get_top_emotions(n=10)  # More than available

        assert len(all_emotions) == len(EmotionType)

    def test_get_active_action_units(
        self, sample_action_units: dict, sample_emotion_probabilities: dict
    ) -> None:
        """Test getting active action units above threshold."""
        prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.8,
            probabilities=sample_emotion_probabilities,
            action_units=sample_action_units,
        )

        # Default threshold is 0.3
        active_aus = prediction.get_active_action_units()

        # Should only include AU6 (0.8) and AU12 (0.7), not AU4 (0.2)
        assert len(active_aus) == 2
        assert all(au.intensity >= 0.3 for au in active_aus)

    def test_get_active_action_units_custom_threshold(
        self, sample_action_units: dict, sample_emotion_probabilities: dict
    ) -> None:
        """Test getting active AUs with custom threshold."""
        prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.8,
            probabilities=sample_emotion_probabilities,
            action_units=sample_action_units,
        )

        # High threshold
        active_aus = prediction.get_active_action_units(threshold=0.75)
        assert len(active_aus) == 1  # Only AU6 (0.8)

        # Low threshold
        active_aus = prediction.get_active_action_units(threshold=0.1)
        assert len(active_aus) == 3  # All three AUs

    def test_to_dict(
        self, sample_emotion_probabilities: dict
    ) -> None:
        """Test conversion to dictionary for serialization."""
        prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.85,
            probabilities=sample_emotion_probabilities,
            features={"mouth_width": 0.6, "eye_openness": 0.8},
            timestamp=1000.0,
            frame_number=10,
        )

        result = prediction.to_dict()

        assert isinstance(result, dict)
        assert result["emotion"] == "happy"
        assert result["confidence"] == 0.85
        assert result["timestamp"] == 1000.0
        assert result["frame_number"] == 10
        assert "probabilities" in result
        assert "features" in result
        assert "action_units" in result

    def test_to_dict_with_action_units(
        self, sample_action_units: dict, sample_emotion_probabilities: dict
    ) -> None:
        """Test to_dict includes action units."""
        prediction = EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.8,
            probabilities=sample_emotion_probabilities,
            action_units=sample_action_units,
        )

        result = prediction.to_dict()

        assert "action_units" in result
        assert len(result["action_units"]) == len(sample_action_units)
        assert "AU6" in result["action_units"]


class TestBaseEmotionAnalyzer:
    """Test suite for BaseEmotionAnalyzer abstract class."""

    def test_cannot_instantiate_directly(self) -> None:
        """Test that BaseEmotionAnalyzer cannot be instantiated."""
        from asdrp.emotion.base import BaseEmotionAnalyzer

        with pytest.raises(TypeError):
            BaseEmotionAnalyzer()  # type: ignore

    def test_context_manager_protocol(self, mock_emotion_analyzer: any) -> None:
        """Test that analyzer can be used as context manager."""
        with mock_emotion_analyzer as analyzer:
            assert analyzer is not None
