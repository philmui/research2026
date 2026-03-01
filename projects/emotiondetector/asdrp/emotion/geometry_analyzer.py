"""Geometry-based emotion analyzer using rule-based classification.

This module implements a rule-based emotion classification system using facial
action units (AUs) and geometric features extracted from facial landmarks.
The approach is based on the Facial Action Coding System (FACS) and established
emotion-AU associations from psychological research.

References:
    Ekman, P., & Friesen, W. V. (2003). Unmasking the face: A guide to recognizing
    emotions from facial clues. Malor Books.
"""

from typing import Optional

import numpy as np

from asdrp.emotion.base import (
    ActionUnitType,
    BaseEmotionAnalyzer,
    EmotionPrediction,
    EmotionType,
)
from asdrp.emotion.features import FeatureExtractor
from asdrp.face.base import FaceLandmarks


class GeometryBasedEmotionAnalyzer(BaseEmotionAnalyzer):
    """Rule-based emotion analyzer using geometric features and action units.

    This analyzer classifies emotions based on the presence and intensity of
    specific combinations of facial action units, following established
    emotion-AU associations from FACS research.

    Emotion Detection Rules:
        - Happy: AU6 (cheek raiser) + AU12 (smile)
        - Sad: AU1 (inner brow raise) + AU4 (brow lower) + AU15 (lip corner down)
        - Angry: AU4 (brow lower) + AU7 (lid tighten) + AU23 (lip tighten)
        - Surprised: AU1 + AU2 (brow raise) + AU5 (eye widen) + AU26 (jaw drop)
        - Fearful: AU1 + AU2 + AU4 + AU5 + AU20 (lip stretch)
        - Disgusted: AU9 (nose wrinkle) + AU15 + AU17 (chin raise)

    Attributes:
        feature_extractor: FeatureExtractor instance for AU detection
        emotion_threshold: Minimum confidence for emotion classification
        au_threshold: Threshold for AU presence detection
    """

    # Emotion-AU associations with weights
    # Format: {EmotionType: {ActionUnitType: weight}}
    EMOTION_AU_RULES = {
        EmotionType.HAPPY: {
            ActionUnitType.AU6: 0.5,   # Cheek raiser
            ActionUnitType.AU12: 0.5,  # Smile
        },
        EmotionType.SAD: {
            ActionUnitType.AU1: 0.3,   # Inner brow raise
            ActionUnitType.AU4: 0.4,   # Brow lower
            ActionUnitType.AU15: 0.3,  # Lip corner down
        },
        EmotionType.ANGRY: {
            ActionUnitType.AU4: 0.4,   # Brow lower
            ActionUnitType.AU7: 0.3,   # Lid tighten
            ActionUnitType.AU23: 0.3,  # Lip tighten
        },
        EmotionType.SURPRISED: {
            ActionUnitType.AU1: 0.2,   # Inner brow raise
            ActionUnitType.AU2: 0.2,   # Outer brow raise
            ActionUnitType.AU5: 0.3,   # Eye widen
            ActionUnitType.AU26: 0.3,  # Jaw drop
        },
        EmotionType.FEARFUL: {
            ActionUnitType.AU1: 0.25,  # Inner brow raise
            ActionUnitType.AU2: 0.2,   # Outer brow raise
            ActionUnitType.AU4: 0.15,  # Brow lower (tension)
            ActionUnitType.AU5: 0.2,   # Eye widen
            ActionUnitType.AU20: 0.2,  # Lip stretch
        },
        EmotionType.DISGUSTED: {
            ActionUnitType.AU9: 0.4,   # Nose wrinkle
            ActionUnitType.AU15: 0.3,  # Lip corner down
            ActionUnitType.AU17: 0.3,  # Chin raise
        },
    }

    def __init__(
        self,
        emotion_threshold: float = 0.3,
        au_threshold: float = 0.3,
        neutral_threshold: float = 0.2
    ):
        """Initialize the geometry-based emotion analyzer.

        Args:
            emotion_threshold: Minimum score for emotion classification (0.0-1.0)
            au_threshold: Minimum intensity for AU to be considered present (0.0-1.0)
            neutral_threshold: Maximum score below which emotion is considered neutral
        """
        self.feature_extractor = FeatureExtractor(au_threshold=au_threshold)
        self.emotion_threshold = emotion_threshold
        self.au_threshold = au_threshold
        self.neutral_threshold = neutral_threshold

    def analyze(self, face_landmarks: FaceLandmarks) -> EmotionPrediction:
        """Analyze facial landmarks to predict emotion.

        Args:
            face_landmarks: Facial landmarks extracted from a face

        Returns:
            EmotionPrediction containing the predicted emotion and associated data

        Raises:
            ValueError: If the landmarks are invalid or insufficient for analysis
        """
        if face_landmarks.num_landmarks < 100:
            raise ValueError(
                f"Insufficient landmarks for emotion analysis: "
                f"got {face_landmarks.num_landmarks}, need at least 100"
            )

        # Extract features and detect action units
        features = self.feature_extractor.extract_features(face_landmarks)
        action_units = self.feature_extractor.detect_action_units(face_landmarks)

        # Compute emotion scores
        emotion_scores = self._compute_emotion_scores(action_units)

        # Determine primary emotion
        max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
        primary_emotion = max_emotion[0]
        confidence = max_emotion[1]

        # Check if emotion is strong enough, otherwise classify as neutral
        if confidence < self.neutral_threshold:
            primary_emotion = EmotionType.NEUTRAL
            confidence = 1.0 - max_emotion[1]  # Inverse of strongest emotion

        # Normalize scores to probabilities
        probabilities = self._normalize_probabilities(emotion_scores)

        return EmotionPrediction(
            emotion=primary_emotion,
            confidence=confidence,
            probabilities=probabilities,
            action_units=action_units,
            features=features,
            timestamp=face_landmarks.timestamp,
            frame_number=face_landmarks.frame_number,
            face_landmarks=face_landmarks
        )

    def analyze_batch(
        self, face_landmarks_list: list[FaceLandmarks]
    ) -> list[EmotionPrediction]:
        """Analyze multiple faces to predict emotions.

        Args:
            face_landmarks_list: List of facial landmarks from multiple faces

        Returns:
            List of EmotionPrediction objects, one for each input face

        Raises:
            ValueError: If any landmarks are invalid or insufficient for analysis
        """
        return [self.analyze(landmarks) for landmarks in face_landmarks_list]

    def _compute_emotion_scores(self, action_units: dict) -> dict[EmotionType, float]:
        """Compute emotion scores based on action unit intensities and rules.

        Args:
            action_units: Dictionary of detected action units

        Returns:
            Dictionary mapping each EmotionType to a score (0.0-1.0)
        """
        emotion_scores = {}

        for emotion, au_rules in self.EMOTION_AU_RULES.items():
            score = 0.0
            total_weight = sum(au_rules.values())

            for au_type, weight in au_rules.items():
                if au_type in action_units:
                    au = action_units[au_type]
                    # Weight the AU intensity by its importance for this emotion
                    score += au.intensity * weight

            # Normalize by total weight
            emotion_scores[emotion] = score / total_weight if total_weight > 0 else 0.0

        return emotion_scores

    def _normalize_probabilities(
        self, emotion_scores: dict[EmotionType, float]
    ) -> dict[EmotionType, float]:
        """Normalize emotion scores into a probability distribution.

        Args:
            emotion_scores: Raw emotion scores

        Returns:
            Normalized probabilities that sum to 1.0
        """
        # Include all emotion types, even those not in scores
        all_emotions = list(EmotionType)

        # Apply softmax-like normalization with temperature
        temperature = 2.0  # Higher temperature = softer distribution
        exp_scores = {}

        for emotion in all_emotions:
            score = emotion_scores.get(emotion, 0.0)
            exp_scores[emotion] = np.exp(score / temperature)

        # Normalize
        total = sum(exp_scores.values())
        probabilities = {
            emotion: score / total
            for emotion, score in exp_scores.items()
        }

        return probabilities

    def set_emotion_threshold(self, threshold: float) -> None:
        """Set the emotion classification threshold.

        Args:
            threshold: New threshold value (0.0-1.0)

        Raises:
            ValueError: If threshold is not in valid range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.emotion_threshold = threshold

    def set_au_threshold(self, threshold: float) -> None:
        """Set the action unit detection threshold.

        Args:
            threshold: New threshold value (0.0-1.0)

        Raises:
            ValueError: If threshold is not in valid range
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Threshold must be between 0.0 and 1.0, got {threshold}")
        self.au_threshold = threshold
        self.feature_extractor.au_threshold = threshold

    def get_emotion_description(self, emotion: EmotionType) -> str:
        """Get a description of the action units associated with an emotion.

        Args:
            emotion: Emotion type to describe

        Returns:
            Human-readable description of the emotion's characteristic AUs
        """
        if emotion not in self.EMOTION_AU_RULES:
            return f"No specific AU pattern defined for {emotion}"

        au_rules = self.EMOTION_AU_RULES[emotion]
        au_names = {
            ActionUnitType.AU1: "inner brow raise",
            ActionUnitType.AU2: "outer brow raise",
            ActionUnitType.AU4: "brow lower",
            ActionUnitType.AU5: "eye widen",
            ActionUnitType.AU6: "cheek raise",
            ActionUnitType.AU7: "lid tighten",
            ActionUnitType.AU9: "nose wrinkle",
            ActionUnitType.AU10: "upper lip raise",
            ActionUnitType.AU12: "lip corner pull (smile)",
            ActionUnitType.AU15: "lip corner depress (frown)",
            ActionUnitType.AU17: "chin raise",
            ActionUnitType.AU20: "lip stretch",
            ActionUnitType.AU23: "lip tighten",
            ActionUnitType.AU25: "lips part",
            ActionUnitType.AU26: "jaw drop",
        }

        descriptions = [
            f"{au_names.get(au, str(au))} ({weight:.0%})"
            for au, weight in sorted(au_rules.items(), key=lambda x: x[1], reverse=True)
        ]

        return f"{emotion.value.capitalize()}: {', '.join(descriptions)}"


class EmotionRuleBuilder:
    """Builder class for creating custom emotion detection rules.

    This utility class allows users to define custom emotion-AU associations
    and weights, enabling experimentation with different rule sets.

    Example:
        >>> builder = EmotionRuleBuilder()
        >>> builder.add_rule(EmotionType.HAPPY, ActionUnitType.AU12, weight=0.6)
        >>> builder.add_rule(EmotionType.HAPPY, ActionUnitType.AU6, weight=0.4)
        >>> analyzer = GeometryBasedEmotionAnalyzer()
        >>> analyzer.EMOTION_AU_RULES = builder.build()
    """

    def __init__(self):
        """Initialize an empty rule builder."""
        self.rules: dict[EmotionType, dict[ActionUnitType, float]] = {}

    def add_rule(
        self,
        emotion: EmotionType,
        action_unit: ActionUnitType,
        weight: float
    ) -> "EmotionRuleBuilder":
        """Add or update a rule for an emotion-AU association.

        Args:
            emotion: The emotion type
            action_unit: The action unit type
            weight: The weight/importance of this AU for the emotion (0.0-1.0)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If weight is not in valid range
        """
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"Weight must be between 0.0 and 1.0, got {weight}")

        if emotion not in self.rules:
            self.rules[emotion] = {}

        self.rules[emotion][action_unit] = weight
        return self

    def remove_rule(
        self,
        emotion: EmotionType,
        action_unit: Optional[ActionUnitType] = None
    ) -> "EmotionRuleBuilder":
        """Remove a rule or all rules for an emotion.

        Args:
            emotion: The emotion type
            action_unit: Specific AU to remove, or None to remove all for emotion

        Returns:
            Self for method chaining
        """
        if emotion in self.rules:
            if action_unit is None:
                del self.rules[emotion]
            elif action_unit in self.rules[emotion]:
                del self.rules[emotion][action_unit]

        return self

    def normalize_weights(self, emotion: EmotionType) -> "EmotionRuleBuilder":
        """Normalize weights for an emotion so they sum to 1.0.

        Args:
            emotion: The emotion type to normalize

        Returns:
            Self for method chaining
        """
        if emotion in self.rules and self.rules[emotion]:
            total = sum(self.rules[emotion].values())
            if total > 0:
                self.rules[emotion] = {
                    au: weight / total
                    for au, weight in self.rules[emotion].items()
                }

        return self

    def build(self) -> dict[EmotionType, dict[ActionUnitType, float]]:
        """Build and return the emotion-AU rules dictionary.

        Returns:
            Dictionary of emotion-AU associations with weights
        """
        return dict(self.rules)

    def from_dict(
        self, rules_dict: dict[EmotionType, dict[ActionUnitType, float]]
    ) -> "EmotionRuleBuilder":
        """Load rules from a dictionary.

        Args:
            rules_dict: Dictionary of emotion-AU rules

        Returns:
            Self for method chaining
        """
        self.rules = dict(rules_dict)
        return self
