"""Base classes and data structures for emotion detection and analysis.

This module provides the foundational classes for emotion detection, including
enums for emotion types and action units, dataclasses for predictions, and
abstract base classes for emotion analyzers.

References:
    Ekman, P., & Friesen, W. V. (1978). Facial Action Coding System (FACS).
    Consulting Psychologists Press.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, IntEnum
from typing import Optional

import numpy as np
import numpy.typing as npt

from asdrp.face.base import FaceLandmarks


class EmotionType(Enum):
    """Basic emotion types based on Ekman's universal emotions.

    These six basic emotions are considered universally recognized across
    cultures according to Paul Ekman's research on facial expressions.

    References:
        Ekman, P. (1992). An argument for basic emotions. Cognition & Emotion, 6(3-4), 169-200.
    """

    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    SURPRISED = "surprised"
    FEARFUL = "fearful"
    DISGUSTED = "disgusted"

    def __str__(self) -> str:
        """Return the emotion name."""
        return self.value


class ActionUnitType(IntEnum):
    """Facial Action Units from the Facial Action Coding System (FACS).

    Action Units (AUs) are the fundamental actions of individual muscles or
    groups of muscles in facial expressions. This enum includes the most
    commonly used AUs for emotion recognition.

    References:
        Ekman, P., Friesen, W. V., & Hager, J. C. (2002). Facial Action Coding
        System: The Manual. Research Nexus.
    """

    # Upper Face AUs
    AU1 = 1   # Inner Brow Raiser (Frontalis, pars medialis)
    AU2 = 2   # Outer Brow Raiser (Frontalis, pars lateralis)
    AU4 = 4   # Brow Lowerer (Corrugator supercilii, Depressor supercilii)
    AU5 = 5   # Upper Lid Raiser (Levator palpebrae superioris)
    AU6 = 6   # Cheek Raiser (Orbicularis oculi, pars orbitalis)
    AU7 = 7   # Lid Tightener (Orbicularis oculi, pars palpebralis)

    # Lower Face AUs
    AU9 = 9   # Nose Wrinkler (Levator labii superioris alaeque nasi)
    AU10 = 10 # Upper Lip Raiser (Levator labii superioris)
    AU12 = 12 # Lip Corner Puller (Zygomaticus major)
    AU15 = 15 # Lip Corner Depressor (Depressor anguli oris)
    AU17 = 17 # Chin Raiser (Mentalis)
    AU20 = 20 # Lip Stretcher (Risorius)
    AU23 = 23 # Lip Tightener (Orbicularis oris)
    AU25 = 25 # Lips Part (Depressor labii inferioris, Relaxation of Mentalis/Orbicularis oris)
    AU26 = 26 # Jaw Drop (Masseter, Temporal, Internal Pterygoid relaxed)
    AU27 = 27 # Mouth Stretch (Pterygoids, Digastric)

    def __str__(self) -> str:
        """Return the AU name with number."""
        return f"AU{self.value}"


@dataclass
class ActionUnit:
    """Facial Action Unit detection result.

    Represents the detection and intensity of a specific facial action unit,
    which corresponds to the movement of specific facial muscles.

    Attributes:
        au_type: The type of action unit detected
        intensity: Normalized intensity value (0.0 to 1.0), where 0 is absent
                  and 1.0 is maximum intensity
        present: Boolean indicating if the AU is considered present based on
                a threshold (typically intensity > 0.3)
        confidence: Confidence score of the detection (0.0 to 1.0)
    """

    au_type: ActionUnitType
    intensity: float
    present: bool
    confidence: float = 1.0

    def __post_init__(self) -> None:
        """Validate intensity and confidence values."""
        if not 0.0 <= self.intensity <= 1.0:
            raise ValueError(f"intensity must be between 0.0 and 1.0, got {self.intensity}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")


@dataclass
class EmotionPrediction:
    """Emotion prediction result from analysis.

    Contains the predicted emotion along with confidence scores, probability
    distribution over all emotions, detected action units, and metadata.

    Attributes:
        emotion: The predicted primary emotion
        confidence: Overall confidence score for the prediction (0.0 to 1.0)
        probabilities: Dictionary mapping each EmotionType to its probability
        action_units: Dictionary mapping ActionUnitType to ActionUnit objects
        features: Dictionary of extracted geometric features used for prediction
        timestamp: Timestamp when the prediction was made
        frame_number: Frame number in video sequence (0 for single images)
        face_landmarks: Optional reference to the facial landmarks used for analysis
    """

    emotion: EmotionType
    confidence: float
    probabilities: dict[EmotionType, float]
    action_units: dict[ActionUnitType, ActionUnit] = field(default_factory=dict)
    features: dict[str, float] = field(default_factory=dict)
    timestamp: float = 0.0
    frame_number: int = 0
    face_landmarks: Optional[FaceLandmarks] = None

    def __post_init__(self) -> None:
        """Validate confidence and probability values."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

        # Validate that probabilities sum to approximately 1.0
        if self.probabilities:
            prob_sum = sum(self.probabilities.values())
            if not 0.99 <= prob_sum <= 1.01:
                raise ValueError(f"probabilities must sum to 1.0, got {prob_sum}")

    def get_top_emotions(self, n: int = 3) -> list[tuple[EmotionType, float]]:
        """Get the top N emotions by probability.

        Args:
            n: Number of top emotions to return

        Returns:
            List of (emotion, probability) tuples, sorted by probability descending
        """
        sorted_emotions = sorted(
            self.probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions[:n]

    def get_active_action_units(self, threshold: float = 0.3) -> list[ActionUnit]:
        """Get action units that are considered active.

        Args:
            threshold: Minimum intensity threshold for an AU to be considered active

        Returns:
            List of ActionUnit objects with intensity >= threshold
        """
        return [
            au for au in self.action_units.values()
            if au.intensity >= threshold
        ]

    def to_dict(self) -> dict:
        """Convert prediction to a dictionary for serialization.

        Returns:
            Dictionary representation of the prediction
        """
        return {
            "emotion": self.emotion.value,
            "confidence": float(self.confidence),
            "probabilities": {e.value: float(p) for e, p in self.probabilities.items()},
            "action_units": {
                str(au.au_type): {
                    "intensity": float(au.intensity),
                    "present": au.present,
                    "confidence": float(au.confidence)
                }
                for au in self.action_units.values()
            },
            "features": {k: float(v) for k, v in self.features.items()},
            "timestamp": float(self.timestamp),
            "frame_number": int(self.frame_number)
        }


class BaseEmotionAnalyzer(ABC):
    """Abstract base class for emotion analyzers.

    Defines the interface that all emotion analysis implementations must follow.
    Implementations may use different approaches such as rule-based systems,
    machine learning models, or hybrid approaches.
    """

    @abstractmethod
    def analyze(self, face_landmarks: FaceLandmarks) -> EmotionPrediction:
        """Analyze facial landmarks to predict emotion.

        Args:
            face_landmarks: Facial landmarks extracted from a face

        Returns:
            EmotionPrediction containing the predicted emotion and associated data

        Raises:
            ValueError: If the landmarks are invalid or insufficient for analysis
            RuntimeError: If the analyzer is not properly initialized
        """
        pass

    @abstractmethod
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
            RuntimeError: If the analyzer is not properly initialized
        """
        pass

    def __enter__(self) -> "BaseEmotionAnalyzer":
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """Context manager exit point."""
        pass
