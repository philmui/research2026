"""Temporal emotion analysis with smoothing and stability detection.

This module provides classes for analyzing emotions over time, including
smoothing predictions, detecting stable emotion states, and identifying
microexpressions (brief, subtle emotional expressions).
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from asdrp.emotion.base import EmotionPrediction, EmotionType
from asdrp.face.base import FaceLandmarks


@dataclass
class EmotionState:
    """Represents a stable emotion state over a period of time.

    Attributes:
        emotion: The emotion type
        start_time: Start timestamp in milliseconds
        end_time: End timestamp in milliseconds
        start_frame: Starting frame number
        end_frame: Ending frame number
        average_confidence: Average confidence during this state
        peak_confidence: Maximum confidence during this state
        predictions: List of predictions that make up this state
    """

    emotion: EmotionType
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    average_confidence: float
    peak_confidence: float
    predictions: list[EmotionPrediction] = field(default_factory=list)

    @property
    def duration(self) -> float:
        """Duration of this emotion state in seconds."""
        return (self.end_time - self.start_time) / 1000.0

    @property
    def frame_count(self) -> int:
        """Number of frames in this emotion state."""
        return self.end_frame - self.start_frame + 1


@dataclass
class Microexpression:
    """Represents a brief, subtle emotional expression.

    Microexpressions are involuntary facial expressions that occur in less than
    0.5 seconds, often revealing concealed emotions.

    Attributes:
        emotion: The microexpression emotion
        timestamp: Timestamp when detected
        frame_number: Frame number where detected
        duration: Duration in seconds
        intensity: Intensity/strength of the microexpression
        prediction: The emotion prediction that triggered detection
    """

    emotion: EmotionType
    timestamp: float
    frame_number: int
    duration: float
    intensity: float
    prediction: EmotionPrediction


class TemporalEmotionAnalyzer:
    """Analyzer for temporal patterns in emotion sequences.

    This class provides methods for smoothing emotion predictions over time,
    detecting stable emotion states, and identifying microexpressions using
    temporal filtering and hysteresis.

    Attributes:
        window_size: Number of frames for moving average smoothing
        hysteresis_threshold: Confidence difference required to change emotion
        min_state_duration: Minimum duration (seconds) for a stable emotion state
        microexpression_duration: Maximum duration (seconds) for microexpressions
        history: Buffer storing recent predictions for temporal analysis
    """

    def __init__(
        self,
        window_size: int = 5,
        hysteresis_threshold: float = 0.15,
        min_state_duration: float = 0.5,
        microexpression_duration: float = 0.5
    ):
        """Initialize the temporal emotion analyzer.

        Args:
            window_size: Size of moving average window (number of frames)
            hysteresis_threshold: Confidence difference needed to change emotion
            min_state_duration: Minimum duration for stable emotion states (seconds)
            microexpression_duration: Maximum duration for microexpressions (seconds)
        """
        self.window_size = window_size
        self.hysteresis_threshold = hysteresis_threshold
        self.min_state_duration = min_state_duration
        self.microexpression_duration = microexpression_duration

        # History buffer
        self.history: deque[EmotionPrediction] = deque(maxlen=window_size)

        # Current state tracking
        self.current_emotion: Optional[EmotionType] = None
        self.current_state_start: Optional[float] = None
        self.current_state_frame: Optional[int] = None

        # Detected states and microexpressions
        self.emotion_states: list[EmotionState] = []
        self.microexpressions: list[Microexpression] = []

    def smooth_prediction(self, prediction: EmotionPrediction) -> EmotionPrediction:
        """Apply temporal smoothing to an emotion prediction.

        Uses moving average of probability distributions and hysteresis to
        reduce jitter and create more stable predictions.

        Args:
            prediction: The new emotion prediction to smooth

        Returns:
            Smoothed emotion prediction
        """
        # Add to history
        self.history.append(prediction)

        if len(self.history) < 2:
            # Not enough history, return as-is but initialize state
            self.current_emotion = prediction.emotion
            self.current_state_start = prediction.timestamp
            self.current_state_frame = prediction.frame_number
            return prediction

        # Compute smoothed probabilities using moving average
        smoothed_probs = self._compute_moving_average_probabilities()

        # Determine smoothed emotion
        smoothed_emotion = max(smoothed_probs.items(), key=lambda x: x[1])[0]
        smoothed_confidence = smoothed_probs[smoothed_emotion]

        # Apply hysteresis: only change emotion if confidence difference is significant
        if self.current_emotion is not None and smoothed_emotion != self.current_emotion:
            current_prob = smoothed_probs[self.current_emotion]
            new_prob = smoothed_probs[smoothed_emotion]

            # Require new emotion to be significantly more confident
            if new_prob - current_prob < self.hysteresis_threshold:
                smoothed_emotion = self.current_emotion
                smoothed_confidence = current_prob

        # Check for state transition
        if smoothed_emotion != self.current_emotion:
            self._handle_state_transition(prediction, smoothed_emotion)
        else:
            self.current_emotion = smoothed_emotion

        # Create smoothed prediction
        smoothed_prediction = EmotionPrediction(
            emotion=smoothed_emotion,
            confidence=smoothed_confidence,
            probabilities=smoothed_probs,
            action_units=prediction.action_units,
            features=prediction.features,
            timestamp=prediction.timestamp,
            frame_number=prediction.frame_number,
            face_landmarks=prediction.face_landmarks
        )

        return smoothed_prediction

    def detect_microexpression(self, prediction: EmotionPrediction) -> Optional[Microexpression]:
        """Detect if a prediction represents a microexpression.

        Microexpressions are brief emotional expressions that differ from the
        current stable emotion state.

        Args:
            prediction: The emotion prediction to analyze

        Returns:
            Microexpression object if detected, None otherwise
        """
        if len(self.history) < self.window_size:
            return None

        # Check if this prediction differs from current stable emotion
        if self.current_emotion is None or prediction.emotion == self.current_emotion:
            return None

        # Check if duration is brief enough
        if self.current_state_start is not None:
            current_duration = (prediction.timestamp - self.current_state_start) / 1000.0
            if current_duration > self.microexpression_duration:
                return None

        # Calculate intensity based on confidence and AU activations
        active_aus = prediction.get_active_action_units()
        au_intensity = np.mean([au.intensity for au in active_aus]) if active_aus else 0.0
        intensity = (prediction.confidence + au_intensity) / 2.0

        # Create microexpression if intensity is significant
        if intensity > 0.4:
            microexp = Microexpression(
                emotion=prediction.emotion,
                timestamp=prediction.timestamp,
                frame_number=prediction.frame_number,
                duration=current_duration if self.current_state_start else 0.0,
                intensity=intensity,
                prediction=prediction
            )

            self.microexpressions.append(microexp)
            return microexp

        return None

    def get_emotion_states(
        self,
        min_duration: Optional[float] = None
    ) -> list[EmotionState]:
        """Get detected emotion states, optionally filtered by duration.

        Args:
            min_duration: Minimum duration in seconds (uses default if None)

        Returns:
            List of detected emotion states
        """
        if min_duration is None:
            min_duration = self.min_state_duration

        return [
            state for state in self.emotion_states
            if state.duration >= min_duration
        ]

    def get_microexpressions(
        self,
        emotion: Optional[EmotionType] = None
    ) -> list[Microexpression]:
        """Get detected microexpressions, optionally filtered by emotion.

        Args:
            emotion: Filter by specific emotion type (returns all if None)

        Returns:
            List of detected microexpressions
        """
        if emotion is None:
            return list(self.microexpressions)

        return [me for me in self.microexpressions if me.emotion == emotion]

    def reset(self) -> None:
        """Reset the analyzer state and clear history."""
        self.history.clear()
        self.current_emotion = None
        self.current_state_start = None
        self.current_state_frame = None
        self.emotion_states.clear()
        self.microexpressions.clear()

    def _compute_moving_average_probabilities(self) -> dict[EmotionType, float]:
        """Compute moving average of emotion probabilities.

        Returns:
            Dictionary of averaged probabilities for each emotion
        """
        # Initialize with all emotion types
        all_emotions = list(EmotionType)
        prob_sums = {emotion: 0.0 for emotion in all_emotions}

        # Sum probabilities across history
        for pred in self.history:
            for emotion in all_emotions:
                prob_sums[emotion] += pred.probabilities.get(emotion, 0.0)

        # Compute averages
        count = len(self.history)
        averaged_probs = {
            emotion: prob_sum / count
            for emotion, prob_sum in prob_sums.items()
        }

        return averaged_probs

    def _handle_state_transition(
        self,
        prediction: EmotionPrediction,
        new_emotion: EmotionType
    ) -> None:
        """Handle transition from current emotion to a new emotion.

        Args:
            prediction: The prediction triggering the transition
            new_emotion: The new emotion state
        """
        # Finalize previous state if it exists
        if (self.current_emotion is not None and
            self.current_state_start is not None and
            self.current_state_frame is not None):

            # Get predictions for the finished state
            state_predictions = [
                p for p in self.history
                if p.emotion == self.current_emotion and
                p.timestamp >= self.current_state_start
            ]

            if state_predictions:
                confidences = [p.confidence for p in state_predictions]
                state = EmotionState(
                    emotion=self.current_emotion,
                    start_time=self.current_state_start,
                    end_time=prediction.timestamp,
                    start_frame=self.current_state_frame,
                    end_frame=prediction.frame_number - 1,
                    average_confidence=float(np.mean(confidences)),
                    peak_confidence=float(np.max(confidences)),
                    predictions=state_predictions
                )

                self.emotion_states.append(state)

        # Start new state
        self.current_emotion = new_emotion
        self.current_state_start = prediction.timestamp
        self.current_state_frame = prediction.frame_number


class TemporalFilter:
    """Utility class for applying various temporal filters to emotion sequences.

    Provides static methods for different filtering approaches including
    median filtering, exponential smoothing, and Kalman-inspired filtering.
    """

    @staticmethod
    def median_filter(
        predictions: list[EmotionPrediction],
        window_size: int = 5
    ) -> list[EmotionPrediction]:
        """Apply median filtering to emotion predictions.

        Args:
            predictions: List of predictions to filter
            window_size: Size of the median filter window (must be odd)

        Returns:
            Filtered list of predictions

        Raises:
            ValueError: If window_size is not odd
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")

        if len(predictions) < window_size:
            return predictions

        filtered = []
        half_window = window_size // 2

        for i in range(len(predictions)):
            # Get window of predictions
            start = max(0, i - half_window)
            end = min(len(predictions), i + half_window + 1)
            window = predictions[start:end]

            # Find median emotion (most common in window)
            emotion_counts = {}
            for pred in window:
                emotion_counts[pred.emotion] = emotion_counts.get(pred.emotion, 0) + 1

            median_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

            # Create filtered prediction
            filtered_pred = EmotionPrediction(
                emotion=median_emotion,
                confidence=predictions[i].confidence,
                probabilities=predictions[i].probabilities,
                action_units=predictions[i].action_units,
                features=predictions[i].features,
                timestamp=predictions[i].timestamp,
                frame_number=predictions[i].frame_number,
                face_landmarks=predictions[i].face_landmarks
            )

            filtered.append(filtered_pred)

        return filtered

    @staticmethod
    def exponential_smoothing(
        predictions: list[EmotionPrediction],
        alpha: float = 0.3
    ) -> list[EmotionPrediction]:
        """Apply exponential smoothing to emotion probabilities.

        Args:
            predictions: List of predictions to smooth
            alpha: Smoothing factor (0 = no smoothing, 1 = no memory)

        Returns:
            Smoothed list of predictions

        Raises:
            ValueError: If alpha is not in [0, 1]
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha must be between 0.0 and 1.0")

        if not predictions:
            return []

        smoothed = []
        smoothed_probs = dict(predictions[0].probabilities)

        for pred in predictions:
            # Update smoothed probabilities
            for emotion in EmotionType:
                current_prob = pred.probabilities.get(emotion, 0.0)
                smoothed_probs[emotion] = (
                    alpha * current_prob +
                    (1 - alpha) * smoothed_probs.get(emotion, 0.0)
                )

            # Normalize
            total = sum(smoothed_probs.values())
            if total > 0:
                smoothed_probs = {e: p / total for e, p in smoothed_probs.items()}

            # Determine smoothed emotion
            smoothed_emotion = max(smoothed_probs.items(), key=lambda x: x[1])[0]
            smoothed_confidence = smoothed_probs[smoothed_emotion]

            # Create smoothed prediction
            smoothed_pred = EmotionPrediction(
                emotion=smoothed_emotion,
                confidence=smoothed_confidence,
                probabilities=dict(smoothed_probs),
                action_units=pred.action_units,
                features=pred.features,
                timestamp=pred.timestamp,
                frame_number=pred.frame_number,
                face_landmarks=pred.face_landmarks
            )

            smoothed.append(smoothed_pred)

        return smoothed

    @staticmethod
    def remove_transients(
        predictions: list[EmotionPrediction],
        min_duration: float = 0.3
    ) -> list[EmotionPrediction]:
        """Remove transient emotion spikes that last less than min_duration.

        Args:
            predictions: List of predictions to filter
            min_duration: Minimum duration (seconds) for an emotion to be kept

        Returns:
            Filtered list of predictions with transients removed
        """
        if len(predictions) < 2:
            return predictions

        filtered = []
        current_emotion = predictions[0].emotion
        emotion_start_time = predictions[0].timestamp
        segment_start_idx = 0

        for i in range(1, len(predictions)):
            if predictions[i].emotion != current_emotion:
                # Check duration of previous emotion segment
                duration = (predictions[i].timestamp - emotion_start_time) / 1000.0

                if duration >= min_duration:
                    # Keep segment
                    filtered.extend(predictions[segment_start_idx:i])
                else:
                    # Replace transient with most common neighboring emotion
                    if filtered:
                        replacement_emotion = filtered[-1].emotion
                    else:
                        replacement_emotion = predictions[i].emotion

                    for j in range(segment_start_idx, i):
                        replaced = EmotionPrediction(
                            emotion=replacement_emotion,
                            confidence=predictions[j].confidence,
                            probabilities=predictions[j].probabilities,
                            action_units=predictions[j].action_units,
                            features=predictions[j].features,
                            timestamp=predictions[j].timestamp,
                            frame_number=predictions[j].frame_number,
                            face_landmarks=predictions[j].face_landmarks
                        )
                        filtered.append(replaced)

                # Start new segment
                current_emotion = predictions[i].emotion
                emotion_start_time = predictions[i].timestamp
                segment_start_idx = i

        # Handle last segment
        filtered.extend(predictions[segment_start_idx:])

        return filtered
