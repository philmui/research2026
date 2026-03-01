"""Metrics and statistics for emotion analysis.

This module provides classes and functions for aggregating, analyzing, and
computing statistics on emotion predictions over time. It includes tools for
tracking emotion distributions, transitions, and temporal patterns.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import numpy.typing as npt

from asdrp.emotion.base import ActionUnitType, EmotionPrediction, EmotionType


@dataclass
class EmotionMetrics:
    """Aggregated metrics for emotion predictions over a sequence.

    This class computes and stores various statistics about emotions detected
    in a sequence of frames, useful for analyzing emotional patterns over time.

    Attributes:
        total_frames: Total number of frames analyzed
        emotion_counts: Count of frames for each emotion type
        emotion_durations: Total duration (in seconds) for each emotion
        average_confidences: Average confidence for each emotion type
        dominant_emotion: Most frequently occurring emotion
        emotion_distribution: Normalized distribution of emotions (probabilities)
        transition_matrix: Matrix showing emotion-to-emotion transitions
        average_au_intensities: Average intensity of each action unit
        timestamp_start: Start timestamp of the sequence
        timestamp_end: End timestamp of the sequence
    """

    total_frames: int = 0
    emotion_counts: dict[EmotionType, int] = field(default_factory=dict)
    emotion_durations: dict[EmotionType, float] = field(default_factory=dict)
    average_confidences: dict[EmotionType, float] = field(default_factory=dict)
    dominant_emotion: Optional[EmotionType] = None
    emotion_distribution: dict[EmotionType, float] = field(default_factory=dict)
    transition_matrix: dict[tuple[EmotionType, EmotionType], int] = field(default_factory=dict)
    average_au_intensities: dict[ActionUnitType, float] = field(default_factory=dict)
    timestamp_start: float = 0.0
    timestamp_end: float = 0.0

    @property
    def duration(self) -> float:
        """Total duration of the analyzed sequence in seconds."""
        return (self.timestamp_end - self.timestamp_start) / 1000.0  # Convert ms to seconds

    @property
    def fps(self) -> float:
        """Estimated frames per second of the sequence."""
        if self.duration > 0:
            return self.total_frames / self.duration
        return 0.0

    def get_emotion_percentage(self, emotion: EmotionType) -> float:
        """Get the percentage of frames with a specific emotion.

        Args:
            emotion: The emotion type to query

        Returns:
            Percentage (0-100) of frames with this emotion
        """
        if self.total_frames == 0:
            return 0.0
        count = self.emotion_counts.get(emotion, 0)
        return (count / self.total_frames) * 100.0

    def get_emotion_duration(self, emotion: EmotionType) -> float:
        """Get the total duration of a specific emotion in seconds.

        Args:
            emotion: The emotion type to query

        Returns:
            Total duration in seconds
        """
        return self.emotion_durations.get(emotion, 0.0)

    def get_transition_probability(
        self, from_emotion: EmotionType, to_emotion: EmotionType
    ) -> float:
        """Get the probability of transitioning from one emotion to another.

        Args:
            from_emotion: Starting emotion
            to_emotion: Target emotion

        Returns:
            Transition probability (0.0-1.0)
        """
        transitions_from = sum(
            count for (e1, e2), count in self.transition_matrix.items()
            if e1 == from_emotion
        )

        if transitions_from == 0:
            return 0.0

        transition_count = self.transition_matrix.get((from_emotion, to_emotion), 0)
        return transition_count / transitions_from

    def to_dict(self) -> dict:
        """Convert metrics to a dictionary for serialization.

        Returns:
            Dictionary representation of metrics
        """
        return {
            "total_frames": self.total_frames,
            "duration": self.duration,
            "fps": self.fps,
            "emotion_counts": {e.value: count for e, count in self.emotion_counts.items()},
            "emotion_durations": {e.value: dur for e, dur in self.emotion_durations.items()},
            "emotion_distribution": {e.value: prob for e, prob in self.emotion_distribution.items()},
            "average_confidences": {e.value: conf for e, conf in self.average_confidences.items()},
            "dominant_emotion": self.dominant_emotion.value if self.dominant_emotion else None,
            "average_au_intensities": {str(au): intensity for au, intensity in self.average_au_intensities.items()},
        }


def compute_emotion_metrics(predictions: list[EmotionPrediction]) -> EmotionMetrics:
    """Compute aggregated metrics from a list of emotion predictions.

    Args:
        predictions: List of emotion predictions to analyze

    Returns:
        EmotionMetrics object with computed statistics

    Raises:
        ValueError: If predictions list is empty
    """
    if not predictions:
        raise ValueError("Cannot compute metrics from empty predictions list")

    metrics = EmotionMetrics()
    metrics.total_frames = len(predictions)

    # Initialize counters
    emotion_counts: Counter = Counter()
    emotion_confidences: defaultdict = defaultdict(list)
    emotion_timestamps: defaultdict = defaultdict(list)
    au_intensities: defaultdict = defaultdict(list)

    # Track previous emotion for transitions
    prev_emotion: Optional[EmotionType] = None

    # Set time bounds
    metrics.timestamp_start = predictions[0].timestamp
    metrics.timestamp_end = predictions[-1].timestamp

    # Aggregate data
    for pred in predictions:
        # Count emotions
        emotion_counts[pred.emotion] += 1

        # Collect confidences
        emotion_confidences[pred.emotion].append(pred.confidence)

        # Collect timestamps for duration calculation
        emotion_timestamps[pred.emotion].append(pred.timestamp)

        # Collect AU intensities
        for au_type, au in pred.action_units.items():
            au_intensities[au_type].append(au.intensity)

        # Track transitions
        if prev_emotion is not None and prev_emotion != pred.emotion:
            transition_key = (prev_emotion, pred.emotion)
            metrics.transition_matrix[transition_key] = (
                metrics.transition_matrix.get(transition_key, 0) + 1
            )

        prev_emotion = pred.emotion

    # Compute emotion statistics
    metrics.emotion_counts = dict(emotion_counts)

    # Dominant emotion
    metrics.dominant_emotion = emotion_counts.most_common(1)[0][0]

    # Average confidences
    for emotion, confidences in emotion_confidences.items():
        metrics.average_confidences[emotion] = float(np.mean(confidences))

    # Emotion distribution
    total_count = sum(emotion_counts.values())
    metrics.emotion_distribution = {
        emotion: count / total_count
        for emotion, count in emotion_counts.items()
    }

    # Compute emotion durations (approximate)
    # Use time difference between first and last occurrence
    for emotion, timestamps in emotion_timestamps.items():
        if len(timestamps) > 1:
            duration = (max(timestamps) - min(timestamps)) / 1000.0  # Convert ms to seconds
        else:
            # Single frame: estimate duration based on fps
            duration = 1.0 / metrics.fps if metrics.fps > 0 else 0.0

        metrics.emotion_durations[emotion] = duration

    # Average AU intensities
    for au_type, intensities in au_intensities.items():
        metrics.average_au_intensities[au_type] = float(np.mean(intensities))

    return metrics


def compute_emotion_distribution(
    predictions: list[EmotionPrediction],
    normalize: bool = True
) -> dict[EmotionType, float]:
    """Compute the distribution of emotions in predictions.

    Args:
        predictions: List of emotion predictions
        normalize: If True, return normalized probabilities; otherwise return counts

    Returns:
        Dictionary mapping each emotion to its frequency or probability
    """
    if not predictions:
        return {}

    emotion_counts: Counter = Counter(pred.emotion for pred in predictions)

    if normalize:
        total = sum(emotion_counts.values())
        return {emotion: count / total for emotion, count in emotion_counts.items()}
    else:
        return dict(emotion_counts)


def detect_emotion_transitions(
    predictions: list[EmotionPrediction],
    min_duration: float = 0.5
) -> list[tuple[EmotionType, EmotionType, float]]:
    """Detect significant emotion transitions in a sequence.

    Args:
        predictions: List of emotion predictions in temporal order
        min_duration: Minimum duration (seconds) for an emotion to be considered stable

    Returns:
        List of (from_emotion, to_emotion, timestamp) tuples for transitions
    """
    if len(predictions) < 2:
        return []

    transitions = []
    current_emotion = predictions[0].emotion
    current_start_time = predictions[0].timestamp

    for pred in predictions[1:]:
        if pred.emotion != current_emotion:
            # Check if previous emotion lasted long enough
            duration = (pred.timestamp - current_start_time) / 1000.0
            if duration >= min_duration:
                transitions.append((current_emotion, pred.emotion, pred.timestamp))

            current_emotion = pred.emotion
            current_start_time = pred.timestamp

    return transitions


def compute_emotion_stability(predictions: list[EmotionPrediction]) -> float:
    """Compute a stability score indicating how consistent emotions are.

    Higher scores indicate more stable emotions (fewer transitions).

    Args:
        predictions: List of emotion predictions

    Returns:
        Stability score between 0.0 (highly variable) and 1.0 (completely stable)
    """
    if len(predictions) < 2:
        return 1.0

    # Count transitions
    transitions = sum(
        1 for i in range(1, len(predictions))
        if predictions[i].emotion != predictions[i-1].emotion
    )

    # Normalize by maximum possible transitions
    max_transitions = len(predictions) - 1
    stability = 1.0 - (transitions / max_transitions)

    return stability


def compute_confidence_statistics(
    predictions: list[EmotionPrediction]
) -> dict[str, float]:
    """Compute statistics about confidence scores.

    Args:
        predictions: List of emotion predictions

    Returns:
        Dictionary with mean, std, min, max, median confidence values
    """
    if not predictions:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0
        }

    confidences = np.array([pred.confidence for pred in predictions])

    return {
        "mean": float(np.mean(confidences)),
        "std": float(np.std(confidences)),
        "min": float(np.min(confidences)),
        "max": float(np.max(confidences)),
        "median": float(np.median(confidences))
    }


def compute_au_statistics(
    predictions: list[EmotionPrediction]
) -> dict[ActionUnitType, dict[str, float]]:
    """Compute statistics for each action unit across predictions.

    Args:
        predictions: List of emotion predictions

    Returns:
        Dictionary mapping each AU to its statistics (mean, std, max intensity, presence rate)
    """
    if not predictions:
        return {}

    # Collect AU data
    au_data: defaultdict = defaultdict(list)
    au_presence: defaultdict = defaultdict(int)

    for pred in predictions:
        for au_type, au in pred.action_units.items():
            au_data[au_type].append(au.intensity)
            if au.present:
                au_presence[au_type] += 1

    # Compute statistics
    au_stats = {}
    total_frames = len(predictions)

    for au_type, intensities in au_data.items():
        intensities_array = np.array(intensities)
        au_stats[au_type] = {
            "mean": float(np.mean(intensities_array)),
            "std": float(np.std(intensities_array)),
            "max": float(np.max(intensities_array)),
            "presence_rate": au_presence[au_type] / total_frames
        }

    return au_stats


def find_peak_emotions(
    predictions: list[EmotionPrediction],
    emotion: EmotionType,
    top_n: int = 5
) -> list[EmotionPrediction]:
    """Find frames with the strongest expression of a specific emotion.

    Args:
        predictions: List of emotion predictions
        emotion: The emotion to find peaks for
        top_n: Number of top predictions to return

    Returns:
        List of top N predictions for the specified emotion, sorted by confidence
    """
    # Filter predictions for the target emotion
    emotion_predictions = [pred for pred in predictions if pred.emotion == emotion]

    # Sort by confidence descending
    sorted_predictions = sorted(
        emotion_predictions,
        key=lambda p: p.confidence,
        reverse=True
    )

    return sorted_predictions[:top_n]


def compute_emotion_timeline(
    predictions: list[EmotionPrediction],
    window_size: float = 1.0
) -> list[tuple[float, dict[EmotionType, float]]]:
    """Compute emotion distribution over time using a sliding window.

    Args:
        predictions: List of emotion predictions
        window_size: Size of the sliding window in seconds

    Returns:
        List of (timestamp, emotion_distribution) tuples
    """
    if not predictions:
        return []

    window_size_ms = window_size * 1000.0
    timeline = []

    # Sort predictions by timestamp
    sorted_preds = sorted(predictions, key=lambda p: p.timestamp)

    start_time = sorted_preds[0].timestamp
    end_time = sorted_preds[-1].timestamp

    # Slide window through time
    current_time = start_time

    while current_time <= end_time:
        # Get predictions in current window
        window_preds = [
            pred for pred in sorted_preds
            if current_time <= pred.timestamp < current_time + window_size_ms
        ]

        if window_preds:
            # Compute distribution for this window
            distribution = compute_emotion_distribution(window_preds, normalize=True)
            timeline.append((current_time, distribution))

        # Move window forward
        current_time += window_size_ms / 2.0  # 50% overlap

    return timeline
