"""Heatmap visualization for temporal emotion analysis.

This module provides the EmotionHeatmap class for creating temporal heatmaps
showing emotion intensity over time and transition matrices.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.figure import Figure

from asdrp.emotion.base import EmotionPrediction, EmotionType


class EmotionHeatmap:
    """Create heatmaps for temporal emotion analysis.

    This class provides methods to visualize emotion intensities and transitions
    over time using heatmaps and other matrix visualizations.
    """

    def __init__(self, cmap: str = "YlOrRd"):
        """Initialize the emotion heatmap generator.

        Args:
            cmap: Matplotlib colormap name to use for heatmaps
        """
        self.cmap = cmap
        sns.set_theme(style="white")

    def create_temporal_heatmap(
        self,
        emotion_predictions: list[EmotionPrediction],
        window_size: int = 10,
        title: str = "Emotion Intensity Over Time",
        figsize: tuple[int, int] = (14, 6),
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """Create a heatmap showing emotion intensities over time.

        The heatmap shows emotion probabilities with time windows on the x-axis
        and emotion types on the y-axis.

        Args:
            emotion_predictions: List of EmotionPrediction objects in chronological order
            window_size: Number of frames to aggregate into each time window
            title: Plot title
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        if not emotion_predictions:
            raise ValueError("emotion_predictions list is empty")

        # Get all emotion types
        emotions = list(EmotionType)
        n_emotions = len(emotions)

        # Calculate number of windows
        n_frames = len(emotion_predictions)
        n_windows = (n_frames + window_size - 1) // window_size

        # Initialize data matrix (emotions x time_windows)
        data = np.zeros((n_emotions, n_windows))

        # Aggregate probabilities into windows
        for window_idx in range(n_windows):
            start_idx = window_idx * window_size
            end_idx = min(start_idx + window_size, n_frames)

            # Average probabilities in this window
            for emotion_idx, emotion in enumerate(emotions):
                probs = [
                    pred.probabilities.get(emotion, 0.0)
                    for pred in emotion_predictions[start_idx:end_idx]
                ]
                data[emotion_idx, window_idx] = np.mean(probs)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            data,
            cmap=self.cmap,
            cbar_kws={'label': 'Probability'},
            yticklabels=[e.value.capitalize() for e in emotions],
            xticklabels=[f"{i*window_size}-{min((i+1)*window_size, n_frames)}" for i in range(n_windows)],
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_xlabel("Frame Range", fontsize=12)
        ax.set_ylabel("Emotion", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Rotate x-axis labels if there are many windows
        if n_windows > 10:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_transition_heatmap(
        self,
        emotion_predictions: list[EmotionPrediction],
        normalize: bool = True,
        title: str = "Emotion Transition Matrix",
        figsize: tuple[int, int] = (10, 8),
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """Create a heatmap showing emotion transitions.

        Args:
            emotion_predictions: List of EmotionPrediction objects in chronological order
            normalize: Whether to normalize counts to probabilities
            title: Plot title
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        if len(emotion_predictions) < 2:
            raise ValueError("Need at least 2 predictions to compute transitions")

        # Initialize transition matrix
        emotions = list(EmotionType)
        n_emotions = len(emotions)
        transition_matrix = np.zeros((n_emotions, n_emotions))

        # Map emotions to indices
        emotion_to_idx = {e: i for i, e in enumerate(emotions)}

        # Count transitions
        for i in range(len(emotion_predictions) - 1):
            current_emotion = emotion_predictions[i].emotion
            next_emotion = emotion_predictions[i + 1].emotion

            current_idx = emotion_to_idx[current_emotion]
            next_idx = emotion_to_idx[next_emotion]

            transition_matrix[current_idx, next_idx] += 1

        # Normalize if requested
        if normalize:
            row_sums = transition_matrix.sum(axis=1, keepdims=True)
            # Avoid division by zero
            row_sums[row_sums == 0] = 1
            transition_matrix = transition_matrix / row_sums

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        labels = [e.value.capitalize() for e in emotions]
        sns.heatmap(
            transition_matrix,
            annot=True,
            fmt='.2f' if normalize else '.0f',
            cmap=self.cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Probability' if normalize else 'Count'},
            ax=ax,
            vmin=0,
            vmax=1 if normalize else None
        )

        ax.set_xlabel("Next Emotion", fontsize=12)
        ax.set_ylabel("Current Emotion", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_correlation_heatmap(
        self,
        emotion_predictions: list[EmotionPrediction],
        title: str = "Emotion Correlation Matrix",
        figsize: tuple[int, int] = (10, 8),
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """Create a heatmap showing correlations between emotion probabilities.

        Args:
            emotion_predictions: List of EmotionPrediction objects
            title: Plot title
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        if not emotion_predictions:
            raise ValueError("emotion_predictions list is empty")

        # Get all emotion types
        emotions = list(EmotionType)
        n_emotions = len(emotions)
        n_frames = len(emotion_predictions)

        # Create data matrix (frames x emotions)
        data = np.zeros((n_frames, n_emotions))

        for frame_idx, pred in enumerate(emotion_predictions):
            for emotion_idx, emotion in enumerate(emotions):
                data[frame_idx, emotion_idx] = pred.probabilities.get(emotion, 0.0)

        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(data.T)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        labels = [e.value.capitalize() for e in emotions]
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Correlation'},
            ax=ax,
            vmin=-1,
            vmax=1,
            center=0
        )

        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_intensity_matrix(
        self,
        emotion_predictions_list: list[list[EmotionPrediction]],
        session_labels: Optional[list[str]] = None,
        title: str = "Emotion Intensity Across Sessions",
        figsize: tuple[int, int] = (12, 8),
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """Create a heatmap comparing emotion intensities across multiple sessions.

        Args:
            emotion_predictions_list: List of lists, where each inner list contains
                                     EmotionPrediction objects for one session
            session_labels: Optional labels for each session
            title: Plot title
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        if not emotion_predictions_list:
            raise ValueError("emotion_predictions_list is empty")

        # Get all emotion types
        emotions = list(EmotionType)
        n_emotions = len(emotions)
        n_sessions = len(emotion_predictions_list)

        # Initialize data matrix (sessions x emotions)
        data = np.zeros((n_sessions, n_emotions))

        # Calculate average probability for each emotion in each session
        for session_idx, predictions in enumerate(emotion_predictions_list):
            if not predictions:
                continue

            for emotion_idx, emotion in enumerate(emotions):
                probs = [pred.probabilities.get(emotion, 0.0) for pred in predictions]
                data[session_idx, emotion_idx] = np.mean(probs)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Generate session labels if not provided
        if session_labels is None:
            session_labels = [f"Session {i+1}" for i in range(n_sessions)]

        # Create heatmap
        emotion_labels = [e.value.capitalize() for e in emotions]
        sns.heatmap(
            data,
            annot=True,
            fmt='.2f',
            cmap=self.cmap,
            xticklabels=emotion_labels,
            yticklabels=session_labels,
            cbar_kws={'label': 'Average Probability'},
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_xlabel("Emotion", fontsize=12)
        ax.set_ylabel("Session", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_emotion_flow_diagram(
        self,
        emotion_predictions: list[EmotionPrediction],
        min_transition_count: int = 2,
        title: str = "Emotion Flow Diagram",
        figsize: tuple[int, int] = (12, 10),
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """Create a flow diagram showing significant emotion transitions.

        Args:
            emotion_predictions: List of EmotionPrediction objects in chronological order
            min_transition_count: Minimum number of transitions to display
            title: Plot title
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        if len(emotion_predictions) < 2:
            raise ValueError("Need at least 2 predictions to compute transitions")

        # Initialize transition matrix
        emotions = list(EmotionType)
        n_emotions = len(emotions)
        transition_matrix = np.zeros((n_emotions, n_emotions))

        # Map emotions to indices
        emotion_to_idx = {e: i for i, e in enumerate(emotions)}

        # Count transitions
        for i in range(len(emotion_predictions) - 1):
            current_emotion = emotion_predictions[i].emotion
            next_emotion = emotion_predictions[i + 1].emotion

            current_idx = emotion_to_idx[current_emotion]
            next_idx = emotion_to_idx[next_emotion]

            transition_matrix[current_idx, next_idx] += 1

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Filter transitions below threshold
        filtered_matrix = transition_matrix.copy()
        filtered_matrix[filtered_matrix < min_transition_count] = 0

        # Create heatmap with filtered data
        labels = [e.value.capitalize() for e in emotions]
        sns.heatmap(
            filtered_matrix,
            annot=True,
            fmt='.0f',
            cmap=self.cmap,
            xticklabels=labels,
            yticklabels=labels,
            cbar_kws={'label': 'Transition Count'},
            ax=ax,
            vmin=0
        )

        ax.set_xlabel("Next Emotion", fontsize=12)
        ax.set_ylabel("Current Emotion", fontsize=12)
        ax.set_title(f"{title} (min transitions: {min_transition_count})", fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig

    def create_sliding_window_heatmap(
        self,
        emotion_predictions: list[EmotionPrediction],
        window_size: int = 30,
        stride: int = 10,
        title: str = "Sliding Window Emotion Analysis",
        figsize: tuple[int, int] = (14, 6),
        save_path: Optional[Union[str, Path]] = None
    ) -> Figure:
        """Create a heatmap using sliding window analysis.

        Args:
            emotion_predictions: List of EmotionPrediction objects in chronological order
            window_size: Size of the sliding window in frames
            stride: Number of frames to move the window each step
            title: Plot title
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure

        Returns:
            Matplotlib Figure object
        """
        if not emotion_predictions:
            raise ValueError("emotion_predictions list is empty")

        if len(emotion_predictions) < window_size:
            raise ValueError(f"Need at least {window_size} predictions for window analysis")

        # Get all emotion types
        emotions = list(EmotionType)
        n_emotions = len(emotions)

        # Calculate number of windows
        n_frames = len(emotion_predictions)
        n_windows = ((n_frames - window_size) // stride) + 1

        # Initialize data matrix (emotions x windows)
        data = np.zeros((n_emotions, n_windows))

        # Calculate emotion probabilities for each window
        for window_idx in range(n_windows):
            start_idx = window_idx * stride
            end_idx = start_idx + window_size

            # Average probabilities in this window
            for emotion_idx, emotion in enumerate(emotions):
                probs = [
                    pred.probabilities.get(emotion, 0.0)
                    for pred in emotion_predictions[start_idx:end_idx]
                ]
                data[emotion_idx, window_idx] = np.mean(probs)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create heatmap
        sns.heatmap(
            data,
            cmap=self.cmap,
            cbar_kws={'label': 'Average Probability'},
            yticklabels=[e.value.capitalize() for e in emotions],
            xticklabels=[f"{i*stride}" for i in range(n_windows)],
            ax=ax,
            vmin=0,
            vmax=1
        )

        ax.set_xlabel(f"Window Start Frame (window size: {window_size}, stride: {stride})", fontsize=12)
        ax.set_ylabel("Emotion", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')

        # Rotate x-axis labels if there are many windows
        if n_windows > 15:
            plt.xticks(rotation=45, ha='right')

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')

        return fig
