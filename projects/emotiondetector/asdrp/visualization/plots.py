"""Statistical plotting functions for emotion analysis.

This module provides functions to create various statistical plots and charts
for analyzing emotion detection results using matplotlib and seaborn.
"""

from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import seaborn as sns
from matplotlib.figure import Figure

from asdrp.emotion.base import ActionUnitType, EmotionPrediction, EmotionType

# Set default style
sns.set_theme(style="whitegrid")


def plot_emotion_distribution(
    emotion_predictions: list[EmotionPrediction],
    plot_type: str = "bar",
    title: str = "Emotion Distribution",
    figsize: tuple[int, int] = (10, 6),
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """Plot distribution of emotions across all predictions.

    Args:
        emotion_predictions: List of EmotionPrediction objects
        plot_type: Type of plot ('bar' or 'pie')
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object
    """
    if not emotion_predictions:
        raise ValueError("emotion_predictions list is empty")

    # Count emotions
    emotion_counts = {}
    for pred in emotion_predictions:
        emotion = pred.emotion
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    # Sort by emotion type
    emotions = sorted(emotion_counts.keys(), key=lambda x: x.value)
    counts = [emotion_counts[e] for e in emotions]
    labels = [e.value.capitalize() for e in emotions]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    if plot_type == "bar":
        colors = ['#C8C8C8', '#FFFF00', '#FF0000', '#0000FF', '#FF8000', '#800080', '#008000']
        bars = ax.bar(labels, counts, color=colors[:len(labels)])
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlabel("Emotion", fontsize=12)

        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=10
            )

    elif plot_type == "pie":
        colors = ['#C8C8C8', '#FFFF00', '#FF0000', '#0000FF', '#FF8000', '#800080', '#008000']
        wedges, texts, autotexts = ax.pie(
            counts,
            labels=labels,
            autopct='%1.1f%%',
            colors=colors[:len(labels)],
            startangle=90
        )

        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}. Use 'bar' or 'pie'.")

    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_emotion_timeline(
    emotion_predictions: list[EmotionPrediction],
    title: str = "Emotion Timeline",
    figsize: tuple[int, int] = (14, 6),
    save_path: Optional[Union[str, Path]] = None,
    use_frame_numbers: bool = True
) -> Figure:
    """Plot emotion changes over time.

    Args:
        emotion_predictions: List of EmotionPrediction objects (should be in chronological order)
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        use_frame_numbers: Use frame numbers for x-axis, otherwise use index

    Returns:
        Matplotlib Figure object
    """
    if not emotion_predictions:
        raise ValueError("emotion_predictions list is empty")

    # Prepare data
    if use_frame_numbers:
        x_values = [pred.frame_number for pred in emotion_predictions]
        x_label = "Frame Number"
    else:
        x_values = list(range(len(emotion_predictions)))
        x_label = "Prediction Index"

    # Map emotions to numeric values for plotting
    emotion_to_num = {e: i for i, e in enumerate(EmotionType)}
    y_values = [emotion_to_num[pred.emotion] for pred in emotion_predictions]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create color map
    colors = ['#C8C8C8', '#FFFF00', '#FF0000', '#0000FF', '#FF8000', '#800080', '#008000']
    emotion_colors = [colors[emotion_to_num[pred.emotion]] for pred in emotion_predictions]

    # Plot as scatter with connecting lines
    ax.scatter(x_values, y_values, c=emotion_colors, s=50, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.plot(x_values, y_values, color='gray', alpha=0.3, linewidth=1)

    # Set y-axis labels
    ax.set_yticks(list(emotion_to_num.values()))
    ax.set_yticklabels([e.value.capitalize() for e in EmotionType])

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Emotion", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_confidence_over_time(
    emotion_predictions: list[EmotionPrediction],
    title: str = "Confidence Over Time",
    figsize: tuple[int, int] = (14, 6),
    save_path: Optional[Union[str, Path]] = None,
    use_frame_numbers: bool = True,
    show_moving_average: bool = True,
    window_size: int = 10
) -> Figure:
    """Plot confidence scores over time.

    Args:
        emotion_predictions: List of EmotionPrediction objects
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        use_frame_numbers: Use frame numbers for x-axis
        show_moving_average: Whether to show moving average line
        window_size: Window size for moving average

    Returns:
        Matplotlib Figure object
    """
    if not emotion_predictions:
        raise ValueError("emotion_predictions list is empty")

    # Prepare data
    if use_frame_numbers:
        x_values = [pred.frame_number for pred in emotion_predictions]
        x_label = "Frame Number"
    else:
        x_values = list(range(len(emotion_predictions)))
        x_label = "Prediction Index"

    confidences = [pred.confidence for pred in emotion_predictions]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot confidence values
    ax.plot(x_values, confidences, marker='o', markersize=3, alpha=0.6, label='Confidence')

    # Add moving average if requested
    if show_moving_average and len(confidences) >= window_size:
        moving_avg = np.convolve(
            confidences,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        ma_x = x_values[window_size - 1:]
        ax.plot(ma_x, moving_avg, color='red', linewidth=2, label=f'Moving Avg (window={window_size})')

    # Add horizontal line at 0.5 (threshold)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='Threshold (0.5)')

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Confidence", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_action_units(
    emotion_predictions: list[EmotionPrediction],
    au_types: Optional[list[ActionUnitType]] = None,
    title: str = "Action Unit Activation Over Time",
    figsize: tuple[int, int] = (14, 8),
    save_path: Optional[Union[str, Path]] = None,
    use_frame_numbers: bool = True
) -> Figure:
    """Plot action unit activations over time.

    Args:
        emotion_predictions: List of EmotionPrediction objects
        au_types: Optional list of specific AUs to plot. If None, plots all detected AUs.
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        use_frame_numbers: Use frame numbers for x-axis

    Returns:
        Matplotlib Figure object
    """
    if not emotion_predictions:
        raise ValueError("emotion_predictions list is empty")

    # Prepare data
    if use_frame_numbers:
        x_values = [pred.frame_number for pred in emotion_predictions]
        x_label = "Frame Number"
    else:
        x_values = list(range(len(emotion_predictions)))
        x_label = "Prediction Index"

    # Collect all AU types if not specified
    if au_types is None:
        all_aus = set()
        for pred in emotion_predictions:
            all_aus.update(pred.action_units.keys())
        au_types = sorted(list(all_aus), key=lambda x: x.value)

    if not au_types:
        raise ValueError("No action units found in predictions")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot each AU
    for au_type in au_types:
        intensities = []
        for pred in emotion_predictions:
            if au_type in pred.action_units:
                intensities.append(pred.action_units[au_type].intensity)
            else:
                intensities.append(0.0)

        ax.plot(x_values, intensities, marker='o', markersize=2, label=str(au_type), alpha=0.7)

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Intensity", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_emotion_transitions(
    emotion_predictions: list[EmotionPrediction],
    title: str = "Emotion Transition Matrix",
    figsize: tuple[int, int] = (10, 8),
    save_path: Optional[Union[str, Path]] = None,
    normalize: bool = True
) -> Figure:
    """Plot emotion transition matrix as a heatmap.

    Shows how frequently each emotion transitions to another emotion.

    Args:
        emotion_predictions: List of EmotionPrediction objects (should be in chronological order)
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        normalize: Whether to normalize counts to probabilities

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
        cmap='YlOrRd',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Probability' if normalize else 'Count'},
        ax=ax
    )

    ax.set_xlabel("Next Emotion", fontsize=12)
    ax.set_ylabel("Current Emotion", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_emotion_probabilities_over_time(
    emotion_predictions: list[EmotionPrediction],
    emotions_to_plot: Optional[list[EmotionType]] = None,
    title: str = "Emotion Probabilities Over Time",
    figsize: tuple[int, int] = (14, 8),
    save_path: Optional[Union[str, Path]] = None,
    use_frame_numbers: bool = True,
    stacked: bool = False
) -> Figure:
    """Plot emotion probabilities over time for multiple emotions.

    Args:
        emotion_predictions: List of EmotionPrediction objects
        emotions_to_plot: Optional list of emotions to plot. If None, plots all.
        title: Plot title
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure
        use_frame_numbers: Use frame numbers for x-axis
        stacked: Whether to create stacked area plot

    Returns:
        Matplotlib Figure object
    """
    if not emotion_predictions:
        raise ValueError("emotion_predictions list is empty")

    # Prepare data
    if use_frame_numbers:
        x_values = [pred.frame_number for pred in emotion_predictions]
        x_label = "Frame Number"
    else:
        x_values = list(range(len(emotion_predictions)))
        x_label = "Prediction Index"

    # Get emotions to plot
    if emotions_to_plot is None:
        emotions_to_plot = list(EmotionType)

    # Collect probabilities for each emotion
    emotion_probs = {emotion: [] for emotion in emotions_to_plot}
    for pred in emotion_predictions:
        for emotion in emotions_to_plot:
            prob = pred.probabilities.get(emotion, 0.0)
            emotion_probs[emotion].append(prob)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color scheme
    colors = {
        EmotionType.NEUTRAL: '#C8C8C8',
        EmotionType.HAPPY: '#FFFF00',
        EmotionType.SAD: '#FF0000',
        EmotionType.ANGRY: '#0000FF',
        EmotionType.SURPRISED: '#FF8000',
        EmotionType.FEARFUL: '#800080',
        EmotionType.DISGUSTED: '#008000'
    }

    if stacked:
        # Create stacked area plot
        prob_array = np.array([emotion_probs[e] for e in emotions_to_plot])
        ax.stackplot(
            x_values,
            *prob_array,
            labels=[e.value.capitalize() for e in emotions_to_plot],
            colors=[colors[e] for e in emotions_to_plot],
            alpha=0.7
        )
    else:
        # Create line plot
        for emotion in emotions_to_plot:
            ax.plot(
                x_values,
                emotion_probs[emotion],
                label=emotion.value.capitalize(),
                color=colors[emotion],
                linewidth=2,
                alpha=0.7
            )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig


def plot_emotion_summary(
    emotion_predictions: list[EmotionPrediction],
    title: str = "Emotion Analysis Summary",
    figsize: tuple[int, int] = (16, 10),
    save_path: Optional[Union[str, Path]] = None
) -> Figure:
    """Create a comprehensive summary plot with multiple subplots.

    Args:
        emotion_predictions: List of EmotionPrediction objects
        title: Main title for the figure
        figsize: Figure size as (width, height)
        save_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object with multiple subplots
    """
    if not emotion_predictions:
        raise ValueError("emotion_predictions list is empty")

    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # 1. Emotion distribution (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    emotion_counts = {}
    for pred in emotion_predictions:
        emotion = pred.emotion
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    emotions = sorted(emotion_counts.keys(), key=lambda x: x.value)
    counts = [emotion_counts[e] for e in emotions]
    labels = [e.value.capitalize() for e in emotions]
    colors = ['#C8C8C8', '#FFFF00', '#FF0000', '#0000FF', '#FF8000', '#800080', '#008000']

    ax1.pie(counts, labels=labels, autopct='%1.1f%%', colors=colors[:len(labels)], startangle=90)
    ax1.set_title("Emotion Distribution", fontweight='bold')

    # 2. Emotion timeline
    ax2 = fig.add_subplot(gs[0, 1])
    emotion_to_num = {e: i for i, e in enumerate(EmotionType)}
    x_values = list(range(len(emotion_predictions)))
    y_values = [emotion_to_num[pred.emotion] for pred in emotion_predictions]
    emotion_colors = [colors[emotion_to_num[pred.emotion]] for pred in emotion_predictions]

    ax2.scatter(x_values, y_values, c=emotion_colors, s=20, alpha=0.8)
    ax2.plot(x_values, y_values, color='gray', alpha=0.3, linewidth=1)
    ax2.set_yticks(list(emotion_to_num.values()))
    ax2.set_yticklabels([e.value.capitalize() for e in EmotionType])
    ax2.set_xlabel("Frame")
    ax2.set_title("Emotion Timeline", fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # 3. Confidence over time
    ax3 = fig.add_subplot(gs[1, :])
    confidences = [pred.confidence for pred in emotion_predictions]
    ax3.plot(x_values, confidences, marker='o', markersize=2, alpha=0.6)
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Frame")
    ax3.set_ylabel("Confidence")
    ax3.set_title("Confidence Over Time", fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # 4. Top emotions bar chart
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.bar(labels, counts, color=colors[:len(labels)])
    ax4.set_xlabel("Emotion")
    ax4.set_ylabel("Count")
    ax4.set_title("Emotion Frequency", fontweight='bold')

    # 5. Statistics text
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    # Calculate statistics
    avg_confidence = np.mean(confidences)
    min_confidence = np.min(confidences)
    max_confidence = np.max(confidences)
    most_common = max(emotion_counts.items(), key=lambda x: x[1])

    stats_text = f"""
    Statistics:

    Total Frames: {len(emotion_predictions)}

    Average Confidence: {avg_confidence:.2%}
    Min Confidence: {min_confidence:.2%}
    Max Confidence: {max_confidence:.2%}

    Most Common Emotion: {most_common[0].value.capitalize()}
    ({most_common[1]} frames, {most_common[1]/len(emotion_predictions):.1%})
    """

    ax5.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', family='monospace')

    fig.suptitle(title, fontsize=16, fontweight='bold')

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig
