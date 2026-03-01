"""Export utilities for saving analysis results.

This module provides functions for exporting facial landmarks, emotion predictions,
and other analysis results to various file formats including JSON and CSV.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import numpy.typing as npt


def export_to_json(
    data: Dict[str, Any],
    output_path: str | Path,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Export data to JSON file.

    Args:
        data: Dictionary containing data to export.
        output_path: Path to the output JSON file.
        indent: Number of spaces for JSON indentation. Default is 2.
        ensure_ascii: Whether to escape non-ASCII characters. Default is False.

    Raises:
        IOError: If file cannot be written.
        TypeError: If data contains non-serializable objects.

    Example:
        >>> data = {"frame": 0, "emotions": {"happy": 0.8, "sad": 0.2}}
        >>> export_to_json(data, "results.json")
    """
    output_path = Path(output_path)

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Custom JSON encoder for numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj: Any) -> Any:
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, Path):
                return str(obj)
            return super().default(obj)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii, cls=NumpyEncoder)
    except Exception as e:
        raise IOError(f"Failed to write JSON file to {output_path}: {e}") from e


def export_to_csv(
    data: List[Dict[str, Any]],
    output_path: str | Path,
    fieldnames: Optional[List[str]] = None,
) -> None:
    """Export data to CSV file.

    Args:
        data: List of dictionaries, where each dictionary represents a row.
        output_path: Path to the output CSV file.
        fieldnames: Optional list of field names for CSV header. If None,
                   uses keys from the first data dictionary.

    Raises:
        IOError: If file cannot be written.
        ValueError: If data is empty or fieldnames is invalid.

    Example:
        >>> data = [
        ...     {"frame": 0, "emotion": "happy", "confidence": 0.8},
        ...     {"frame": 1, "emotion": "sad", "confidence": 0.6}
        ... ]
        >>> export_to_csv(data, "emotions.csv")
    """
    if not data:
        raise ValueError("Data list must not be empty")

    output_path = Path(output_path)

    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine field names
    if fieldnames is None:
        fieldnames = list(data[0].keys())

    try:
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in data:
                # Convert numpy types to native Python types
                converted_row = {}
                for key, value in row.items():
                    if isinstance(value, (np.integer, np.floating)):
                        converted_row[key] = float(value) if isinstance(value, np.floating) else int(value)
                    elif isinstance(value, np.ndarray):
                        converted_row[key] = value.tolist()
                    elif isinstance(value, Path):
                        converted_row[key] = str(value)
                    else:
                        converted_row[key] = value
                writer.writerow(converted_row)
    except Exception as e:
        raise IOError(f"Failed to write CSV file to {output_path}: {e}") from e


def export_landmarks_to_json(
    landmarks_list: List[npt.NDArray[np.float32]],
    output_path: str | Path,
    frame_numbers: Optional[List[int]] = None,
    timestamps: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Export facial landmarks to JSON file.

    Args:
        landmarks_list: List of landmark arrays, each of shape (N, 3).
        output_path: Path to the output JSON file.
        frame_numbers: Optional list of frame numbers corresponding to landmarks.
        timestamps: Optional list of timestamps corresponding to landmarks.
        metadata: Optional metadata dictionary to include in the export.

    Raises:
        ValueError: If list lengths don't match.
        IOError: If file cannot be written.

    Example:
        >>> landmarks = [np.random.rand(478, 3).astype(np.float32) for _ in range(10)]
        >>> export_landmarks_to_json(landmarks, "landmarks.json", frame_numbers=list(range(10)))
    """
    if frame_numbers is not None and len(frame_numbers) != len(landmarks_list):
        raise ValueError(
            f"frame_numbers length ({len(frame_numbers)}) must match "
            f"landmarks_list length ({len(landmarks_list)})"
        )

    if timestamps is not None and len(timestamps) != len(landmarks_list):
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match "
            f"landmarks_list length ({len(landmarks_list)})"
        )

    # Build data structure
    data: Dict[str, Any] = {
        "num_frames": len(landmarks_list),
        "num_landmarks": landmarks_list[0].shape[0] if landmarks_list else 0,
        "frames": [],
    }

    if metadata is not None:
        data["metadata"] = metadata

    for i, landmarks in enumerate(landmarks_list):
        frame_data: Dict[str, Any] = {
            "frame_number": frame_numbers[i] if frame_numbers else i,
            "landmarks": landmarks.tolist(),
        }

        if timestamps is not None:
            frame_data["timestamp"] = timestamps[i]

        data["frames"].append(frame_data)

    export_to_json(data, output_path)


def export_landmarks_to_csv(
    landmarks_list: List[npt.NDArray[np.float32]],
    output_path: str | Path,
    frame_numbers: Optional[List[int]] = None,
    timestamps: Optional[List[float]] = None,
    landmark_indices: Optional[List[int]] = None,
) -> None:
    """Export facial landmarks to CSV file.

    Each row represents one frame, with columns for frame metadata and landmark
    coordinates (x, y, z for each landmark).

    Args:
        landmarks_list: List of landmark arrays, each of shape (N, 3).
        output_path: Path to the output CSV file.
        frame_numbers: Optional list of frame numbers corresponding to landmarks.
        timestamps: Optional list of timestamps corresponding to landmarks.
        landmark_indices: Optional list of specific landmark indices to export.
                         If None, exports all landmarks.

    Raises:
        ValueError: If list lengths don't match.
        IOError: If file cannot be written.

    Example:
        >>> landmarks = [np.random.rand(478, 3).astype(np.float32) for _ in range(10)]
        >>> export_landmarks_to_csv(landmarks, "landmarks.csv", landmark_indices=[0, 1, 2])
    """
    if not landmarks_list:
        raise ValueError("landmarks_list must not be empty")

    if frame_numbers is not None and len(frame_numbers) != len(landmarks_list):
        raise ValueError(
            f"frame_numbers length ({len(frame_numbers)}) must match "
            f"landmarks_list length ({len(landmarks_list)})"
        )

    if timestamps is not None and len(timestamps) != len(landmarks_list):
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match "
            f"landmarks_list length ({len(landmarks_list)})"
        )

    # Determine which landmarks to export
    num_landmarks = landmarks_list[0].shape[0]
    if landmark_indices is None:
        landmark_indices = list(range(num_landmarks))

    # Build field names
    fieldnames = ["frame_number"]
    if timestamps is not None:
        fieldnames.append("timestamp")

    for idx in landmark_indices:
        fieldnames.extend([f"landmark_{idx}_x", f"landmark_{idx}_y", f"landmark_{idx}_z"])

    # Build data rows
    data_rows = []
    for i, landmarks in enumerate(landmarks_list):
        row: Dict[str, Any] = {
            "frame_number": frame_numbers[i] if frame_numbers else i,
        }

        if timestamps is not None:
            row["timestamp"] = timestamps[i]

        for idx in landmark_indices:
            row[f"landmark_{idx}_x"] = float(landmarks[idx, 0])
            row[f"landmark_{idx}_y"] = float(landmarks[idx, 1])
            row[f"landmark_{idx}_z"] = float(landmarks[idx, 2])

        data_rows.append(row)

    export_to_csv(data_rows, output_path, fieldnames=fieldnames)


def export_emotions_to_json(
    emotions_list: List[Dict[str, float]],
    output_path: str | Path,
    frame_numbers: Optional[List[int]] = None,
    timestamps: Optional[List[float]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Export emotion predictions to JSON file.

    Args:
        emotions_list: List of emotion dictionaries, each mapping emotion names to scores.
        output_path: Path to the output JSON file.
        frame_numbers: Optional list of frame numbers corresponding to emotions.
        timestamps: Optional list of timestamps corresponding to emotions.
        metadata: Optional metadata dictionary to include in the export.

    Raises:
        ValueError: If list lengths don't match.
        IOError: If file cannot be written.

    Example:
        >>> emotions = [
        ...     {"happy": 0.8, "sad": 0.1, "angry": 0.1},
        ...     {"happy": 0.2, "sad": 0.7, "angry": 0.1}
        ... ]
        >>> export_emotions_to_json(emotions, "emotions.json", frame_numbers=[0, 1])
    """
    if frame_numbers is not None and len(frame_numbers) != len(emotions_list):
        raise ValueError(
            f"frame_numbers length ({len(frame_numbers)}) must match "
            f"emotions_list length ({len(emotions_list)})"
        )

    if timestamps is not None and len(timestamps) != len(emotions_list):
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match "
            f"emotions_list length ({len(emotions_list)})"
        )

    # Build data structure
    data: Dict[str, Any] = {
        "num_frames": len(emotions_list),
        "emotion_classes": list(emotions_list[0].keys()) if emotions_list else [],
        "frames": [],
    }

    if metadata is not None:
        data["metadata"] = metadata

    for i, emotions in enumerate(emotions_list):
        frame_data: Dict[str, Any] = {
            "frame_number": frame_numbers[i] if frame_numbers else i,
            "emotions": emotions,
        }

        # Add predicted emotion (highest score)
        if emotions:
            predicted_emotion = max(emotions.items(), key=lambda x: x[1])
            frame_data["predicted_emotion"] = predicted_emotion[0]
            frame_data["confidence"] = predicted_emotion[1]

        if timestamps is not None:
            frame_data["timestamp"] = timestamps[i]

        data["frames"].append(frame_data)

    export_to_json(data, output_path)


def export_emotions_to_csv(
    emotions_list: List[Dict[str, float]],
    output_path: str | Path,
    frame_numbers: Optional[List[int]] = None,
    timestamps: Optional[List[float]] = None,
) -> None:
    """Export emotion predictions to CSV file.

    Each row represents one frame with columns for frame metadata, predicted
    emotion, confidence, and individual emotion scores.

    Args:
        emotions_list: List of emotion dictionaries, each mapping emotion names to scores.
        output_path: Path to the output CSV file.
        frame_numbers: Optional list of frame numbers corresponding to emotions.
        timestamps: Optional list of timestamps corresponding to emotions.

    Raises:
        ValueError: If list lengths don't match or emotions_list is empty.
        IOError: If file cannot be written.

    Example:
        >>> emotions = [
        ...     {"happy": 0.8, "sad": 0.1, "angry": 0.1},
        ...     {"happy": 0.2, "sad": 0.7, "angry": 0.1}
        ... ]
        >>> export_emotions_to_csv(emotions, "emotions.csv", frame_numbers=[0, 1])
    """
    if not emotions_list:
        raise ValueError("emotions_list must not be empty")

    if frame_numbers is not None and len(frame_numbers) != len(emotions_list):
        raise ValueError(
            f"frame_numbers length ({len(frame_numbers)}) must match "
            f"emotions_list length ({len(emotions_list)})"
        )

    if timestamps is not None and len(timestamps) != len(emotions_list):
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match "
            f"emotions_list length ({len(emotions_list)})"
        )

    # Get emotion classes from first frame
    emotion_classes = list(emotions_list[0].keys())

    # Build field names
    fieldnames = ["frame_number"]
    if timestamps is not None:
        fieldnames.append("timestamp")
    fieldnames.extend(["predicted_emotion", "confidence"])
    fieldnames.extend(emotion_classes)

    # Build data rows
    data_rows = []
    for i, emotions in enumerate(emotions_list):
        row: Dict[str, Any] = {
            "frame_number": frame_numbers[i] if frame_numbers else i,
        }

        if timestamps is not None:
            row["timestamp"] = timestamps[i]

        # Add predicted emotion (highest score)
        if emotions:
            predicted_emotion = max(emotions.items(), key=lambda x: x[1])
            row["predicted_emotion"] = predicted_emotion[0]
            row["confidence"] = float(predicted_emotion[1])

        # Add individual emotion scores
        for emotion in emotion_classes:
            row[emotion] = float(emotions.get(emotion, 0.0))

        data_rows.append(row)

    export_to_csv(data_rows, output_path, fieldnames=fieldnames)


def export_analysis_summary(
    landmarks_list: List[npt.NDArray[np.float32]],
    emotions_list: List[Dict[str, float]],
    output_path: str | Path,
    frame_numbers: Optional[List[int]] = None,
    timestamps: Optional[List[float]] = None,
    video_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Export comprehensive analysis summary to JSON file.

    Combines landmarks, emotions, and metadata into a single comprehensive export.

    Args:
        landmarks_list: List of landmark arrays, each of shape (N, 3).
        emotions_list: List of emotion dictionaries.
        output_path: Path to the output JSON file.
        frame_numbers: Optional list of frame numbers.
        timestamps: Optional list of timestamps.
        video_metadata: Optional video metadata to include.

    Raises:
        ValueError: If list lengths don't match.
        IOError: If file cannot be written.

    Example:
        >>> landmarks = [np.random.rand(478, 3).astype(np.float32) for _ in range(10)]
        >>> emotions = [{"happy": 0.8, "sad": 0.2} for _ in range(10)]
        >>> export_analysis_summary(landmarks, emotions, "summary.json")
    """
    if len(landmarks_list) != len(emotions_list):
        raise ValueError(
            f"landmarks_list length ({len(landmarks_list)}) must match "
            f"emotions_list length ({len(emotions_list)})"
        )

    if frame_numbers is not None and len(frame_numbers) != len(landmarks_list):
        raise ValueError(
            f"frame_numbers length ({len(frame_numbers)}) must match "
            f"data length ({len(landmarks_list)})"
        )

    if timestamps is not None and len(timestamps) != len(landmarks_list):
        raise ValueError(
            f"timestamps length ({len(timestamps)}) must match "
            f"data length ({len(landmarks_list)})"
        )

    # Build comprehensive data structure
    data: Dict[str, Any] = {
        "num_frames": len(landmarks_list),
        "num_landmarks": landmarks_list[0].shape[0] if landmarks_list else 0,
        "emotion_classes": list(emotions_list[0].keys()) if emotions_list else [],
        "frames": [],
    }

    if video_metadata is not None:
        data["video_metadata"] = video_metadata

    # Compute emotion statistics
    if emotions_list:
        emotion_classes = list(emotions_list[0].keys())
        emotion_stats = {}
        for emotion in emotion_classes:
            scores = [frame_emotions[emotion] for frame_emotions in emotions_list]
            emotion_stats[emotion] = {
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
            }
        data["emotion_statistics"] = emotion_stats

    for i in range(len(landmarks_list)):
        frame_data: Dict[str, Any] = {
            "frame_number": frame_numbers[i] if frame_numbers else i,
            "landmarks": landmarks_list[i].tolist(),
            "emotions": emotions_list[i],
        }

        # Add predicted emotion
        if emotions_list[i]:
            predicted_emotion = max(emotions_list[i].items(), key=lambda x: x[1])
            frame_data["predicted_emotion"] = predicted_emotion[0]
            frame_data["confidence"] = predicted_emotion[1]

        if timestamps is not None:
            frame_data["timestamp"] = timestamps[i]

        data["frames"].append(frame_data)

    export_to_json(data, output_path)
