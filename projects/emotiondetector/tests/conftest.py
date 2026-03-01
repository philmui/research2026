"""Shared pytest fixtures for emotion detector tests.

This module provides reusable test fixtures including sample images, mock objects,
and test data generators for all test modules.
"""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, Mock

import cv2
import numpy as np
import pytest

from asdrp.emotion.base import (
    ActionUnit,
    ActionUnitType,
    EmotionPrediction,
    EmotionType,
)
from asdrp.face.base import BoundingBox, FaceLandmarks


@pytest.fixture
def sample_face_image() -> np.ndarray:
    """Create a synthetic face image for testing.

    Returns:
        A 480x640x3 BGR image with a simple face-like pattern.
    """
    # Create a blank image
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a simple face pattern
    # Face oval (circle)
    cv2.circle(image, (320, 240), 150, (200, 200, 200), -1)

    # Eyes (smaller circles)
    cv2.circle(image, (270, 200), 20, (50, 50, 50), -1)
    cv2.circle(image, (370, 200), 20, (50, 50, 50), -1)

    # Nose (triangle)
    pts = np.array([[320, 220], [310, 260], [330, 260]], np.int32)
    cv2.fillPoly(image, [pts], (150, 150, 150))

    # Mouth (arc)
    cv2.ellipse(image, (320, 290), (40, 20), 0, 0, 180, (100, 100, 100), 2)

    return image


@pytest.fixture
def sample_rgb_image() -> np.ndarray:
    """Create a simple RGB test image.

    Returns:
        A 240x320x3 RGB image.
    """
    image = np.random.randint(0, 256, (240, 320, 3), dtype=np.uint8)
    return image


@pytest.fixture
def sample_face_landmarks() -> FaceLandmarks:
    """Create sample face landmarks for testing.

    Returns:
        FaceLandmarks object with 478 normalized landmarks.
    """
    # Create 478 random landmarks (MediaPipe standard)
    num_landmarks = 478
    landmarks = np.random.rand(num_landmarks, 3).astype(np.float32)

    # Ensure landmarks are normalized (0 to 1)
    landmarks[:, :2] = np.clip(landmarks[:, :2], 0.0, 1.0)
    landmarks[:, 2] = landmarks[:, 2] * 0.1  # Small z values

    visibility = np.random.rand(num_landmarks).astype(np.float32)
    bounding_box = BoundingBox(x_min=0.2, y_min=0.3, width=0.5, height=0.6)

    return FaceLandmarks(
        landmarks=landmarks,
        visibility=visibility,
        bounding_box=bounding_box,
        timestamp=1000.0,
        frame_number=10,
        face_id=0,
    )


@pytest.fixture
def sample_bounding_box() -> BoundingBox:
    """Create a sample bounding box.

    Returns:
        BoundingBox with typical normalized coordinates.
    """
    return BoundingBox(x_min=0.25, y_min=0.2, width=0.5, height=0.6)


@pytest.fixture
def mock_face_detector() -> Mock:
    """Create a mock face detector for testing.

    Returns:
        Mock MediaPipeFaceDetector with detect and detect_batch methods.
    """
    detector = Mock()

    # Mock detect method
    def mock_detect(image: np.ndarray, timestamp_ms: float = 0.0) -> list:
        landmarks = np.random.rand(478, 3).astype(np.float32)
        return [
            FaceLandmarks(
                landmarks=landmarks,
                bounding_box=BoundingBox(0.2, 0.2, 0.6, 0.6),
                timestamp=timestamp_ms,
            )
        ]

    detector.detect = Mock(side_effect=mock_detect)
    detector.detect_batch = Mock(return_value=[[]])
    detector.close = Mock()

    return detector


@pytest.fixture
def sample_emotion_predictions() -> list[EmotionPrediction]:
    """Create sample emotion predictions for testing.

    Returns:
        List of EmotionPrediction objects with different emotions.
    """
    predictions = []

    emotions_data = [
        (EmotionType.HAPPY, 0.85),
        (EmotionType.SAD, 0.72),
        (EmotionType.ANGRY, 0.68),
        (EmotionType.NEUTRAL, 0.90),
    ]

    for idx, (emotion, confidence) in enumerate(emotions_data):
        # Create probability distribution
        probabilities = {e: 0.1 for e in EmotionType}
        probabilities[emotion] = confidence

        # Normalize to sum to 1.0
        total = sum(probabilities.values())
        probabilities = {e: p / total for e, p in probabilities.items()}

        prediction = EmotionPrediction(
            emotion=emotion,
            confidence=confidence,
            probabilities=probabilities,
            timestamp=float(idx * 1000),
            frame_number=idx,
        )
        predictions.append(prediction)

    return predictions


@pytest.fixture
def sample_action_units() -> dict[ActionUnitType, ActionUnit]:
    """Create sample action units for testing.

    Returns:
        Dictionary mapping ActionUnitType to ActionUnit objects.
    """
    action_units = {
        ActionUnitType.AU6: ActionUnit(
            au_type=ActionUnitType.AU6,
            intensity=0.8,
            present=True,
            confidence=0.9,
        ),
        ActionUnitType.AU12: ActionUnit(
            au_type=ActionUnitType.AU12,
            intensity=0.7,
            present=True,
            confidence=0.85,
        ),
        ActionUnitType.AU4: ActionUnit(
            au_type=ActionUnitType.AU4,
            intensity=0.2,
            present=False,
            confidence=0.6,
        ),
    }
    return action_units


@pytest.fixture
def temp_video_file() -> Generator[Path, None, None]:
    """Create a temporary video file for testing.

    Yields:
        Path to a temporary video file with a few frames.
    """
    # Create temporary file
    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    try:
        # Create a simple video with a few frames
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(temp_path), fourcc, 30.0, (640, 480))

        # Write 10 frames
        for i in range(10):
            frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            # Draw frame number
            cv2.putText(
                frame,
                f"Frame {i}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )
            writer.write(frame)

        writer.release()

        yield temp_path
    finally:
        # Cleanup
        if temp_path.exists():
            temp_path.unlink()


@pytest.fixture
def temp_output_dir() -> Generator[Path, None, None]:
    """Create a temporary output directory for testing.

    Yields:
        Path to a temporary directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_mediapipe_detector() -> Mock:
    """Create a mock MediaPipe detector for testing.

    Returns:
        Mock with detect and detect_for_video methods.
    """
    detector = Mock()

    # Mock detection result
    result = Mock()
    result.face_landmarks = []

    detector.detect = Mock(return_value=result)
    detector.detect_for_video = Mock(return_value=result)
    detector.close = Mock()

    return detector


@pytest.fixture
def mock_emotion_analyzer() -> Mock:
    """Create a mock emotion analyzer for testing.

    Returns:
        Mock with analyze and analyze_batch methods.
    """
    analyzer = Mock()

    # Mock analyze method
    def mock_analyze(landmarks: FaceLandmarks) -> EmotionPrediction:
        probabilities = {e: 0.1 for e in EmotionType}
        probabilities[EmotionType.HAPPY] = 0.4

        # Normalize
        total = sum(probabilities.values())
        probabilities = {e: p / total for e, p in probabilities.items()}

        return EmotionPrediction(
            emotion=EmotionType.HAPPY,
            confidence=0.8,
            probabilities=probabilities,
            timestamp=landmarks.timestamp,
            frame_number=landmarks.frame_number,
        )

    analyzer.analyze = Mock(side_effect=mock_analyze)
    analyzer.analyze_batch = Mock(return_value=[])

    return analyzer


@pytest.fixture
def sample_features() -> dict[str, float]:
    """Create sample feature dictionary for testing.

    Returns:
        Dictionary of feature names to values.
    """
    return {
        "left_eye_openness": 0.8,
        "right_eye_openness": 0.75,
        "left_eyebrow_height": 0.3,
        "right_eyebrow_height": 0.32,
        "mouth_openness": 0.4,
        "mouth_width": 0.6,
        "lip_corner_pull": 0.7,
        "eyebrow_furrow": 0.2,
        "nose_wrinkle": 0.1,
    }


@pytest.fixture
def sample_emotion_probabilities() -> dict[EmotionType, float]:
    """Create sample emotion probability distribution.

    Returns:
        Dictionary mapping EmotionType to probability values.
    """
    probabilities = {
        EmotionType.NEUTRAL: 0.1,
        EmotionType.HAPPY: 0.6,
        EmotionType.SAD: 0.05,
        EmotionType.ANGRY: 0.05,
        EmotionType.SURPRISED: 0.1,
        EmotionType.FEARFUL: 0.05,
        EmotionType.DISGUSTED: 0.05,
    }

    # Ensure sum is exactly 1.0
    total = sum(probabilities.values())
    return {e: p / total for e, p in probabilities.items()}
