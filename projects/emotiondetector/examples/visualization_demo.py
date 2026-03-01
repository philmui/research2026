"""Demonstration of the visualization module capabilities.

This script shows how to use the various visualization components to display
emotion detection results on video frames and create statistical plots.
"""

import numpy as np
import cv2
from pathlib import Path

from asdrp.face.base import FaceLandmarks, BoundingBox
from asdrp.emotion.base import EmotionPrediction, EmotionType, ActionUnit, ActionUnitType
from asdrp.visualization import (
    FaceOverlay,
    OverlayStyle,
    EmotionDisplay,
    DisplayStyle,
    EmotionHeatmap,
    plot_emotion_distribution,
    plot_emotion_timeline,
    plot_confidence_over_time,
    plot_emotion_summary,
)


def create_dummy_landmarks(image_shape: tuple[int, int]) -> FaceLandmarks:
    """Create dummy face landmarks for demonstration.

    Args:
        image_shape: (height, width) of the image

    Returns:
        FaceLandmarks object with synthetic data
    """
    h, w = image_shape

    # Create 478 landmarks (MediaPipe standard)
    landmarks = np.random.rand(478, 3).astype(np.float32)

    # Create bounding box
    bbox = BoundingBox(x_min=0.2, y_min=0.2, width=0.6, height=0.6)

    return FaceLandmarks(
        landmarks=landmarks,
        bounding_box=bbox,
        timestamp=0.0,
        frame_number=0
    )


def create_dummy_emotion_prediction(frame_number: int = 0) -> EmotionPrediction:
    """Create dummy emotion prediction for demonstration.

    Args:
        frame_number: Frame number for the prediction

    Returns:
        EmotionPrediction object with synthetic data
    """
    # Generate random probabilities that sum to 1
    probs = np.random.dirichlet(np.ones(7))

    probabilities = {
        EmotionType.NEUTRAL: float(probs[0]),
        EmotionType.HAPPY: float(probs[1]),
        EmotionType.SAD: float(probs[2]),
        EmotionType.ANGRY: float(probs[3]),
        EmotionType.SURPRISED: float(probs[4]),
        EmotionType.FEARFUL: float(probs[5]),
        EmotionType.DISGUSTED: float(probs[6]),
    }

    # Get emotion with highest probability
    emotion = max(probabilities.items(), key=lambda x: x[1])[0]
    confidence = probabilities[emotion]

    # Create some action units
    action_units = {
        ActionUnitType.AU12: ActionUnit(
            au_type=ActionUnitType.AU12,
            intensity=0.7,
            present=True,
            confidence=0.9
        ),
        ActionUnitType.AU6: ActionUnit(
            au_type=ActionUnitType.AU6,
            intensity=0.5,
            present=True,
            confidence=0.85
        ),
    }

    return EmotionPrediction(
        emotion=emotion,
        confidence=confidence,
        probabilities=probabilities,
        action_units=action_units,
        timestamp=frame_number * 33.33,  # ~30 fps
        frame_number=frame_number
    )


def demo_face_overlay():
    """Demonstrate FaceOverlay class."""
    print("=== Face Overlay Demo ===")

    # Create a blank image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (50, 50, 50)  # Dark gray background

    # Create dummy landmarks
    landmarks = create_dummy_landmarks((480, 640))

    # Create overlay with custom style
    style = OverlayStyle(
        landmark_color=(0, 255, 0),
        landmark_radius=3,
        connection_color=(255, 255, 255),
        bbox_color=(255, 0, 0),
        bbox_thickness=2
    )
    overlay = FaceOverlay(style=style)

    # Draw complete face
    overlay.draw_complete_face(image, landmarks)

    # Save result
    output_path = Path("output/overlay_demo.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Saved face overlay demo to {output_path}")


def demo_emotion_display():
    """Demonstrate EmotionDisplay class."""
    print("\n=== Emotion Display Demo ===")

    # Create a blank image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (50, 50, 50)

    # Create dummy emotion prediction
    prediction = create_dummy_emotion_prediction(frame_number=1)
    prediction.emotion = EmotionType.HAPPY
    prediction.confidence = 0.85

    # Create display with custom style
    style = DisplayStyle(
        text_scale=0.8,
        show_probabilities=True,
        show_confidence=True,
        show_action_units=True,
        position="top_left"
    )
    display = EmotionDisplay(style=style)

    # Draw complete display
    display.draw_complete_display(image, prediction, top_n_emotions=5)

    # Save result
    output_path = Path("output/emotion_display_demo.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Saved emotion display demo to {output_path}")


def demo_statistical_plots():
    """Demonstrate statistical plotting functions."""
    print("\n=== Statistical Plots Demo ===")

    # Create a series of emotion predictions
    predictions = []
    for i in range(100):
        pred = create_dummy_emotion_prediction(frame_number=i)
        predictions.append(pred)

    output_dir = Path("output/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Emotion distribution
    fig1 = plot_emotion_distribution(
        predictions,
        plot_type="bar",
        save_path=output_dir / "emotion_distribution.png"
    )
    print(f"Saved emotion distribution plot")

    # 2. Emotion timeline
    fig2 = plot_emotion_timeline(
        predictions,
        save_path=output_dir / "emotion_timeline.png"
    )
    print(f"Saved emotion timeline plot")

    # 3. Confidence over time
    fig3 = plot_confidence_over_time(
        predictions,
        save_path=output_dir / "confidence_over_time.png"
    )
    print(f"Saved confidence plot")

    # 4. Summary plot
    fig4 = plot_emotion_summary(
        predictions,
        save_path=output_dir / "emotion_summary.png"
    )
    print(f"Saved summary plot")


def demo_heatmaps():
    """Demonstrate EmotionHeatmap class."""
    print("\n=== Heatmap Demo ===")

    # Create a series of emotion predictions
    predictions = []
    for i in range(150):
        pred = create_dummy_emotion_prediction(frame_number=i)
        predictions.append(pred)

    output_dir = Path("output/heatmaps")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create heatmap generator
    heatmap = EmotionHeatmap(cmap="YlOrRd")

    # 1. Temporal heatmap
    fig1 = heatmap.create_temporal_heatmap(
        predictions,
        window_size=10,
        save_path=output_dir / "temporal_heatmap.png"
    )
    print(f"Saved temporal heatmap")

    # 2. Transition heatmap
    fig2 = heatmap.create_transition_heatmap(
        predictions,
        normalize=True,
        save_path=output_dir / "transition_heatmap.png"
    )
    print(f"Saved transition heatmap")

    # 3. Correlation heatmap
    fig3 = heatmap.create_correlation_heatmap(
        predictions,
        save_path=output_dir / "correlation_heatmap.png"
    )
    print(f"Saved correlation heatmap")

    # 4. Sliding window heatmap
    fig4 = heatmap.create_sliding_window_heatmap(
        predictions,
        window_size=30,
        stride=10,
        save_path=output_dir / "sliding_window_heatmap.png"
    )
    print(f"Saved sliding window heatmap")


def demo_combined_visualization():
    """Demonstrate combining face overlay and emotion display."""
    print("\n=== Combined Visualization Demo ===")

    # Create a blank image
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    image[:] = (50, 50, 50)

    # Create dummy data
    landmarks = create_dummy_landmarks((480, 640))
    prediction = create_dummy_emotion_prediction(frame_number=1)
    prediction.emotion = EmotionType.SURPRISED
    prediction.confidence = 0.92

    # Create visualizers
    overlay = FaceOverlay()
    display = EmotionDisplay()

    # Draw face landmarks
    overlay.draw_complete_face(image, landmarks, draw_all_landmarks=True, draw_all_connections=True)

    # Draw emotion information
    display.draw_complete_display(image, prediction, top_n_emotions=3)

    # Save result
    output_path = Path("output/combined_demo.jpg")
    output_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Saved combined visualization demo to {output_path}")


def main():
    """Run all demonstration functions."""
    print("Emotion Detector Visualization Module Demo")
    print("=" * 50)

    # Create output directory
    Path("output").mkdir(exist_ok=True)

    # Run demonstrations
    demo_face_overlay()
    demo_emotion_display()
    demo_statistical_plots()
    demo_heatmaps()
    demo_combined_visualization()

    print("\n" + "=" * 50)
    print("All demos completed! Check the 'output' directory for results.")


if __name__ == "__main__":
    main()
