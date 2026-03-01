"""Emotion display visualization for showing emotion predictions on frames.

This module provides the EmotionDisplay class for drawing emotion labels,
confidence scores, probability bars, and other emotion-related information
on video frames.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from asdrp.emotion.base import EmotionPrediction, EmotionType


# Color scheme for emotions (BGR format)
EMOTION_COLORS = {
    EmotionType.NEUTRAL: (200, 200, 200),  # Gray
    EmotionType.HAPPY: (0, 255, 255),      # Yellow
    EmotionType.SAD: (255, 0, 0),          # Blue
    EmotionType.ANGRY: (0, 0, 255),        # Red
    EmotionType.SURPRISED: (255, 128, 0),  # Orange
    EmotionType.FEARFUL: (128, 0, 128),    # Purple
    EmotionType.DISGUSTED: (0, 128, 0),    # Dark Green
}


@dataclass
class DisplayStyle:
    """Configuration for emotion display styles.

    Attributes:
        text_color: BGR color for text labels
        text_scale: Scale factor for text size
        text_thickness: Thickness of text in pixels
        bar_height: Height of probability bars in pixels
        bar_width: Width of probability bars in pixels
        bar_spacing: Spacing between probability bars in pixels
        background_alpha: Alpha transparency for background overlays (0.0-1.0)
        show_probabilities: Whether to show probability bars for all emotions
        show_confidence: Whether to show overall confidence score
        show_action_units: Whether to show detected action units
        position: Position to draw emotion info ('top_left', 'top_right', 'bottom_left', 'bottom_right')
    """

    text_color: tuple[int, int, int] = (255, 255, 255)  # White
    text_scale: float = 0.7
    text_thickness: int = 2
    bar_height: int = 20
    bar_width: int = 200
    bar_spacing: int = 5
    background_alpha: float = 0.6
    show_probabilities: bool = True
    show_confidence: bool = True
    show_action_units: bool = False
    position: str = "top_left"


class EmotionDisplay:
    """Display emotion predictions and information on video frames.

    This class provides methods to visualize emotion detection results including
    emotion labels, confidence scores, probability distributions, and action units.
    """

    def __init__(self, style: Optional[DisplayStyle] = None):
        """Initialize the emotion display renderer.

        Args:
            style: Optional DisplayStyle configuration. If None, default style is used.
        """
        self.style = style or DisplayStyle()

    def _get_position(self, image_shape: tuple[int, int], offset: tuple[int, int] = (10, 30)) -> tuple[int, int]:
        """Calculate position based on image shape and position setting.

        Args:
            image_shape: (height, width) of the image
            offset: (x_offset, y_offset) from the corner

        Returns:
            (x, y) position for drawing
        """
        h, w = image_shape
        x_off, y_off = offset

        if self.style.position == "top_left":
            return (x_off, y_off)
        elif self.style.position == "top_right":
            return (w - self.style.bar_width - x_off, y_off)
        elif self.style.position == "bottom_left":
            return (x_off, h - 200)
        elif self.style.position == "bottom_right":
            return (w - self.style.bar_width - x_off, h - 200)
        else:
            return (x_off, y_off)

    def _draw_background_rect(
        self,
        image: npt.NDArray[np.uint8],
        top_left: tuple[int, int],
        bottom_right: tuple[int, int],
        color: tuple[int, int, int] = (0, 0, 0)
    ) -> None:
        """Draw semi-transparent background rectangle.

        Args:
            image: Image to draw on (modified in-place)
            top_left: (x, y) of top-left corner
            bottom_right: (x, y) of bottom-right corner
            color: BGR color for background
        """
        overlay = image.copy()
        cv2.rectangle(overlay, top_left, bottom_right, color, -1)
        cv2.addWeighted(overlay, self.style.background_alpha, image, 1 - self.style.background_alpha, 0, image)

    def draw_emotion_label(
        self,
        image: npt.NDArray[np.uint8],
        emotion_prediction: EmotionPrediction,
        position: Optional[tuple[int, int]] = None
    ) -> npt.NDArray[np.uint8]:
        """Draw primary emotion label with confidence.

        Args:
            image: Input image as BGR numpy array of shape (H, W, 3)
            emotion_prediction: EmotionPrediction object
            position: Optional (x, y) position override

        Returns:
            Image with emotion label drawn (modifies input image in-place and returns it)
        """
        if position is None:
            position = self._get_position(image.shape[:2])

        emotion = emotion_prediction.emotion
        confidence = emotion_prediction.confidence
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        # Format label text
        label = f"{emotion.value.upper()}"
        if self.style.show_confidence:
            label += f" ({confidence:.1%})"

        # Get text size for background
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.text_scale,
            self.style.text_thickness
        )[0]

        # Draw background rectangle
        padding = 10
        self._draw_background_rect(
            image,
            (position[0] - padding, position[1] - text_size[1] - padding),
            (position[0] + text_size[0] + padding, position[1] + padding),
            (0, 0, 0)
        )

        # Draw colored indicator box
        indicator_size = text_size[1]
        cv2.rectangle(
            image,
            (position[0] - padding, position[1] - text_size[1] - padding),
            (position[0] - padding + indicator_size, position[1] + padding),
            color,
            -1
        )

        # Draw text
        cv2.putText(
            image,
            label,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.text_scale,
            self.style.text_color,
            self.style.text_thickness,
            cv2.LINE_AA
        )

        return image

    def draw_probability_bars(
        self,
        image: npt.NDArray[np.uint8],
        emotion_prediction: EmotionPrediction,
        position: Optional[tuple[int, int]] = None,
        top_n: Optional[int] = None
    ) -> npt.NDArray[np.uint8]:
        """Draw probability bars for emotions.

        Args:
            image: Input image as BGR numpy array
            emotion_prediction: EmotionPrediction object
            position: Optional (x, y) position override
            top_n: Optional limit to show only top N emotions

        Returns:
            Image with probability bars drawn (modifies input image in-place and returns it)
        """
        if position is None:
            base_pos = self._get_position(image.shape[:2])
            position = (base_pos[0], base_pos[1] + 40)

        # Get emotions sorted by probability
        sorted_emotions = sorted(
            emotion_prediction.probabilities.items(),
            key=lambda x: x[1],
            reverse=True
        )

        if top_n is not None:
            sorted_emotions = sorted_emotions[:top_n]

        x, y = position

        for emotion, probability in sorted_emotions:
            color = EMOTION_COLORS.get(emotion, (255, 255, 255))

            # Draw emotion name
            label = f"{emotion.value.capitalize()}"
            cv2.putText(
                image,
                label,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.style.text_scale * 0.6,
                self.style.text_color,
                1,
                cv2.LINE_AA
            )

            # Draw probability bar background
            bar_y = y + 5
            cv2.rectangle(
                image,
                (x, bar_y),
                (x + self.style.bar_width, bar_y + self.style.bar_height),
                (50, 50, 50),
                -1
            )

            # Draw filled probability bar
            bar_fill_width = int(self.style.bar_width * probability)
            cv2.rectangle(
                image,
                (x, bar_y),
                (x + bar_fill_width, bar_y + self.style.bar_height),
                color,
                -1
            )

            # Draw probability percentage
            percentage_text = f"{probability:.1%}"
            cv2.putText(
                image,
                percentage_text,
                (x + self.style.bar_width + 10, y + 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.style.text_scale * 0.5,
                self.style.text_color,
                1,
                cv2.LINE_AA
            )

            y += self.style.bar_height + self.style.bar_spacing + 15

        return image

    def draw_action_units(
        self,
        image: npt.NDArray[np.uint8],
        emotion_prediction: EmotionPrediction,
        position: Optional[tuple[int, int]] = None,
        threshold: float = 0.3
    ) -> npt.NDArray[np.uint8]:
        """Draw active action units.

        Args:
            image: Input image as BGR numpy array
            emotion_prediction: EmotionPrediction object
            position: Optional (x, y) position override
            threshold: Minimum intensity to display an action unit

        Returns:
            Image with action units drawn (modifies input image in-place and returns it)
        """
        if not emotion_prediction.action_units:
            return image

        if position is None:
            h, w = image.shape[:2]
            position = (10, h - 100)

        # Get active action units
        active_aus = [
            (au.au_type, au.intensity)
            for au in emotion_prediction.action_units.values()
            if au.intensity >= threshold
        ]

        if not active_aus:
            return image

        x, y = position

        # Draw title
        cv2.putText(
            image,
            "Action Units:",
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.style.text_scale * 0.6,
            self.style.text_color,
            1,
            cv2.LINE_AA
        )

        y += 20

        # Draw each active AU
        for au_type, intensity in active_aus:
            au_text = f"{str(au_type)}: {intensity:.2f}"
            cv2.putText(
                image,
                au_text,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.style.text_scale * 0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )
            y += 15

        return image

    def draw_complete_display(
        self,
        image: npt.NDArray[np.uint8],
        emotion_prediction: EmotionPrediction,
        show_bars: Optional[bool] = None,
        show_aus: Optional[bool] = None,
        top_n_emotions: int = 3
    ) -> npt.NDArray[np.uint8]:
        """Draw complete emotion display with all information.

        Args:
            image: Input image as BGR numpy array
            emotion_prediction: EmotionPrediction object
            show_bars: Override style setting for probability bars
            show_aus: Override style setting for action units
            top_n_emotions: Number of top emotions to show in bars

        Returns:
            Image with complete emotion display (modifies input image in-place and returns it)
        """
        # Draw primary emotion label
        self.draw_emotion_label(image, emotion_prediction)

        # Draw probability bars if enabled
        if show_bars if show_bars is not None else self.style.show_probabilities:
            self.draw_probability_bars(image, emotion_prediction, top_n=top_n_emotions)

        # Draw action units if enabled
        if show_aus if show_aus is not None else self.style.show_action_units:
            self.draw_action_units(image, emotion_prediction)

        return image

    def draw_timeline_marker(
        self,
        image: npt.NDArray[np.uint8],
        emotion_predictions: list[EmotionPrediction],
        current_frame: int,
        timeline_height: int = 50,
        position: str = "bottom"
    ) -> npt.NDArray[np.uint8]:
        """Draw emotion timeline at the bottom or top of the frame.

        Args:
            image: Input image as BGR numpy array
            emotion_predictions: List of EmotionPrediction objects for timeline
            current_frame: Current frame index
            timeline_height: Height of timeline in pixels
            position: 'top' or 'bottom'

        Returns:
            Image with timeline drawn (modifies input image in-place and returns it)
        """
        h, w = image.shape[:2]

        if not emotion_predictions:
            return image

        # Calculate timeline position
        if position == "bottom":
            timeline_y = h - timeline_height
        else:
            timeline_y = 0

        # Draw semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (0, timeline_y), (w, timeline_y + timeline_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, image, 0.5, 0, image)

        # Calculate pixels per frame
        pixels_per_frame = w / len(emotion_predictions)

        # Draw emotion segments
        for i, pred in enumerate(emotion_predictions):
            x_start = int(i * pixels_per_frame)
            x_end = int((i + 1) * pixels_per_frame)
            color = EMOTION_COLORS.get(pred.emotion, (255, 255, 255))

            cv2.rectangle(
                image,
                (x_start, timeline_y),
                (x_end, timeline_y + timeline_height),
                color,
                -1
            )

        # Draw current frame marker
        marker_x = int(current_frame * pixels_per_frame)
        cv2.line(
            image,
            (marker_x, timeline_y),
            (marker_x, timeline_y + timeline_height),
            (255, 255, 255),
            3
        )

        return image

    def create_emotion_indicator(
        self,
        emotion: EmotionType,
        size: tuple[int, int] = (100, 100)
    ) -> npt.NDArray[np.uint8]:
        """Create a colored square indicator for an emotion.

        Args:
            emotion: EmotionType to create indicator for
            size: (width, height) of the indicator

        Returns:
            BGR image array with colored indicator and emotion label
        """
        indicator = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        color = EMOTION_COLORS.get(emotion, (255, 255, 255))

        # Fill with emotion color
        indicator[:] = color

        # Add emotion label
        label = emotion.value.upper()
        text_size = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            2
        )[0]

        text_x = (size[0] - text_size[0]) // 2
        text_y = (size[1] + text_size[1]) // 2

        cv2.putText(
            indicator,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        return indicator
