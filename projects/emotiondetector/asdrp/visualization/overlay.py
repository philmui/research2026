"""Face overlay visualization for drawing landmarks and bounding boxes on frames.

This module provides the FaceOverlay class for drawing facial landmarks, connections,
and bounding boxes on video frames for visualization purposes.
"""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import numpy.typing as npt

from asdrp.face.base import BoundingBox, FaceLandmarks
from asdrp.face.landmarker import LANDMARK_GROUPS


@dataclass
class OverlayStyle:
    """Configuration for overlay drawing styles.

    Attributes:
        landmark_color: BGR color for landmark points (default: green)
        landmark_radius: Radius of landmark circles in pixels
        connection_color: BGR color for connections between landmarks (default: white)
        connection_thickness: Thickness of connection lines in pixels
        bbox_color: BGR color for bounding box (default: blue)
        bbox_thickness: Thickness of bounding box lines in pixels
        fill_landmarks: Whether to fill landmark circles
        draw_indices: Whether to draw landmark indices as text
        text_color: BGR color for text labels (default: yellow)
        text_scale: Scale factor for text size
        text_thickness: Thickness of text in pixels
    """

    landmark_color: tuple[int, int, int] = (0, 255, 0)  # Green
    landmark_radius: int = 2
    connection_color: tuple[int, int, int] = (255, 255, 255)  # White
    connection_thickness: int = 1
    bbox_color: tuple[int, int, int] = (255, 0, 0)  # Blue
    bbox_thickness: int = 2
    fill_landmarks: bool = True
    draw_indices: bool = False
    text_color: tuple[int, int, int] = (0, 255, 255)  # Yellow
    text_scale: float = 0.3
    text_thickness: int = 1


# Define connections for different facial regions
# Format: list of (start_index, end_index) tuples
LEFT_EYE_CONNECTIONS = [
    (33, 133), (133, 153), (153, 154), (154, 155), (155, 133),
    (33, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133)
]

RIGHT_EYE_CONNECTIONS = [
    (362, 263), (263, 380), (380, 381), (381, 382), (382, 263),
    (362, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 263)
]

LEFT_EYEBROW_CONNECTIONS = [
    (70, 63), (63, 105), (105, 66), (66, 107)
]

RIGHT_EYEBROW_CONNECTIONS = [
    (300, 293), (293, 334), (334, 296), (296, 336)
]

NOSE_CONNECTIONS = [
    (6, 168), (168, 1), (1, 98), (1, 327)
]

MOUTH_CONNECTIONS = [
    (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267), (267, 269),
    (269, 270), (270, 409), (409, 291), (291, 375), (375, 321), (321, 405),
    (405, 314), (314, 17), (17, 84), (84, 181), (181, 91), (91, 146), (146, 61)
]

FACE_OVAL_CONNECTIONS = [
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389),
    (389, 356), (356, 454), (454, 323), (323, 361), (361, 288), (288, 397),
    (397, 365), (365, 379), (379, 378), (378, 400), (400, 377), (377, 152),
    (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21),
    (21, 54), (54, 103), (103, 67), (67, 109), (109, 10)
]

ALL_CONNECTIONS = (
    LEFT_EYE_CONNECTIONS + RIGHT_EYE_CONNECTIONS +
    LEFT_EYEBROW_CONNECTIONS + RIGHT_EYEBROW_CONNECTIONS +
    NOSE_CONNECTIONS + MOUTH_CONNECTIONS + FACE_OVAL_CONNECTIONS
)


class FaceOverlay:
    """Draw facial landmarks and annotations on video frames.

    This class provides methods to draw facial landmarks, connections between
    landmarks, and bounding boxes on images for visualization purposes.
    """

    def __init__(self, style: Optional[OverlayStyle] = None):
        """Initialize the face overlay renderer.

        Args:
            style: Optional OverlayStyle configuration. If None, default style is used.
        """
        self.style = style or OverlayStyle()

    def draw_landmarks(
        self,
        image: npt.NDArray[np.uint8],
        face_landmarks: FaceLandmarks,
        color: Optional[tuple[int, int, int]] = None
    ) -> npt.NDArray[np.uint8]:
        """Draw all facial landmarks on the image.

        Args:
            image: Input image as BGR numpy array of shape (H, W, 3)
            face_landmarks: FaceLandmarks object containing landmark coordinates
            color: Optional BGR color override for landmarks

        Returns:
            Image with landmarks drawn (modifies input image in-place and returns it)
        """
        h, w = image.shape[:2]
        landmark_color = color or self.style.landmark_color

        # Convert normalized coordinates to absolute pixel coordinates
        absolute_landmarks = face_landmarks.to_absolute(w, h)

        # Draw each landmark
        for i, landmark in enumerate(absolute_landmarks):
            x, y = int(landmark[0]), int(landmark[1])

            # Draw landmark point
            if self.style.fill_landmarks:
                cv2.circle(image, (x, y), self.style.landmark_radius, landmark_color, -1)
            else:
                cv2.circle(image, (x, y), self.style.landmark_radius, landmark_color, 1)

            # Optionally draw landmark index
            if self.style.draw_indices:
                cv2.putText(
                    image,
                    str(i),
                    (x + 2, y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.style.text_scale,
                    self.style.text_color,
                    self.style.text_thickness
                )

        return image

    def draw_landmark_group(
        self,
        image: npt.NDArray[np.uint8],
        face_landmarks: FaceLandmarks,
        group_name: str,
        color: Optional[tuple[int, int, int]] = None
    ) -> npt.NDArray[np.uint8]:
        """Draw specific landmark group on the image.

        Args:
            image: Input image as BGR numpy array of shape (H, W, 3)
            face_landmarks: FaceLandmarks object containing landmark coordinates
            group_name: Name of landmark group ('left_eye', 'right_eye', 'mouth_outer', etc.)
            color: Optional BGR color override for landmarks

        Returns:
            Image with landmark group drawn (modifies input image in-place and returns it)
        """
        if group_name not in LANDMARK_GROUPS:
            raise ValueError(f"Unknown landmark group: {group_name}")

        h, w = image.shape[:2]
        landmark_color = color or self.style.landmark_color

        # Get indices for this group
        indices = [idx.value for idx in LANDMARK_GROUPS[group_name]]
        absolute_landmarks = face_landmarks.to_absolute(w, h)

        # Draw landmarks for this group
        for idx in indices:
            landmark = absolute_landmarks[idx]
            x, y = int(landmark[0]), int(landmark[1])

            if self.style.fill_landmarks:
                cv2.circle(image, (x, y), self.style.landmark_radius, landmark_color, -1)
            else:
                cv2.circle(image, (x, y), self.style.landmark_radius, landmark_color, 1)

        return image

    def draw_connections(
        self,
        image: npt.NDArray[np.uint8],
        face_landmarks: FaceLandmarks,
        connections: Optional[list[tuple[int, int]]] = None,
        color: Optional[tuple[int, int, int]] = None
    ) -> npt.NDArray[np.uint8]:
        """Draw connections between facial landmarks.

        Args:
            image: Input image as BGR numpy array of shape (H, W, 3)
            face_landmarks: FaceLandmarks object containing landmark coordinates
            connections: Optional list of (start_idx, end_idx) tuples. If None, draws all connections.
            color: Optional BGR color override for connections

        Returns:
            Image with connections drawn (modifies input image in-place and returns it)
        """
        h, w = image.shape[:2]
        connection_color = color or self.style.connection_color
        connections_to_draw = connections or ALL_CONNECTIONS

        # Convert normalized coordinates to absolute pixel coordinates
        absolute_landmarks = face_landmarks.to_absolute(w, h)

        # Draw each connection
        for start_idx, end_idx in connections_to_draw:
            if start_idx >= len(absolute_landmarks) or end_idx >= len(absolute_landmarks):
                continue

            start = absolute_landmarks[start_idx]
            end = absolute_landmarks[end_idx]

            start_pt = (int(start[0]), int(start[1]))
            end_pt = (int(end[0]), int(end[1]))

            cv2.line(
                image,
                start_pt,
                end_pt,
                connection_color,
                self.style.connection_thickness
            )

        return image

    def draw_eyes(
        self,
        image: npt.NDArray[np.uint8],
        face_landmarks: FaceLandmarks,
        color: Optional[tuple[int, int, int]] = None
    ) -> npt.NDArray[np.uint8]:
        """Draw eye landmarks and connections.

        Args:
            image: Input image as BGR numpy array
            face_landmarks: FaceLandmarks object
            color: Optional BGR color override

        Returns:
            Image with eyes drawn
        """
        self.draw_landmark_group(image, face_landmarks, "left_eye", color)
        self.draw_landmark_group(image, face_landmarks, "right_eye", color)
        self.draw_connections(image, face_landmarks, LEFT_EYE_CONNECTIONS + RIGHT_EYE_CONNECTIONS, color)
        return image

    def draw_eyebrows(
        self,
        image: npt.NDArray[np.uint8],
        face_landmarks: FaceLandmarks,
        color: Optional[tuple[int, int, int]] = None
    ) -> npt.NDArray[np.uint8]:
        """Draw eyebrow landmarks and connections.

        Args:
            image: Input image as BGR numpy array
            face_landmarks: FaceLandmarks object
            color: Optional BGR color override

        Returns:
            Image with eyebrows drawn
        """
        self.draw_landmark_group(image, face_landmarks, "left_eyebrow", color)
        self.draw_landmark_group(image, face_landmarks, "right_eyebrow", color)
        self.draw_connections(image, face_landmarks, LEFT_EYEBROW_CONNECTIONS + RIGHT_EYEBROW_CONNECTIONS, color)
        return image

    def draw_mouth(
        self,
        image: npt.NDArray[np.uint8],
        face_landmarks: FaceLandmarks,
        color: Optional[tuple[int, int, int]] = None
    ) -> npt.NDArray[np.uint8]:
        """Draw mouth landmarks and connections.

        Args:
            image: Input image as BGR numpy array
            face_landmarks: FaceLandmarks object
            color: Optional BGR color override

        Returns:
            Image with mouth drawn
        """
        self.draw_landmark_group(image, face_landmarks, "mouth_outer", color)
        self.draw_connections(image, face_landmarks, MOUTH_CONNECTIONS, color)
        return image

    def draw_face_oval(
        self,
        image: npt.NDArray[np.uint8],
        face_landmarks: FaceLandmarks,
        color: Optional[tuple[int, int, int]] = None
    ) -> npt.NDArray[np.uint8]:
        """Draw face oval landmarks and connections.

        Args:
            image: Input image as BGR numpy array
            face_landmarks: FaceLandmarks object
            color: Optional BGR color override

        Returns:
            Image with face oval drawn
        """
        self.draw_landmark_group(image, face_landmarks, "face_oval", color)
        self.draw_connections(image, face_landmarks, FACE_OVAL_CONNECTIONS, color)
        return image

    def draw_bounding_box(
        self,
        image: npt.NDArray[np.uint8],
        bounding_box: BoundingBox,
        color: Optional[tuple[int, int, int]] = None,
        label: Optional[str] = None
    ) -> npt.NDArray[np.uint8]:
        """Draw bounding box around detected face.

        Args:
            image: Input image as BGR numpy array
            bounding_box: BoundingBox object with normalized coordinates
            color: Optional BGR color override
            label: Optional text label to display above the box

        Returns:
            Image with bounding box drawn (modifies input image in-place and returns it)
        """
        h, w = image.shape[:2]
        bbox_color = color or self.style.bbox_color

        # Convert to absolute coordinates
        x_min, y_min, x_max, y_max = bounding_box.to_absolute(w, h)

        # Draw rectangle
        cv2.rectangle(
            image,
            (x_min, y_min),
            (x_max, y_max),
            bbox_color,
            self.style.bbox_thickness
        )

        # Draw label if provided
        if label:
            # Get text size for background rectangle
            text_size = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                self.style.text_scale * 1.5,
                self.style.text_thickness
            )[0]

            # Draw background rectangle for text
            cv2.rectangle(
                image,
                (x_min, y_min - text_size[1] - 10),
                (x_min + text_size[0] + 10, y_min),
                bbox_color,
                -1
            )

            # Draw text
            cv2.putText(
                image,
                label,
                (x_min + 5, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.style.text_scale * 1.5,
                (255, 255, 255),
                self.style.text_thickness
            )

        return image

    def draw_complete_face(
        self,
        image: npt.NDArray[np.uint8],
        face_landmarks: FaceLandmarks,
        draw_bbox: bool = True,
        draw_all_landmarks: bool = True,
        draw_all_connections: bool = True
    ) -> npt.NDArray[np.uint8]:
        """Draw complete face visualization with all features.

        Args:
            image: Input image as BGR numpy array
            face_landmarks: FaceLandmarks object
            draw_bbox: Whether to draw bounding box
            draw_all_landmarks: Whether to draw all landmarks
            draw_all_connections: Whether to draw connections between landmarks

        Returns:
            Image with complete face visualization (modifies input image in-place and returns it)
        """
        if draw_all_connections:
            self.draw_connections(image, face_landmarks)

        if draw_all_landmarks:
            self.draw_landmarks(image, face_landmarks)

        if draw_bbox and face_landmarks.bounding_box is not None:
            self.draw_bounding_box(image, face_landmarks.bounding_box)

        return image
