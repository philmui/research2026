"""Feature extraction from facial landmarks for emotion analysis.

This module provides classes and methods for extracting geometric features
and detecting Action Units (AUs) from facial landmarks. These features form
the basis for rule-based emotion classification.

References:
    Ekman, P., Friesen, W. V., & Hager, J. C. (2002). Facial Action Coding
    System: The Manual. Research Nexus.
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

from asdrp.emotion.base import ActionUnit, ActionUnitType
from asdrp.face.base import FaceLandmarks, FaceLandmarkIndex


class FeatureExtractor:
    """Extract geometric features and action units from facial landmarks.

    This class computes various geometric measurements from facial landmarks
    and uses them to detect the presence and intensity of facial action units
    according to the Facial Action Coding System (FACS).

    Attributes:
        au_threshold: Minimum intensity threshold for an AU to be considered present
    """

    def __init__(self, au_threshold: float = 0.3):
        """Initialize the feature extractor.

        Args:
            au_threshold: Threshold for AU presence detection (default: 0.3)
        """
        self.au_threshold = au_threshold

    def extract_features(self, landmarks: FaceLandmarks) -> dict[str, float]:
        """Extract all geometric features from facial landmarks.

        Args:
            landmarks: Facial landmarks to extract features from

        Returns:
            Dictionary of feature names to values
        """
        features = {}

        # Eye features
        features.update(self._extract_eye_features(landmarks))

        # Eyebrow features
        features.update(self._extract_eyebrow_features(landmarks))

        # Mouth features
        features.update(self._extract_mouth_features(landmarks))

        # Nose features
        features.update(self._extract_nose_features(landmarks))

        # Face shape features
        features.update(self._extract_face_features(landmarks))

        return features

    def detect_action_units(self, landmarks: FaceLandmarks) -> dict[ActionUnitType, ActionUnit]:
        """Detect all action units from facial landmarks.

        Args:
            landmarks: Facial landmarks to analyze

        Returns:
            Dictionary mapping ActionUnitType to detected ActionUnit objects
        """
        features = self.extract_features(landmarks)
        action_units = {}

        # Upper face AUs
        action_units[ActionUnitType.AU1] = self._detect_au1(features)
        action_units[ActionUnitType.AU2] = self._detect_au2(features)
        action_units[ActionUnitType.AU4] = self._detect_au4(features)
        action_units[ActionUnitType.AU5] = self._detect_au5(features)
        action_units[ActionUnitType.AU6] = self._detect_au6(features)
        action_units[ActionUnitType.AU7] = self._detect_au7(features)

        # Lower face AUs
        action_units[ActionUnitType.AU9] = self._detect_au9(features)
        action_units[ActionUnitType.AU10] = self._detect_au10(features)
        action_units[ActionUnitType.AU12] = self._detect_au12(features)
        action_units[ActionUnitType.AU15] = self._detect_au15(features)
        action_units[ActionUnitType.AU17] = self._detect_au17(features)
        action_units[ActionUnitType.AU20] = self._detect_au20(features)
        action_units[ActionUnitType.AU23] = self._detect_au23(features)
        action_units[ActionUnitType.AU25] = self._detect_au25(features)
        action_units[ActionUnitType.AU26] = self._detect_au26(features)

        return action_units

    # ============================================================================
    # Feature Extraction Methods
    # ============================================================================

    def _extract_eye_features(self, landmarks: FaceLandmarks) -> dict[str, float]:
        """Extract eye-related geometric features."""
        features = {}

        # Left eye
        left_eye_height = self._compute_distance(
            landmarks, FaceLandmarkIndex.LEFT_EYE_TOP_UPPER, FaceLandmarkIndex.LEFT_EYE_BOTTOM_LOWER
        )
        left_eye_width = self._compute_distance(
            landmarks, FaceLandmarkIndex.LEFT_EYE_OUTER_CORNER, FaceLandmarkIndex.LEFT_EYE_INNER_CORNER
        )
        features["left_eye_aspect_ratio"] = left_eye_height / (left_eye_width + 1e-6)

        # Right eye
        right_eye_height = self._compute_distance(
            landmarks, FaceLandmarkIndex.RIGHT_EYE_TOP_UPPER, FaceLandmarkIndex.RIGHT_EYE_BOTTOM_LOWER
        )
        right_eye_width = self._compute_distance(
            landmarks, FaceLandmarkIndex.RIGHT_EYE_OUTER_CORNER, FaceLandmarkIndex.RIGHT_EYE_INNER_CORNER
        )
        features["right_eye_aspect_ratio"] = right_eye_height / (right_eye_width + 1e-6)

        # Average eye openness
        features["eye_aspect_ratio"] = (features["left_eye_aspect_ratio"] + features["right_eye_aspect_ratio"]) / 2.0

        return features

    def _extract_eyebrow_features(self, landmarks: FaceLandmarks) -> dict[str, float]:
        """Extract eyebrow-related geometric features."""
        features = {}

        # Left eyebrow position relative to eye
        left_brow_to_eye = self._compute_distance(
            landmarks, FaceLandmarkIndex.LEFT_EYEBROW_CENTER, FaceLandmarkIndex.LEFT_EYE_TOP_UPPER
        )
        left_eye_height = self._compute_distance(
            landmarks, FaceLandmarkIndex.LEFT_EYE_TOP_UPPER, FaceLandmarkIndex.LEFT_EYE_BOTTOM_LOWER
        )
        features["left_eyebrow_height"] = left_brow_to_eye / (left_eye_height + 1e-6)

        # Right eyebrow position relative to eye
        right_brow_to_eye = self._compute_distance(
            landmarks, FaceLandmarkIndex.RIGHT_EYEBROW_CENTER, FaceLandmarkIndex.RIGHT_EYE_TOP_UPPER
        )
        right_eye_height = self._compute_distance(
            landmarks, FaceLandmarkIndex.RIGHT_EYE_TOP_UPPER, FaceLandmarkIndex.RIGHT_EYE_BOTTOM_LOWER
        )
        features["right_eyebrow_height"] = right_brow_to_eye / (right_eye_height + 1e-6)

        # Average eyebrow height
        features["eyebrow_height"] = (features["left_eyebrow_height"] + features["right_eyebrow_height"]) / 2.0

        # Inner eyebrow distance (for frowning)
        features["inner_eyebrow_distance"] = self._compute_distance(
            landmarks, FaceLandmarkIndex.LEFT_EYEBROW_INNER, FaceLandmarkIndex.RIGHT_EYEBROW_INNER
        )

        # Eyebrow angle (slant)
        left_brow_angle = self._compute_angle(
            landmarks,
            FaceLandmarkIndex.LEFT_EYEBROW_INNER,
            FaceLandmarkIndex.LEFT_EYEBROW_CENTER,
            FaceLandmarkIndex.LEFT_EYEBROW_OUTER
        )
        right_brow_angle = self._compute_angle(
            landmarks,
            FaceLandmarkIndex.RIGHT_EYEBROW_INNER,
            FaceLandmarkIndex.RIGHT_EYEBROW_CENTER,
            FaceLandmarkIndex.RIGHT_EYEBROW_OUTER
        )
        features["eyebrow_angle"] = (left_brow_angle + right_brow_angle) / 2.0

        return features

    def _extract_mouth_features(self, landmarks: FaceLandmarks) -> dict[str, float]:
        """Extract mouth-related geometric features."""
        features = {}

        # Mouth opening (height)
        mouth_height = self._compute_distance(
            landmarks, FaceLandmarkIndex.MOUTH_UPPER_LIP_TOP_CENTER, FaceLandmarkIndex.MOUTH_LOWER_LIP_BOTTOM_CENTER
        )
        mouth_width = self._compute_distance(
            landmarks, FaceLandmarkIndex.MOUTH_LEFT_CORNER, FaceLandmarkIndex.MOUTH_RIGHT_CORNER
        )
        features["mouth_aspect_ratio"] = mouth_height / (mouth_width + 1e-6)

        # Mouth width normalized by face width
        face_width = self._compute_distance(
            landmarks, FaceLandmarkIndex.FACE_OVAL_LEFT_MIDDLE, FaceLandmarkIndex.FACE_OVAL_RIGHT_MIDDLE
        )
        features["mouth_width_ratio"] = mouth_width / (face_width + 1e-6)

        # Mouth corner positions (smile/frown)
        left_corner = landmarks.get_landmark(FaceLandmarkIndex.MOUTH_LEFT_CORNER)
        right_corner = landmarks.get_landmark(FaceLandmarkIndex.MOUTH_RIGHT_CORNER)
        mouth_center = landmarks.get_landmark(FaceLandmarkIndex.MOUTH_UPPER_LIP_BOTTOM_CENTER)

        # Vertical position of corners relative to center (positive = smile, negative = frown)
        features["mouth_corner_height"] = (
            (mouth_center[1] - left_corner[1]) + (mouth_center[1] - right_corner[1])
        ) / 2.0

        # Lip thickness ratio
        upper_lip_thickness = self._compute_distance(
            landmarks, FaceLandmarkIndex.MOUTH_UPPER_LIP_TOP_CENTER, FaceLandmarkIndex.MOUTH_UPPER_LIP_BOTTOM_CENTER
        )
        lower_lip_thickness = self._compute_distance(
            landmarks, FaceLandmarkIndex.MOUTH_LOWER_LIP_TOP_CENTER, FaceLandmarkIndex.MOUTH_LOWER_LIP_BOTTOM_CENTER
        )
        features["lip_thickness_ratio"] = upper_lip_thickness / (lower_lip_thickness + 1e-6)

        return features

    def _extract_nose_features(self, landmarks: FaceLandmarks) -> dict[str, float]:
        """Extract nose-related geometric features."""
        features = {}

        # Nostril width
        nostril_width = self._compute_distance(
            landmarks, FaceLandmarkIndex.NOSE_LEFT_NOSTRIL, FaceLandmarkIndex.NOSE_RIGHT_NOSTRIL
        )

        # Nose width (at alae)
        nose_width = self._compute_distance(
            landmarks, FaceLandmarkIndex.NOSE_LEFT_ALAR, FaceLandmarkIndex.NOSE_RIGHT_ALAR
        )

        features["nostril_width_ratio"] = nostril_width / (nose_width + 1e-6)

        return features

    def _extract_face_features(self, landmarks: FaceLandmarks) -> dict[str, float]:
        """Extract overall face shape features."""
        features = {}

        # Face width
        face_width = self._compute_distance(
            landmarks, FaceLandmarkIndex.FACE_OVAL_LEFT_MIDDLE, FaceLandmarkIndex.FACE_OVAL_RIGHT_MIDDLE
        )

        # Face height
        face_height = self._compute_distance(
            landmarks, FaceLandmarkIndex.FACE_OVAL_FOREHEAD_CENTER, FaceLandmarkIndex.FACE_OVAL_CHIN_CENTER
        )

        features["face_aspect_ratio"] = face_height / (face_width + 1e-6)

        return features

    # ============================================================================
    # Action Unit Detection Methods
    # ============================================================================

    def _detect_au1(self, features: dict[str, float]) -> ActionUnit:
        """AU1: Inner Brow Raiser.

        Detected by raised inner eyebrow position relative to baseline.
        """
        # Higher eyebrow height indicates raised brows
        eyebrow_height = features.get("eyebrow_height", 0.0)

        # Baseline is typically around 1.5-2.0, raised is > 2.2
        intensity = np.clip((eyebrow_height - 1.8) / 0.8, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU1,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au2(self, features: dict[str, float]) -> ActionUnit:
        """AU2: Outer Brow Raiser.

        Detected by raised outer eyebrow position.
        """
        eyebrow_height = features.get("eyebrow_height", 0.0)

        # Similar to AU1 but focuses on outer brow
        intensity = np.clip((eyebrow_height - 1.8) / 0.8, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU2,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au4(self, features: dict[str, float]) -> ActionUnit:
        """AU4: Brow Lowerer.

        Detected by lowered and drawn together eyebrows (frowning).
        """
        eyebrow_height = features.get("eyebrow_height", 0.0)
        inner_distance = features.get("inner_eyebrow_distance", 0.0)

        # Lower eyebrows and closer together
        height_component = np.clip((1.5 - eyebrow_height) / 0.5, 0.0, 1.0)
        distance_component = np.clip((0.15 - inner_distance) / 0.05, 0.0, 1.0)

        intensity = (height_component + distance_component) / 2.0

        return ActionUnit(
            au_type=ActionUnitType.AU4,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au5(self, features: dict[str, float]) -> ActionUnit:
        """AU5: Upper Lid Raiser.

        Detected by increased eye opening (wide eyes).
        """
        eye_ratio = features.get("eye_aspect_ratio", 0.0)

        # Normal eye aspect ratio is ~0.25-0.3, wide eyes > 0.35
        intensity = np.clip((eye_ratio - 0.28) / 0.15, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU5,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au6(self, features: dict[str, float]) -> ActionUnit:
        """AU6: Cheek Raiser.

        Detected by raised cheeks, often co-occurring with smiling.
        This is approximated by the combination of eye narrowing and mouth corners.
        """
        eye_ratio = features.get("eye_aspect_ratio", 0.0)
        mouth_corner = features.get("mouth_corner_height", 0.0)

        # Cheek raising causes slight eye narrowing when smiling
        eye_component = np.clip((0.25 - eye_ratio) / 0.08, 0.0, 1.0) if mouth_corner > 0 else 0.0
        mouth_component = np.clip(mouth_corner / 0.03, 0.0, 1.0)

        intensity = (eye_component * 0.4 + mouth_component * 0.6)

        return ActionUnit(
            au_type=ActionUnitType.AU6,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au7(self, features: dict[str, float]) -> ActionUnit:
        """AU7: Lid Tightener.

        Detected by tightened eyelids (narrowed eyes).
        """
        eye_ratio = features.get("eye_aspect_ratio", 0.0)

        # Tightened lids have lower aspect ratio
        intensity = np.clip((0.22 - eye_ratio) / 0.08, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU7,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au9(self, features: dict[str, float]) -> ActionUnit:
        """AU9: Nose Wrinkler.

        Detected by nostril flaring and nose wrinkling.
        """
        nostril_ratio = features.get("nostril_width_ratio", 0.0)

        # Nose wrinkling increases nostril width ratio
        intensity = np.clip((nostril_ratio - 0.65) / 0.15, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU9,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au10(self, features: dict[str, float]) -> ActionUnit:
        """AU10: Upper Lip Raiser.

        Detected by raised upper lip exposing teeth.
        """
        mouth_ratio = features.get("mouth_aspect_ratio", 0.0)
        lip_ratio = features.get("lip_thickness_ratio", 0.0)

        # Upper lip raising increases mouth opening and changes lip ratio
        intensity = np.clip((mouth_ratio - 0.3) / 0.4, 0.0, 1.0) * np.clip((1.2 - lip_ratio) / 0.4, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU10,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au12(self, features: dict[str, float]) -> ActionUnit:
        """AU12: Lip Corner Puller (Smile).

        Detected by pulled lip corners (smiling action).
        """
        mouth_corner = features.get("mouth_corner_height", 0.0)
        mouth_width = features.get("mouth_width_ratio", 0.0)

        # Smiling raises mouth corners and may widen mouth slightly
        corner_component = np.clip(mouth_corner / 0.03, 0.0, 1.0)
        width_component = np.clip((mouth_width - 0.45) / 0.1, 0.0, 1.0)

        intensity = corner_component * 0.8 + width_component * 0.2

        return ActionUnit(
            au_type=ActionUnitType.AU12,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au15(self, features: dict[str, float]) -> ActionUnit:
        """AU15: Lip Corner Depressor.

        Detected by lowered lip corners (frowning action).
        """
        mouth_corner = features.get("mouth_corner_height", 0.0)

        # Negative mouth corner height indicates depression
        intensity = np.clip(-mouth_corner / 0.02, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU15,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au17(self, features: dict[str, float]) -> ActionUnit:
        """AU17: Chin Raiser.

        Detected by raised and pushed up chin (chin boss).
        This is approximated by lower lip position and mouth shape.
        """
        mouth_ratio = features.get("mouth_aspect_ratio", 0.0)

        # Chin raising often accompanies mouth shape changes
        intensity = np.clip((0.15 - mouth_ratio) / 0.1, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU17,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au20(self, features: dict[str, float]) -> ActionUnit:
        """AU20: Lip Stretcher.

        Detected by horizontally stretched lips.
        """
        mouth_width = features.get("mouth_width_ratio", 0.0)

        # Stretched lips have increased width ratio
        intensity = np.clip((mouth_width - 0.45) / 0.15, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU20,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au23(self, features: dict[str, float]) -> ActionUnit:
        """AU23: Lip Tightener.

        Detected by tightened and narrowed lips.
        """
        mouth_width = features.get("mouth_width_ratio", 0.0)

        # Tightened lips have decreased width
        intensity = np.clip((0.42 - mouth_width) / 0.08, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU23,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au25(self, features: dict[str, float]) -> ActionUnit:
        """AU25: Lips Part.

        Detected by separated lips without jaw drop.
        """
        mouth_ratio = features.get("mouth_aspect_ratio", 0.0)

        # Lips parting creates moderate mouth opening
        intensity = np.clip((mouth_ratio - 0.2) / 0.3, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU25,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    def _detect_au26(self, features: dict[str, float]) -> ActionUnit:
        """AU26: Jaw Drop.

        Detected by large mouth opening.
        """
        mouth_ratio = features.get("mouth_aspect_ratio", 0.0)

        # Jaw drop creates large mouth opening
        intensity = np.clip((mouth_ratio - 0.4) / 0.5, 0.0, 1.0)

        return ActionUnit(
            au_type=ActionUnitType.AU26,
            intensity=float(intensity),
            present=intensity >= self.au_threshold
        )

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def _compute_distance(
        self,
        landmarks: FaceLandmarks,
        idx1: FaceLandmarkIndex,
        idx2: FaceLandmarkIndex
    ) -> float:
        """Compute Euclidean distance between two landmarks.

        Args:
            landmarks: Facial landmarks
            idx1: First landmark index
            idx2: Second landmark index

        Returns:
            Euclidean distance between the two points
        """
        p1 = landmarks.get_landmark(idx1)
        p2 = landmarks.get_landmark(idx2)
        return float(np.linalg.norm(p1[:2] - p2[:2]))

    def _compute_angle(
        self,
        landmarks: FaceLandmarks,
        idx1: FaceLandmarkIndex,
        idx2: FaceLandmarkIndex,
        idx3: FaceLandmarkIndex
    ) -> float:
        """Compute angle formed by three landmarks.

        Args:
            landmarks: Facial landmarks
            idx1: First landmark index (starting point)
            idx2: Second landmark index (vertex)
            idx3: Third landmark index (end point)

        Returns:
            Angle in radians at the vertex point
        """
        p1 = landmarks.get_landmark(idx1)[:2]
        p2 = landmarks.get_landmark(idx2)[:2]
        p3 = landmarks.get_landmark(idx3)[:2]

        v1 = p1 - p2
        v2 = p3 - p2

        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))

        return float(angle)
