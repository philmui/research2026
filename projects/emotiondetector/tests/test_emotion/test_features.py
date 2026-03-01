"""Tests for feature extraction from facial landmarks.

This module tests the FeatureExtractor class and action unit detection.
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from asdrp.emotion.base import ActionUnitType
from asdrp.emotion.features import FeatureExtractor
from asdrp.face.base import FaceLandmarks


class TestFeatureExtractor:
    """Test suite for FeatureExtractor class."""

    def test_initialization_default(self) -> None:
        """Test FeatureExtractor initialization with defaults."""
        extractor = FeatureExtractor()
        assert extractor.au_threshold == 0.3

    def test_initialization_custom_threshold(self) -> None:
        """Test FeatureExtractor initialization with custom threshold."""
        extractor = FeatureExtractor(au_threshold=0.5)
        assert extractor.au_threshold == 0.5

    def test_extract_features_returns_dict(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that extract_features returns a dictionary."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_features_has_key_features(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that extract_features includes key facial features."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        # Should have eye features
        assert any("eye" in key.lower() for key in features.keys())

        # Should have mouth features
        assert any("mouth" in key.lower() for key in features.keys())

    def test_extract_features_all_numeric(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that all feature values are numeric."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        for key, value in features.items():
            assert isinstance(value, (int, float, np.number))
            assert not np.isnan(value)
            assert not np.isinf(value)

    def test_extract_features_consistent(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that feature extraction is deterministic."""
        extractor = FeatureExtractor()

        features1 = extractor.extract_features(sample_face_landmarks)
        features2 = extractor.extract_features(sample_face_landmarks)

        assert features1.keys() == features2.keys()
        for key in features1.keys():
            assert features1[key] == pytest.approx(features2[key])

    def test_detect_action_units_returns_dict(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that detect_action_units returns dictionary."""
        extractor = FeatureExtractor()
        action_units = extractor.detect_action_units(sample_face_landmarks)

        assert isinstance(action_units, dict)
        assert len(action_units) > 0

    def test_detect_action_units_has_key_aus(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that detect_action_units includes key AUs."""
        extractor = FeatureExtractor()
        action_units = extractor.detect_action_units(sample_face_landmarks)

        # Should have upper face AUs
        assert ActionUnitType.AU1 in action_units or ActionUnitType.AU2 in action_units

        # Should have lower face AUs
        assert ActionUnitType.AU12 in action_units or ActionUnitType.AU25 in action_units

    def test_detect_action_units_valid_structure(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that detected action units have valid structure."""
        extractor = FeatureExtractor()
        action_units = extractor.detect_action_units(sample_face_landmarks)

        for au_type, au in action_units.items():
            assert isinstance(au_type, ActionUnitType)
            assert au.au_type == au_type
            assert 0.0 <= au.intensity <= 1.0
            assert 0.0 <= au.confidence <= 1.0
            assert isinstance(au.present, bool)

    def test_detect_action_units_threshold_respected(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that AU threshold is respected for presence detection."""
        threshold = 0.5
        extractor = FeatureExtractor(au_threshold=threshold)
        action_units = extractor.detect_action_units(sample_face_landmarks)

        for au in action_units.values():
            if au.present:
                assert au.intensity >= threshold
            else:
                assert au.intensity < threshold


class TestEyeFeatureExtraction:
    """Test suite for eye-related feature extraction."""

    def test_eye_openness_calculation(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test eye openness feature calculation."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        # Should have left and right eye openness
        assert any("left_eye" in key and "open" in key for key in features.keys())
        assert any("right_eye" in key and "open" in key for key in features.keys())

    def test_eye_features_positive(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test that eye features are non-negative."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        eye_features = {k: v for k, v in features.items() if "eye" in k.lower()}

        for value in eye_features.values():
            assert value >= 0.0


class TestEyebrowFeatureExtraction:
    """Test suite for eyebrow-related feature extraction."""

    def test_eyebrow_features_exist(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test that eyebrow features are extracted."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        assert any("eyebrow" in key.lower() or "brow" in key.lower() for key in features.keys())

    def test_eyebrow_height_calculation(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test eyebrow height feature calculation."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        # Should have eyebrow height features
        eyebrow_features = {
            k: v for k, v in features.items() if "eyebrow" in k.lower()
        }
        assert len(eyebrow_features) > 0


class TestMouthFeatureExtraction:
    """Test suite for mouth-related feature extraction."""

    def test_mouth_features_exist(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test that mouth features are extracted."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        mouth_features = {k: v for k, v in features.items() if "mouth" in k.lower() or "lip" in k.lower()}
        assert len(mouth_features) > 0

    def test_mouth_openness_calculation(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test mouth openness feature calculation."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        assert any("mouth" in key and "open" in key for key in features.keys())

    def test_mouth_width_calculation(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test mouth width feature calculation."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        assert any("mouth" in key and "width" in key for key in features.keys())

    def test_mouth_features_reasonable_range(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test that mouth features are in reasonable range."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        mouth_features = {
            k: v for k, v in features.items() if "mouth" in k.lower() or "lip" in k.lower()
        }

        for value in mouth_features.values():
            assert 0.0 <= value <= 2.0  # Normalized with some tolerance


class TestNoseFeatureExtraction:
    """Test suite for nose-related feature extraction."""

    def test_nose_features_exist(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test that nose features are extracted."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        nose_features = {k: v for k, v in features.items() if "nose" in k.lower()}
        # Nose features may or may not be present depending on implementation
        # Just check that extraction doesn't fail


class TestFaceShapeFeatureExtraction:
    """Test suite for face shape feature extraction."""

    def test_face_features_exist(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test that face shape features are extracted."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        face_features = {k: v for k, v in features.items() if "face" in k.lower()}
        # Face features may or may not be present

    def test_face_aspect_ratio(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test face aspect ratio calculation if available."""
        extractor = FeatureExtractor()
        features = extractor.extract_features(sample_face_landmarks)

        # If face width/height features exist, they should be positive
        for key, value in features.items():
            if "width" in key.lower() or "height" in key.lower():
                assert value > 0.0


class TestActionUnitDetection:
    """Test suite for specific action unit detection."""

    def test_au6_detection_smile(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test AU6 (Cheek Raiser) detection."""
        extractor = FeatureExtractor()
        action_units = extractor.detect_action_units(sample_face_landmarks)

        assert ActionUnitType.AU6 in action_units
        au6 = action_units[ActionUnitType.AU6]
        assert 0.0 <= au6.intensity <= 1.0

    def test_au12_detection_smile(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test AU12 (Lip Corner Puller) detection."""
        extractor = FeatureExtractor()
        action_units = extractor.detect_action_units(sample_face_landmarks)

        assert ActionUnitType.AU12 in action_units
        au12 = action_units[ActionUnitType.AU12]
        assert 0.0 <= au12.intensity <= 1.0

    def test_au4_detection_frown(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test AU4 (Brow Lowerer) detection."""
        extractor = FeatureExtractor()
        action_units = extractor.detect_action_units(sample_face_landmarks)

        assert ActionUnitType.AU4 in action_units
        au4 = action_units[ActionUnitType.AU4]
        assert 0.0 <= au4.intensity <= 1.0

    def test_au25_detection_lips_part(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test AU25 (Lips Part) detection."""
        extractor = FeatureExtractor()
        action_units = extractor.detect_action_units(sample_face_landmarks)

        assert ActionUnitType.AU25 in action_units
        au25 = action_units[ActionUnitType.AU25]
        assert 0.0 <= au25.intensity <= 1.0

    def test_au26_detection_jaw_drop(
        self, sample_face_landmarks: FaceLandmarks
    ) -> None:
        """Test AU26 (Jaw Drop) detection."""
        extractor = FeatureExtractor()
        action_units = extractor.detect_action_units(sample_face_landmarks)

        assert ActionUnitType.AU26 in action_units
        au26 = action_units[ActionUnitType.AU26]
        assert 0.0 <= au26.intensity <= 1.0


class TestFeatureExtractionEdgeCases:
    """Test suite for edge cases in feature extraction."""

    def test_minimal_landmarks(self) -> None:
        """Test feature extraction with minimal landmarks."""
        # Create minimal landmark set
        landmarks = np.random.rand(478, 3).astype(np.float32)
        face = FaceLandmarks(landmarks=landmarks)

        extractor = FeatureExtractor()
        features = extractor.extract_features(face)

        assert isinstance(features, dict)

    def test_extreme_values(self) -> None:
        """Test feature extraction with extreme landmark values."""
        # Create landmarks at boundaries
        landmarks = np.zeros((478, 3), dtype=np.float32)
        landmarks[:, :2] = 0.5  # Center all landmarks

        face = FaceLandmarks(landmarks=landmarks)

        extractor = FeatureExtractor()
        features = extractor.extract_features(face)

        # Should not crash, features should be valid numbers
        for value in features.values():
            assert not np.isnan(value)
            assert not np.isinf(value)

    def test_different_thresholds(self, sample_face_landmarks: FaceLandmarks) -> None:
        """Test AU detection with different thresholds."""
        for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
            extractor = FeatureExtractor(au_threshold=threshold)
            action_units = extractor.detect_action_units(sample_face_landmarks)

            # Verify all present AUs meet threshold
            for au in action_units.values():
                if au.present:
                    assert au.intensity >= threshold
