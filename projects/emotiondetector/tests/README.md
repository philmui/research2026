# Emotion Detector Test Suite

This directory contains comprehensive tests for the emotion detection project using pytest.

## Test Structure

```
tests/
├── conftest.py                      # Shared fixtures and test utilities
├── pytest.ini                       # Pytest configuration (in project root)
├── test_face/                       # Tests for face detection module
│   ├── __init__.py
│   ├── test_base.py                # Tests for BoundingBox, FaceLandmarks
│   ├── test_detector.py            # Tests for MediaPipeFaceDetector
│   └── test_landmarker.py          # Tests for landmark utilities
├── test_emotion/                    # Tests for emotion analysis module
│   ├── __init__.py
│   ├── test_base.py                # Tests for EmotionType, EmotionPrediction
│   ├── test_features.py            # Tests for FeatureExtractor
│   └── test_temporal.py            # Tests for temporal smoothing
├── test_video/                      # Tests for video processing module
│   ├── __init__.py
│   └── test_reader.py              # Tests for VideoFileReader
├── test_utils/                      # Tests for utility modules
│   ├── __init__.py
│   └── test_config.py              # Tests for configuration classes
├── test_visualization/              # Tests for visualization module
│   └── __init__.py
├── test_pipeline.py                 # Basic pipeline tests
└── test_pipeline_comprehensive.py  # Comprehensive end-to-end tests
```

## Running Tests

### Run all tests
```bash
pytest
```

### Run with coverage report
```bash
pytest --cov=asdrp --cov-report=html
```

### Run specific test file
```bash
pytest tests/test_face/test_base.py
```

### Run specific test class
```bash
pytest tests/test_face/test_base.py::TestBoundingBox
```

### Run specific test function
```bash
pytest tests/test_face/test_base.py::TestBoundingBox::test_initialization
```

### Run tests with verbose output
```bash
pytest -v
```

### Run tests matching a pattern
```bash
pytest -k "emotion"  # Run all tests with "emotion" in name
```

### Run tests by marker
```bash
pytest -m unit       # Run only unit tests
pytest -m integration  # Run only integration tests
```

## Test Markers

Tests can be marked with the following markers (defined in pytest.ini):

- `unit`: Unit tests for individual functions/classes
- `integration`: Integration tests for module interactions
- `e2e`: End-to-end tests for complete workflows
- `slow`: Tests that take a long time to run
- `requires_model`: Tests requiring MediaPipe model file
- `requires_video`: Tests requiring video files
- `requires_camera`: Tests requiring camera access
- `visual`: Tests producing visual output

### Using markers
```python
@pytest.mark.unit
def test_something():
    pass

@pytest.mark.slow
@pytest.mark.requires_model
def test_with_real_model():
    pass
```

## Coverage Goals

The test suite aims for:
- **Overall coverage**: ≥85%
- **Critical modules**: ≥90% (face detection, emotion analysis, pipeline)
- **Utility modules**: ≥80%

View coverage report:
```bash
pytest --cov=asdrp --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `sample_face_image`: Synthetic face image for testing
- `sample_rgb_image`: Simple RGB test image
- `sample_face_landmarks`: Mock FaceLandmarks object
- `sample_bounding_box`: Sample BoundingBox
- `mock_face_detector`: Mock face detector
- `sample_emotion_predictions`: List of sample predictions
- `sample_action_units`: Sample action units
- `temp_video_file`: Temporary test video file
- `temp_output_dir`: Temporary output directory
- `mock_emotion_analyzer`: Mock emotion analyzer
- `sample_features`: Sample feature dictionary
- `sample_emotion_probabilities`: Sample probability distribution

## Writing New Tests

### Test Naming Conventions

- Test files: `test_*.py`
- Test classes: `Test*`
- Test functions: `test_*`

### Example Test Structure

```python
"""Module docstring describing what is being tested."""

import pytest
from asdrp.module import ClassToTest


class TestClassName:
    """Test suite for ClassName."""

    def test_feature_description(self, fixture_name):
        \"\"\"Test docstring explaining what is tested.\"\"\"
        # Arrange
        obj = ClassToTest()

        # Act
        result = obj.method()

        # Assert
        assert result == expected_value

    def test_error_handling(self):
        \"\"\"Test that errors are properly raised.\"\"\"
        with pytest.raises(ValueError, match="error message"):
            ClassToTest().invalid_operation()
```

## Mocking Strategy

Tests use mocking to:
1. Avoid requiring actual MediaPipe model files
2. Avoid requiring video files or camera access
3. Speed up test execution
4. Isolate units under test

Example:
```python
from unittest.mock import Mock, patch

@patch("module.MediaPipeFaceDetector")
def test_with_mock(mock_detector):
    mock_detector.detect.return_value = []
    # Test code here
```

## Continuous Integration

Tests are designed to run in CI/CD environments without requiring:
- Physical camera hardware
- Large model files
- Video files
- GUI display

## Troubleshooting

### Tests fail with "Model file not found"
- Tests should use mocking to avoid needing real model files
- Check that `@patch` decorators are properly applied

### Tests fail with "Video file not found"
- Use the `temp_video_file` fixture for tests requiring video
- Or use mocking for VideoFileReader

### Coverage is lower than expected
- Ensure all code paths are tested
- Add tests for edge cases and error handling
- Check for untested private methods

### Tests are slow
- Use mocking instead of real processing
- Mark slow tests with `@pytest.mark.slow`
- Run fast tests with: `pytest -m "not slow"`

## Contributing

When adding new features:
1. Write tests first (TDD approach)
2. Ensure ≥85% coverage for new code
3. Include docstrings for all test functions
4. Test edge cases and error conditions
5. Use appropriate fixtures from conftest.py
6. Add new fixtures to conftest.py if needed

## Resources

- [Pytest Documentation](https://docs.pytest.org/)
- [Pytest-cov Documentation](https://pytest-cov.readthedocs.io/)
- [Python unittest.mock](https://docs.python.org/3/library/unittest.mock.html)
