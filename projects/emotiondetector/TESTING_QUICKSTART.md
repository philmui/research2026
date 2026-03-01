# Testing Quick Start Guide

## Installation

First, ensure you have the development dependencies installed:

```bash
# Using pip
pip install -e ".[dev]"

# Or using uv
uv pip install -e ".[dev]"
```

This installs:
- pytest
- pytest-cov
- All project dependencies

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_face/test_base.py

# Run specific test class
pytest tests/test_face/test_base.py::TestBoundingBox

# Run specific test
pytest tests/test_face/test_base.py::TestBoundingBox::test_initialization
```

### Coverage Reports

```bash
# Run tests with coverage
pytest --cov=asdrp

# Generate HTML coverage report
pytest --cov=asdrp --cov-report=html
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Show missing lines
pytest --cov=asdrp --cov-report=term-missing

# Coverage for specific module
pytest --cov=asdrp.face tests/test_face/
```

### Test Selection

```bash
# Run only unit tests
pytest -m unit

# Run fast tests (exclude slow ones)
pytest -m "not slow"

# Run tests matching pattern
pytest -k "emotion"
pytest -k "test_initialization"

# Run tests in specific directory
pytest tests/test_emotion/
```

### Debugging Tests

```bash
# Stop on first failure
pytest -x

# Show local variables on failure
pytest -l --tb=long

# Run with print statements visible
pytest -s

# Drop into debugger on failure
pytest --pdb

# Run last failed tests only
pytest --lf

# Run failed tests first, then others
pytest --ff
```

## Quick Test Examples

### Example 1: Test Face Detection Module
```bash
# Run all face detection tests
pytest tests/test_face/ -v

# With coverage
pytest tests/test_face/ --cov=asdrp.face --cov-report=term-missing
```

### Example 2: Test Emotion Analysis
```bash
# Run emotion tests
pytest tests/test_emotion/ -v

# Test specific functionality
pytest tests/test_emotion/test_features.py -v
```

### Example 3: Test Pipeline
```bash
# Run pipeline tests
pytest tests/test_pipeline_comprehensive.py -v

# Run with detailed output
pytest tests/test_pipeline_comprehensive.py -v -s
```

## Checking Test Coverage

### Overall Coverage
```bash
# Generate coverage report
pytest --cov=asdrp --cov-report=term-missing

# Expected output:
# Name                              Stmts   Miss  Cover   Missing
# ---------------------------------------------------------------
# asdrp/__init__.py                    10      0   100%
# asdrp/face/base.py                  120     12    90%   45-48, 89-92
# asdrp/emotion/base.py               150     15    90%   ...
# ---------------------------------------------------------------
# TOTAL                              2500    250    90%
```

### Module-Specific Coverage
```bash
# Face module
pytest tests/test_face/ --cov=asdrp.face --cov-report=term-missing

# Emotion module
pytest tests/test_emotion/ --cov=asdrp.emotion --cov-report=term-missing

# Pipeline
pytest tests/test_pipeline*.py --cov=asdrp.pipeline --cov-report=term-missing
```

## Common Testing Scenarios

### Scenario 1: Before Committing Code
```bash
# Run full test suite with coverage
pytest --cov=asdrp --cov-report=term-missing

# Check if coverage meets threshold (85%)
pytest --cov=asdrp --cov-fail-under=85
```

### Scenario 2: After Adding New Feature
```bash
# Run tests for new feature
pytest tests/test_module/test_new_feature.py -v

# Check coverage of new code
pytest tests/test_module/test_new_feature.py --cov=asdrp.module --cov-report=term-missing
```

### Scenario 3: Debugging Failing Test
```bash
# Run failing test with debugger
pytest tests/test_module/test_feature.py::TestClass::test_method --pdb

# Or with detailed output
pytest tests/test_module/test_feature.py::TestClass::test_method -vv -s
```

### Scenario 4: Quick Sanity Check
```bash
# Run only fast unit tests
pytest -m unit -x

# Run specific module tests
pytest tests/test_face/ -x
```

## Expected Test Results

When all tests pass, you should see:

```
============================= test session starts ==============================
platform darwin -- Python 3.12.x, pytest-7.4.x, pluggy-1.x
rootdir: /path/to/emotiondetector
configfile: pytest.ini
testpaths: tests
plugins: cov-4.1.0
collected 255 items

tests/test_face/test_base.py ......................                      [  8%]
tests/test_face/test_detector.py ..................                     [ 16%]
tests/test_face/test_landmarker.py ....................                 [ 24%]
tests/test_emotion/test_base.py .......................                 [ 33%]
tests/test_emotion/test_features.py .............................       [ 44%]
tests/test_emotion/test_temporal.py ....................                [ 52%]
tests/test_video/test_reader.py .................                       [ 59%]
tests/test_utils/test_config.py ............................            [ 70%]
tests/test_utils/test_geometry.py ...................                   [ 77%]
tests/test_visualization/test_overlay.py ..................             [ 84%]
tests/test_pipeline.py .............                                    [ 89%]
tests/test_pipeline_comprehensive.py ....................               [100%]

============================== 255 passed in 12.34s =============================

---------- coverage: platform darwin, python 3.12.x -----------
Name                                    Stmts   Miss  Cover
-----------------------------------------------------------
asdrp/__init__.py                          10      1    90%
asdrp/face/base.py                        120     10    92%
asdrp/face/detector.py                    150     15    90%
asdrp/emotion/base.py                     130     12    91%
asdrp/emotion/features.py                 200     20    90%
asdrp/emotion/geometry_analyzer.py        180     18    90%
asdrp/emotion/temporal.py                  80      8    90%
asdrp/pipeline.py                         250     25    90%
asdrp/utils/config.py                     100     10    90%
asdrp/utils/geometry.py                    60      6    90%
asdrp/video/reader.py                     100     10    90%
asdrp/visualization/overlay.py            120     15    88%
-----------------------------------------------------------
TOTAL                                    1500    150    90%
```

## Troubleshooting

### Issue: Import Errors
```bash
# Solution: Install package in development mode
pip install -e .
```

### Issue: Missing Fixtures
```bash
# Solution: Check conftest.py is in tests/ directory
ls tests/conftest.py
```

### Issue: Tests Not Found
```bash
# Solution: Verify pytest discovery settings
pytest --collect-only
```

### Issue: Mocking Errors
```bash
# Solution: Check mock patch paths match import paths
# Correct: @patch('asdrp.face.detector.MediaPipeFaceDetector')
# Wrong: @patch('MediaPipeFaceDetector')
```

### Issue: Slow Tests
```bash
# Solution: Run without slow tests
pytest -m "not slow"

# Or increase timeout
pytest --timeout=300
```

## Development Workflow

### 1. Write Test First (TDD)
```bash
# Create test file
touch tests/test_module/test_new_feature.py

# Write failing test
# Run test (should fail)
pytest tests/test_module/test_new_feature.py -v

# Implement feature
# Run test (should pass)
pytest tests/test_module/test_new_feature.py -v
```

### 2. Check Coverage
```bash
# Run with coverage
pytest tests/test_module/test_new_feature.py --cov=asdrp.module --cov-report=term-missing

# Aim for >85% coverage
```

### 3. Run Full Suite
```bash
# Before committing
pytest --cov=asdrp --cov-fail-under=85
```

### 4. Commit
```bash
git add tests/test_module/test_new_feature.py
git add asdrp/module/new_feature.py
git commit -m "Add new feature with tests"
```

## CI/CD Integration

### GitHub Actions
Tests run automatically on push/PR:
```yaml
- name: Run tests
  run: pytest --cov=asdrp --cov-report=xml
```

### Local Pre-commit
```bash
# Install pre-commit hook
pre-commit install

# Run manually
pre-commit run --all-files
```

## Next Steps

1. **Explore Tests**: Look at existing tests for examples
2. **Run Tests**: Try the commands above
3. **Check Coverage**: Generate HTML report
4. **Write Tests**: Add tests for new features
5. **Read Docs**: See [tests/README.md](tests/README.md) for details

## Quick Reference Card

```bash
# Most Common Commands
pytest                                    # Run all tests
pytest -v                                # Verbose output
pytest -x                                # Stop on first failure
pytest --cov=asdrp                      # With coverage
pytest --cov=asdrp --cov-report=html    # HTML coverage report
pytest -m "not slow"                     # Skip slow tests
pytest -k "pattern"                      # Run tests matching pattern
pytest tests/test_face/                  # Specific directory
pytest --lf                              # Last failed
pytest --pdb                             # Drop to debugger on fail
```

## Resources

- [Test Suite Summary](TEST_SUITE_SUMMARY.md)
- [Tests README](tests/README.md)
- [Pytest Documentation](https://docs.pytest.org/)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
