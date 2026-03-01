# Setup Guide

This guide will walk you through setting up the Emotion Detector project on your system.

## System Requirements

### Minimum Requirements
- **OS**: macOS 10.14+, Ubuntu 18.04+, Windows 10+
- **Python**: 3.8 or higher (3.10+ recommended)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB for dependencies and models
- **Camera**: Optional, for real-time webcam detection

### Recommended Requirements
- **CPU**: Intel i5/AMD Ryzen 5 or better (for real-time processing)
- **GPU**: Optional, CUDA-capable GPU for faster processing
- **RAM**: 8GB or more
- **Camera**: 720p webcam or better

## Prerequisites

Before starting, ensure you have the following installed:

### 1. Python 3.8+

Check your Python version:

```bash
python3 --version
```

If Python is not installed or the version is too old:

**macOS:**
```bash
# Using Homebrew
brew install python@3.11
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

**Windows:**
Download from [python.org](https://www.python.org/downloads/) or use:
```powershell
winget install Python.Python.3.11
```

### 2. Git

Check if Git is installed:

```bash
git --version
```

If not installed:

**macOS:**
```bash
brew install git
```

**Ubuntu/Debian:**
```bash
sudo apt install git
```

**Windows:**
Download from [git-scm.com](https://git-scm.com/downloads)

## Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the project
git clone <repository-url>
cd emotiondetector

# Or if you already have the project
cd /path/to/emotiondetector
```

### Step 2: Install uv (Recommended Package Manager)

[uv](https://github.com/astral-sh/uv) is a fast, modern Python package installer and resolver. It's significantly faster than pip.

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

Verify installation:
```bash
uv --version
```

### Step 3: Create Virtual Environment

Creating a virtual environment isolates project dependencies from your system Python.

**Using uv (Recommended):**
```bash
# Create virtual environment
uv venv

# Activate the environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

**Using standard Python venv:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows
```

Your prompt should now show `(.venv)` indicating the virtual environment is active.

### Step 4: Install Dependencies

#### Option A: Using uv (Recommended)

If you have a `pyproject.toml`:
```bash
uv pip install -e .
```

If you have a `requirements.txt`:
```bash
uv pip install -r requirements.txt
```

#### Option B: Using pip

```bash
pip install -r requirements.txt
```

#### Core Dependencies

The project requires:

```txt
# Core dependencies
mediapipe>=0.10.0          # Face landmark detection
opencv-python>=4.8.0       # Video processing
numpy>=1.24.0              # Numerical operations
pillow>=10.0.0             # Image handling

# Optional: GPU acceleration
# mediapipe-gpu>=0.10.0    # Uncomment if you have CUDA GPU

# Development dependencies
pytest>=7.4.0              # Testing
pytest-cov>=4.1.0          # Test coverage
black>=23.0.0              # Code formatting
ruff>=0.1.0                # Linting
mypy>=1.5.0                # Type checking

# Notebook support (optional)
jupyter>=1.0.0
ipykernel>=6.25.0
matplotlib>=3.7.0          # Visualization
```

### Step 5: Download MediaPipe Model

MediaPipe Face Landmarker requires a pre-trained model file.

#### Automatic Download (Recommended)

Create a script to download the model:

```bash
# Create models directory
mkdir -p models

# Download the model
curl -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

#### Manual Download

1. Visit: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
2. Download `face_landmarker.task` (or `face_landmarker_v2.task`)
3. Place in `models/` directory:

```
emotiondetector/
├── models/
│   └── face_landmarker.task
├── asdrp/
├── data/
└── ...
```

#### Verify Model Download

```bash
# Check if model exists
ls -lh models/face_landmarker.task

# Should show file size around 22-28 MB
```

### Step 6: Verify Installation

Run the verification script to ensure everything is set up correctly:

```python
# test_installation.py
import sys
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

def test_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    assert version >= (3, 8), "Python 3.8+ required"
    print("✓ Python version OK")

def test_opencv():
    """Check OpenCV installation."""
    print(f"OpenCV version: {cv2.__version__}")
    # Test basic functionality
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    assert img.shape == (100, 100, 3)
    print("✓ OpenCV OK")

def test_mediapipe():
    """Check MediaPipe installation."""
    print(f"MediaPipe version: {mp.__version__}")
    # Check if face landmarker is available
    assert hasattr(mp.solutions, 'face_mesh')
    print("✓ MediaPipe OK")

def test_model_file():
    """Check if model file exists."""
    model_path = Path("models/face_landmarker.task")
    assert model_path.exists(), f"Model not found at {model_path}"
    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f"Model file size: {size_mb:.1f} MB")
    assert size_mb > 10, "Model file seems too small"
    print("✓ Model file OK")

def test_webcam():
    """Test webcam access (optional)."""
    try:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"✓ Webcam accessible ({frame.shape[1]}x{frame.shape[0]})")
            else:
                print("⚠ Webcam found but couldn't read frame")
        else:
            print("⚠ No webcam detected (optional)")
    except Exception as e:
        print(f"⚠ Webcam test failed: {e} (optional)")

if __name__ == "__main__":
    print("=" * 50)
    print("Emotion Detector Installation Verification")
    print("=" * 50)

    try:
        test_python_version()
        test_opencv()
        test_mediapipe()
        test_model_file()
        test_webcam()

        print("\n" + "=" * 50)
        print("✓ All required components verified!")
        print("=" * 50)
        print("\nYou're ready to use the Emotion Detector!")

    except AssertionError as e:
        print(f"\n✗ Verification failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
```

Run the verification:

```bash
python test_installation.py
```

Expected output:
```
==================================================
Emotion Detector Installation Verification
==================================================
Python version: 3.11.5
✓ Python version OK
OpenCV version: 4.8.1
✓ OpenCV OK
MediaPipe version: 0.10.7
✓ MediaPipe OK
Model file size: 26.3 MB
✓ Model file OK
✓ Webcam accessible (1280x720)

==================================================
✓ All required components verified!
==================================================

You're ready to use the Emotion Detector!
```

## Project Structure

After setup, your project should look like:

```
emotiondetector/
├── .venv/                    # Virtual environment (created)
├── models/                   # Model files (downloaded)
│   └── face_landmarker.task
├── asdrp/                    # Main package
│   ├── __init__.py
│   ├── emotion/              # Emotion detection logic
│   ├── face/                 # Face landmark processing
│   ├── video/                # Video input/output
│   ├── visualization/        # Display and rendering
│   └── utils/                # Utilities
├── data/                     # Sample data
│   ├── videos/
│   └── sample_face_video.mp4
├── examples/                 # Example scripts
├── notebooks/                # Jupyter notebooks
├── tests/                    # Test suite
├── docs/                     # Documentation
├── requirements.txt          # Dependencies
├── pyproject.toml           # Project configuration
└── README.md                # Project overview
```

## Configuration

### Environment Variables

Create a `.env` file for configuration (optional):

```bash
# .env
MEDIAPIPE_MODEL_PATH=models/face_landmarker.task
MIN_DETECTION_CONFIDENCE=0.5
MIN_TRACKING_CONFIDENCE=0.5
VIDEO_OUTPUT_DIR=output/
LOG_LEVEL=INFO
```

### Configuration File

Create `config.yaml` (optional):

```yaml
# config.yaml
detector:
  model_path: models/face_landmarker.task
  min_detection_confidence: 0.5
  min_tracking_confidence: 0.5
  max_num_faces: 1

emotion:
  smoothing_window: 5
  confidence_threshold: 0.3
  enable_secondary_emotion: true

video:
  default_fps: 30
  max_frame_width: 1280
  enable_frame_skip: false

visualization:
  show_landmarks: true
  show_confidence: true
  overlay_opacity: 0.7
  font_scale: 0.8
```

## Quick Start Examples

### Example 1: Process Video File

```python
# examples/process_video.py
from asdrp.video import FileReader
from asdrp.face import FaceLandmarker
from asdrp.emotion import EmotionClassifier
from asdrp.visualization import Visualizer

# Initialize components
video = FileReader("data/sample_face_video.mp4")
landmarker = FaceLandmarker("models/face_landmarker.task")
classifier = EmotionClassifier()
visualizer = Visualizer()

# Process video
for frame in video:
    landmarks = landmarker.detect(frame)
    if landmarks:
        emotion = classifier.classify(landmarks)
        annotated = visualizer.render(frame, landmarks, emotion)
        visualizer.show(annotated)

    if visualizer.should_quit():
        break

video.release()
```

Run it:
```bash
python examples/process_video.py
```

### Example 2: Real-time Webcam

```python
# examples/webcam_demo.py
from asdrp.video import WebcamReader
from asdrp.face import FaceLandmarker
from asdrp.emotion import EmotionClassifier
from asdrp.visualization import Visualizer

# Initialize
webcam = WebcamReader(0)  # 0 = default webcam
landmarker = FaceLandmarker("models/face_landmarker.task")
classifier = EmotionClassifier()
visualizer = Visualizer()

print("Press 'q' to quit")

# Real-time loop
while True:
    frame = webcam.read()
    if frame is None:
        break

    landmarks = landmarker.detect(frame)
    if landmarks:
        emotion = classifier.classify(landmarks)
        frame = visualizer.render(frame, landmarks, emotion)

    visualizer.show(frame)

    if visualizer.should_quit():
        break

webcam.release()
```

Run it:
```bash
python examples/webcam_demo.py
```

## Troubleshooting

### Issue: ModuleNotFoundError

```
ModuleNotFoundError: No module named 'mediapipe'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Reinstall dependencies
uv pip install -r requirements.txt
```

### Issue: Model File Not Found

```
FileNotFoundError: models/face_landmarker.task not found
```

**Solution:**
```bash
# Re-download the model
curl -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

### Issue: Webcam Not Accessible

```
Error: Cannot open webcam
```

**Solutions:**
1. Check camera permissions (System Preferences on macOS)
2. Ensure no other application is using the camera
3. Try different camera index: `WebcamReader(1)` instead of `WebcamReader(0)`
4. Check camera connection: `ls /dev/video*` (Linux) or System Info (macOS/Windows)

### Issue: Slow Performance

**Solutions:**
1. Reduce video resolution:
   ```python
   webcam = WebcamReader(0, width=640, height=480)
   ```

2. Enable frame skipping:
   ```python
   video.set_frame_skip(2)  # Process every 2nd frame
   ```

3. Disable landmark visualization:
   ```python
   visualizer = Visualizer(show_landmarks=False)
   ```

4. Use GPU acceleration (if available):
   ```bash
   uv pip install mediapipe-gpu
   ```

### Issue: Import Errors in Custom Code

```
ImportError: cannot import name 'FaceLandmarker'
```

**Solution:**
Ensure package is installed in development mode:
```bash
uv pip install -e .
```

### Issue: OpenCV Window Not Responding

**Solution:**
Add proper waitKey call:
```python
cv2.imshow("Emotion Detector", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```

## Development Setup

For contributors and developers:

### Install Development Dependencies

```bash
uv pip install -e ".[dev]"
```

### Pre-commit Hooks

```bash
# Install pre-commit
uv pip install pre-commit

# Set up git hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# With coverage
pytest --cov=asdrp --cov-report=html

# Run specific test suite
pytest tests/unit/
pytest tests/integration/
pytest tests/e2e/
```

### Code Formatting

```bash
# Format code
black asdrp/ tests/ examples/

# Check formatting
black --check asdrp/

# Lint
ruff check asdrp/

# Type checking
mypy asdrp/
```

## Updating Dependencies

### Update All Packages

```bash
# Using uv
uv pip install --upgrade -r requirements.txt

# Or update specific package
uv pip install --upgrade mediapipe
```

### Check for Outdated Packages

```bash
pip list --outdated
```

## Uninstallation

To completely remove the project:

```bash
# Deactivate virtual environment
deactivate

# Remove virtual environment
rm -rf .venv/

# Remove cached files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete

# Optionally remove the entire project
cd ..
rm -rf emotiondetector/
```

## Next Steps

Now that you've set up the Emotion Detector:

1. **Try the examples**: Run the example scripts in `examples/`
2. **Read the architecture**: See `docs/architecture.md` for system design
3. **Learn the methodology**: Read `docs/emotion_detection.md` for how emotions are detected
4. **Explore notebooks**: Open Jupyter notebooks in `notebooks/`
5. **Build something**: Create your own emotion detection application!

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review the [documentation](README.md) in `docs/`
3. Open an issue on GitHub (if applicable)
4. Check MediaPipe documentation: https://developers.google.com/mediapipe

## Additional Resources

- **MediaPipe Face Landmarker**: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- **OpenCV Documentation**: https://docs.opencv.org/
- **Python Virtual Environments**: https://docs.python.org/3/tutorial/venv.html
- **uv Package Manager**: https://github.com/astral-sh/uv
- **FACS Manual**: https://www.paulekman.com/facial-action-coding-system/

---

**Congratulations!** You've successfully set up the Emotion Detector project. Happy coding!
