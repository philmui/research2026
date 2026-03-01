# ASDRP Emotion Detection from Video

A comprehensive emotion detection system that analyzes facial expressions in videos using computer vision and deep learning.

## Overview

This project implements a complete pipeline for detecting and analyzing emotions from video content. It uses facial landmark detection, expression analysis, and machine learning to identify and track emotions throughout video sequences.

## Features

- Real-time facial detection and landmark extraction
- Multi-emotion classification (happiness, sadness, anger, surprise, fear, disgust, neutral)
- Video processing pipeline with frame extraction
- Temporal emotion tracking and analysis
- Interactive visualizations and output generation
- Support for multiple video formats

## Project Structure

```
emotiondetector/
├── asdrp/                    # Main package
│   ├── face/                 # Face detection and landmarks
│   ├── emotion/              # Emotion classification
│   ├── video/                # Video processing
│   ├── visualization/        # Plotting and visualization
│   └── utils/                # Helper utilities
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   ├── integration/          # Integration tests
│   └── e2e/                  # End-to-end tests
├── docs/                     # Documentation
├── data/                     # Data directory
│   └── videos/               # Input videos
├── examples/                 # Example scripts
└── pyproject.toml            # Project configuration
```

## Installation

This project uses `uv` for fast, reliable Python package management.

### Prerequisites

- Python 3.12+
- uv package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd emotiondetector
```

2. Create and activate virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
uv pip install -e ".[dev,notebooks]"
```

## Quick Start

```python
from asdrp.video import VideoProcessor
from asdrp.face import FaceDetector
from asdrp.emotion import EmotionClassifier

# Initialize components
video_processor = VideoProcessor("path/to/video.mp4")
face_detector = FaceDetector()
emotion_classifier = EmotionClassifier()

# Process video
for frame in video_processor.frames():
    faces = face_detector.detect(frame)
    for face in faces:
        emotion = emotion_classifier.predict(face)
        print(f"Detected emotion: {emotion}")
```

## Development

### Running Tests

```bash
pytest
```

### Code Formatting

```bash
black asdrp tests
ruff check asdrp tests
```

### Type Checking

```bash
mypy asdrp
```

## Dependencies

### Core Dependencies
- PyTorch: Deep learning framework
- OpenCV: Computer vision operations
- MediaPipe: Face detection and landmarks
- NumPy/Pandas: Data processing

### Visualization
- Matplotlib, Seaborn, Plotly: Data visualization

### Development
- pytest: Testing framework
- black: Code formatting
- ruff: Linting
- mypy: Type checking

## License

[Add license information]

## Contributing

[Add contribution guidelines]

## Acknowledgments

ASDRP (Aspiring Scholars Directed Research Program)
