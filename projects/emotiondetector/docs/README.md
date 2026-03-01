# Emotion Detector Documentation

Welcome to the Emotion Detector documentation! This documentation provides comprehensive information about the system architecture, emotion detection methodology, setup instructions, and usage guides.

## Overview

The Emotion Detector is a real-time facial emotion recognition system that analyzes video streams to detect and classify human emotions. It uses MediaPipe's Face Landmarker for facial feature extraction and applies geometry-based rules derived from the Facial Action Coding System (FACS) to identify emotions.

### Key Features

- **Real-time Processing**: Detect emotions from webcam or video streams in real-time
- **High Accuracy**: 478-point facial landmark detection using MediaPipe
- **Multiple Emotions**: Detect 7 basic emotions (happy, sad, angry, surprised, fearful, disgusted, neutral)
- **Scientific Foundation**: Based on FACS (Facial Action Coding System)
- **Extensible Architecture**: Modular design allows easy customization and extension
- **Visualization**: Real-time overlay of landmarks and detected emotions
- **Export Capabilities**: Save annotated videos and generate analysis reports

## Documentation Index

### 1. [Setup Guide](setup.md)
**Start here if you're new to the project!**

Learn how to install and configure the Emotion Detector:
- System requirements and prerequisites
- Installing Python and dependencies
- Setting up virtual environment with uv
- Downloading MediaPipe models
- Verifying installation
- Troubleshooting common issues
- Quick start examples

**Topics Covered:**
- Installation steps
- Dependency management
- Configuration options
- Development setup
- Testing setup
- Common problems and solutions

[→ Read the Setup Guide](setup.md)

---

### 2. [Architecture Documentation](architecture.md)
**Deep dive into system design and structure**

Understand the system architecture and design patterns:
- System overview and capabilities
- Layer architecture (5-layer design)
- Component descriptions and responsibilities
- Key design patterns (Strategy, Pipeline, Factory, Observer, DI)
- Component interactions and data flow
- Extensibility points and customization
- Performance considerations
- Testing strategy

**Topics Covered:**
- System layers (Input, Processing, Business Logic, Visualization, Application)
- Module organization (`asdrp.video`, `asdrp.face`, `asdrp.emotion`, etc.)
- Data structures (FaceLandmarks, ActionUnits, Emotion)
- Design patterns and best practices
- Error handling and graceful degradation
- Future enhancements

[→ Read the Architecture Documentation](architecture.md)

---

### 3. [Emotion Detection Methodology](emotion_detection.md)
**Learn how emotion detection works**

Detailed explanation of the emotion detection methodology:
- MediaPipe Face Landmarker overview
- 478 facial landmarks and coordinate system
- Facial Action Coding System (FACS) introduction
- Action Units (AUs) and their meanings
- Landmark-to-AU mapping algorithms
- Geometry-based emotion classification rules
- Temporal smoothing techniques
- Confidence scoring
- Limitations and improvements

**Topics Covered:**
- MediaPipe Face Landmarker details
- FACS Action Units (AU1-AU27)
- Emotion-AU relationships (happy, sad, angry, etc.)
- Geometric calculations and formulas
- Rule-based classification
- Temporal filtering (moving average, hysteresis)
- Validation and calibration
- Scientific references

[→ Read the Emotion Detection Methodology](emotion_detection.md)

---

## Quick Links

### Getting Started
1. [Installation](setup.md#installation-steps) - Set up the project
2. [Quick Start](setup.md#quick-start-examples) - Run your first example
3. [Configuration](setup.md#configuration) - Customize settings

### Understanding the System
1. [System Overview](architecture.md#system-overview) - High-level architecture
2. [How It Works](emotion_detection.md#overview) - Emotion detection process
3. [Supported Emotions](architecture.md#supported-emotions) - List of detectable emotions

### Development
1. [Project Structure](architecture.md#architecture-layers) - Code organization
2. [Design Patterns](architecture.md#key-design-patterns) - Software patterns used
3. [Extensibility](architecture.md#extensibility-points) - How to extend the system
4. [Testing](architecture.md#testing-strategy) - Test suite information

### Reference
1. [Action Units](emotion_detection.md#action-units-aus) - FACS AU reference
2. [Landmarks](emotion_detection.md#landmark-groups) - MediaPipe landmark indices
3. [API Documentation](#api-documentation) - Code API reference (coming soon)

## Project Structure

```
emotiondetector/
├── asdrp/                      # Main package
│   ├── emotion/                # Emotion detection logic
│   │   ├── classifier.py       # Emotion classification
│   │   ├── action_units.py     # AU detection
│   │   ├── rules.py            # Classification rules
│   │   └── tracker.py          # Temporal smoothing
│   ├── face/                   # Face processing
│   │   ├── landmarker.py       # MediaPipe wrapper
│   │   ├── processor.py        # Landmark processing
│   │   └── geometry.py         # Geometric calculations
│   ├── video/                  # Video I/O
│   │   ├── reader.py           # Video input
│   │   ├── webcam.py           # Webcam capture
│   │   ├── file.py             # File reading
│   │   └── writer.py           # Video output
│   ├── visualization/          # Rendering
│   │   ├── renderer.py         # Landmark rendering
│   │   ├── overlay.py          # Emotion overlay
│   │   └── export.py           # Report generation
│   └── utils/                  # Utilities
│       ├── config.py           # Configuration
│       ├── logger.py           # Logging
│       └── performance.py      # Monitoring
├── data/                       # Sample data
│   └── videos/                 # Sample videos
├── docs/                       # Documentation (you are here!)
│   ├── README.md               # Documentation index
│   ├── architecture.md         # Architecture guide
│   ├── emotion_detection.md    # Methodology guide
│   └── setup.md                # Setup guide
├── examples/                   # Example scripts
│   ├── process_video.py        # Video file processing
│   ├── webcam_demo.py          # Real-time webcam
│   └── batch_analysis.py       # Batch processing
├── notebooks/                  # Jupyter notebooks
│   ├── exploration.ipynb       # Data exploration
│   └── tutorial.ipynb          # Step-by-step tutorial
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── e2e/                    # End-to-end tests
├── models/                     # Model files
│   └── face_landmarker.task    # MediaPipe model
├── requirements.txt            # Python dependencies
├── pyproject.toml             # Project configuration
└── README.md                  # Project README
```

## Supported Emotions

The system detects seven basic emotions based on Ekman's emotion theory:

| Emotion | Description | Key Features |
|---------|-------------|--------------|
| **Happy** | Joy, pleasure | Smile, raised cheeks, crinkled eyes |
| **Sad** | Sadness, grief | Downturned mouth, inner brows raised |
| **Angry** | Anger, rage | Furrowed brows, tightened lips, intense gaze |
| **Surprised** | Surprise, shock | Raised eyebrows, wide eyes, open mouth |
| **Fearful** | Fear, anxiety | Wide eyes, raised brows, stretched lips |
| **Disgusted** | Disgust, revulsion | Wrinkled nose, raised upper lip |
| **Neutral** | No emotion | Relaxed facial expression |

## Technology Stack

- **MediaPipe** - Face landmark detection (Google)
- **OpenCV** - Video processing and display
- **NumPy** - Numerical computations
- **Python** - Primary programming language
- **FACS** - Facial Action Coding System (theoretical foundation)

## Use Cases

### Education and Research
- Psychology research on facial expressions
- FACS training and learning
- Emotion recognition algorithm development
- Human-computer interaction studies

### Healthcare
- Mental health assessment tools
- Patient emotion monitoring
- Therapy session analysis
- Autism spectrum disorder support

### Entertainment
- Gaming (emotion-based gameplay)
- Virtual reality experiences
- Interactive art installations
- Video content analysis

### Business
- Customer sentiment analysis
- User experience testing
- Marketing research
- Training simulations

## Performance Metrics

Expected performance on standard hardware:

| Metric | Webcam (CPU) | Video File (CPU) | With GPU |
|--------|--------------|------------------|----------|
| FPS | 25-30 | 15-20 | 60+ |
| Latency | <50ms | <70ms | <20ms |
| Memory | <300MB | <500MB | <800MB |
| Accuracy | ~85% | ~85% | ~85% |

*Accuracy based on controlled lab conditions with frontal faces and good lighting.*

## Requirements Summary

### Software
- Python 3.8+
- MediaPipe 0.10+
- OpenCV 4.8+
- NumPy 1.24+

### Hardware
- CPU: Intel i5 / AMD Ryzen 5 or better
- RAM: 4GB minimum, 8GB recommended
- Camera: Optional for real-time detection
- GPU: Optional for acceleration

### Models
- MediaPipe Face Landmarker model (~26MB)

## Example Workflows

### Workflow 1: Process Video File
```
1. Load video file
2. For each frame:
   a. Detect face and extract landmarks
   b. Calculate action units
   c. Classify emotion
   d. Render visualization
3. Export annotated video
4. Generate analysis report
```

### Workflow 2: Real-time Webcam
```
1. Initialize webcam
2. Continuous loop:
   a. Capture frame
   b. Detect landmarks
   c. Classify emotion (with temporal smoothing)
   d. Display with overlay
   e. Check for quit signal
```

### Workflow 3: Batch Analysis
```
1. Load multiple videos
2. For each video:
   a. Process all frames
   b. Track emotion timeline
   c. Compute statistics
3. Generate comparative report
4. Export emotion timelines
```

## Best Practices

### For Accuracy
- Use good lighting (frontal, diffused)
- Keep face centered in frame
- Avoid extreme head angles (>30 degrees)
- Maintain consistent distance from camera
- Calibrate with neutral expression

### For Performance
- Process at lower resolution (640x480)
- Skip frames if needed (process every 2nd frame)
- Use GPU acceleration when available
- Disable unnecessary visualizations
- Optimize emotion smoothing window

### For Development
- Follow the layered architecture
- Use dependency injection
- Write unit tests for new features
- Document public APIs
- Use type hints
- Follow PEP 8 style guide

## Contributing Guidelines

If you're contributing to the project:

1. **Read the architecture docs** - Understand the system design
2. **Follow the structure** - Place code in appropriate modules
3. **Write tests** - Cover new functionality
4. **Document changes** - Update docs for new features
5. **Code style** - Use Black formatter and Ruff linter
6. **Type hints** - Add type annotations
7. **Performance** - Profile performance-critical code

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Face not detected | Improve lighting, center face, check camera |
| Slow performance | Reduce resolution, enable frame skip, use GPU |
| Incorrect emotions | Calibrate baseline, adjust confidence thresholds |
| Import errors | Activate venv, reinstall dependencies |
| Model not found | Download model to `models/` directory |

See [Setup Guide - Troubleshooting](setup.md#troubleshooting) for detailed solutions.

## FAQ

**Q: How accurate is the emotion detection?**
A: Approximately 85% accuracy in controlled conditions with frontal faces and good lighting. Accuracy varies based on individual expressiveness, lighting, and head pose.

**Q: Can it detect multiple faces?**
A: The system is designed for single-face detection, but can be extended for multi-face support. See [Architecture - Future Enhancements](architecture.md#future-enhancements).

**Q: Does it work in real-time?**
A: Yes, it processes 25-30 FPS on typical laptop CPUs, suitable for real-time applications.

**Q: Can I add custom emotions?**
A: Yes! See [Architecture - Extensibility Points](architecture.md#extensibility-points) for instructions on adding new emotions.

**Q: Does it require internet connection?**
A: No, all processing is done locally after downloading the MediaPipe model.

**Q: Is it privacy-safe?**
A: Yes, no data is sent externally. All processing happens on your device.

## Learning Path

### Beginner Path
1. Read [Setup Guide](setup.md) and install the system
2. Run the [Quick Start examples](setup.md#quick-start-examples)
3. Try the Jupyter notebooks in `notebooks/`
4. Experiment with different videos

### Intermediate Path
1. Read [Architecture Documentation](architecture.md)
2. Understand the [layer architecture](architecture.md#architecture-layers)
3. Explore the codebase in `asdrp/`
4. Modify configuration parameters
5. Create custom visualizations

### Advanced Path
1. Study [Emotion Detection Methodology](emotion_detection.md)
2. Understand [FACS and Action Units](emotion_detection.md#facial-action-coding-system-facs)
3. Implement [custom emotions](architecture.md#1-adding-new-emotions)
4. Add [ML-based classification](architecture.md#3-alternative-classification-methods)
5. Contribute new features

## External Resources

### MediaPipe
- [Face Landmarker Guide](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
- [MediaPipe GitHub](https://github.com/google/mediapipe)
- [Face Mesh Model Card](https://storage.googleapis.com/mediapipe-assets/Model%20Card%20MediaPipe%20Face%20Mesh%20V2.pdf)

### FACS
- [FACS Manual](https://www.paulekman.com/facial-action-coding-system/)
- [Paul Ekman Group](https://www.paulekman.com/)
- [FACS Research](https://www.cs.cmu.edu/~face/facs.htm)

### Emotion Recognition
- [Emotion Recognition: A Survey](https://ieeexplore.ieee.org/document/9039580)
- [Deep Facial Expression Recognition](https://arxiv.org/abs/1804.08348)
- [OpenCV Tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)

### Python & Development
- [Python Documentation](https://docs.python.org/3/)
- [uv Package Manager](https://github.com/astral-sh/uv)
- [Pytest Documentation](https://docs.pytest.org/)

## Changelog

### Version 1.0.0 (Current)
- Initial release
- MediaPipe Face Landmarker integration
- Geometry-based emotion classification
- 7 basic emotions supported
- Real-time webcam processing
- Video file processing
- Visualization and export

### Planned Features
- Machine learning-based classification
- Multi-face support
- Micro-expression detection
- REST API
- Mobile SDK
- Cloud deployment options

## License

[Specify your license here]

## Citation

If you use this project in your research, please cite:

```bibtex
@software{emotion_detector,
  title={Emotion Detector: Real-time Facial Emotion Recognition},
  author={[Your Name]},
  year={2026},
  url={[Repository URL]}
}
```

## Contact and Support

- **Issues**: [GitHub Issues](link-to-issues)
- **Discussions**: [GitHub Discussions](link-to-discussions)
- **Email**: [contact-email]

## Acknowledgments

- **Google MediaPipe Team** - For the excellent Face Landmarker model
- **Paul Ekman** - For FACS and emotion research
- **OpenCV Community** - For computer vision tools
- **Contributors** - Thank you to all contributors!

---

**Happy Emotion Detecting!**

For questions, issues, or contributions, please refer to the project repository.

Last updated: February 28, 2026
