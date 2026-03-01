# Emotion Detector Jupyter Notebooks

This directory contains comprehensive Jupyter notebooks demonstrating the ASDRP emotion detection library. These notebooks provide interactive tutorials for face detection, emotion analysis, and temporal emotion patterns.

## Overview

The notebooks are designed to be run sequentially, building from basic face detection to advanced temporal analysis:

1. **01_face_detection_demo.ipynb** - Introduction to face detection with MediaPipe
2. **02_emotion_analysis_demo.ipynb** - Emotion detection and analysis
3. **03_temporal_analysis.ipynb** - Advanced temporal patterns and microexpressions

## Prerequisites

### 1. Software Requirements

- Python 3.12 or higher
- Jupyter Notebook or JupyterLab
- All dependencies from the main project (install via `uv` or `pip`)

### 2. Required Files

#### Video File
The notebooks use a sample video for demonstrations. Ensure you have:
- **Path**: `../data/videos/youtube_short_emotion.mp4`
- **Description**: A video containing facial expressions showing various emotions

If you don't have this file, you can:
- Use your own video by modifying the `video_path` variable in the notebooks
- Download a sample video showing various emotions from YouTube or other sources

#### MediaPipe Model
You need the MediaPipe Face Landmarker model:

1. **Download**: Visit [MediaPipe Face Landmarker](https://developers.google.com/mediapipe/solutions/vision/face_landmarker)
2. **File**: Download `face_landmarker_v2_with_blendshapes.task`
3. **Location**: Place it in `../models/face_landmarker.task`
4. **Size**: Approximately 25-30 MB

The notebooks will create the `models/` directory if it doesn't exist and provide instructions if the model is missing.

## Installation

### Using uv (recommended)

```bash
# From the project root directory
cd /path/to/emotiondetector

# Install dependencies (if not already done)
uv sync

# Activate the virtual environment
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows

# Start Jupyter
jupyter notebook notebooks/
```

### Using pip

```bash
# From the project root directory
cd /path/to/emotiondetector

# Install the package in development mode
pip install -e .

# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter
jupyter notebook notebooks/
```

## Notebook Descriptions

### 01_face_detection_demo.ipynb

**Focus**: Introduction to MediaPipe face detection and landmark extraction

**Topics Covered**:
- Loading and displaying video files
- Initializing the MediaPipe Face Detector
- Detecting faces and extracting 478 3D facial landmarks
- Visualizing landmark groups:
  - Eyes (left and right)
  - Eyebrows (left and right)
  - Mouth
  - Nose
  - Face oval
- Understanding landmark coordinates and structure
- Analyzing landmark positions in 3D space
- Tracking landmarks across multiple frames
- Interactive landmark exploration

**Learning Outcomes**:
- Understand how facial landmarks are detected
- Learn about the MediaPipe landmark structure
- Visualize different facial regions
- Explore 3D landmark coordinates

**Estimated Time**: 20-30 minutes

### 02_emotion_analysis_demo.ipynb

**Focus**: Emotion detection and classification from facial landmarks

**Topics Covered**:
- Initializing the emotion analyzer
- Understanding Action Units (FACS - Facial Action Coding System)
- Detecting emotions from facial expressions
- Analyzing emotion probabilities and confidence scores
- Visualizing emotion predictions on video frames
- Creating emotion distribution charts
- Analyzing emotion timelines
- Comparing different detected emotions
- Understanding probability distributions
- Creating heatmaps for temporal analysis

**Key Concepts**:
- **Action Units**: Individual muscle movements that compose facial expressions
- **FACS**: Facial Action Coding System, the scientific standard for facial expression analysis
- **Emotion Classification**: How AUs combine to form different emotions
- **Confidence Scores**: Reliability of emotion predictions
- **Temporal Smoothing**: Reducing noise in emotion detection

**Learning Outcomes**:
- Understand the relationship between facial muscles and emotions
- Learn how Action Units combine to create expressions
- Analyze emotion distributions and patterns
- Create visualizations for emotion analysis
- Interpret confidence scores and probabilities

**Estimated Time**: 30-40 minutes

### 03_temporal_analysis.ipynb

**Focus**: Advanced temporal analysis of emotions over time

**Topics Covered**:
- Temporal smoothing techniques and window sizes
- Emotion stability metrics
- Transition analysis and patterns
- Microexpression detection
- Statistical analysis of emotion sequences
- Entropy and uncertainty measures
- Cumulative emotion duration
- Transition matrices and probabilities
- Comprehensive temporal visualizations

**Key Concepts**:
- **Temporal Smoothing**: Reducing noise by averaging over time windows
- **Emotion Stability**: Metrics measuring consistency of emotions
- **Transitions**: Changes from one emotion to another
- **Microexpressions**: Brief, involuntary facial expressions (1/25 to 1/5 second)
- **Transition Matrix**: Probability of transitioning between emotions
- **Entropy**: Measure of emotion uncertainty/ambiguity

**Advanced Topics**:
- Rolling statistics and moving averages
- Duration distribution analysis
- Transition frequency patterns
- Confidence stability over time
- Comprehensive statistical reporting

**Learning Outcomes**:
- Master temporal analysis techniques
- Detect and analyze microexpressions
- Understand emotion transitions and patterns
- Apply statistical methods to emotion sequences
- Create advanced visualizations

**Estimated Time**: 40-50 minutes

## Usage Guide

### Running the Notebooks

1. **Start Jupyter**:
   ```bash
   jupyter notebook notebooks/
   ```

2. **Open a notebook**: Click on the notebook file (e.g., `01_face_detection_demo.ipynb`)

3. **Run cells**: Execute cells sequentially using `Shift + Enter` or the "Run" button

4. **Interactive exploration**: Modify parameters and re-run cells to explore different configurations

### Tips for Best Experience

1. **Run sequentially**: Execute cells in order from top to bottom
2. **Read markdown cells**: They contain important explanations and context
3. **Experiment**: Try modifying parameters to see different results
4. **Check outputs**: Ensure each cell completes successfully before moving on
5. **Resource management**: Close detectors/readers when done to free resources

### Common Issues and Solutions

#### Model Not Found
```
Error: Model not found at .../models/face_landmarker.task
```
**Solution**: Download the MediaPipe model as described in Prerequisites section

#### Video Not Found
```
Error: Video not found at .../data/videos/youtube_short_emotion.mp4
```
**Solution**: Either:
- Place a video at the specified path, or
- Modify the `video_path` variable to point to your video

#### Import Errors
```
ModuleNotFoundError: No module named 'asdrp'
```
**Solution**: Ensure the project is installed:
```bash
# From project root
pip install -e .
# or
uv sync
```

#### Memory Issues
If you encounter memory issues with large videos:
- Reduce the number of frames processed (modify `frames_to_process`)
- Increase `skip_frames` to process fewer frames
- Use a shorter or lower resolution video

## Customization

### Using Your Own Video

To use your own video file, modify the video path in each notebook:

```python
# Change this line:
video_path = project_root / "data" / "videos" / "youtube_short_emotion.mp4"

# To your video path:
video_path = Path("/path/to/your/video.mp4")
```

### Adjusting Processing Parameters

You can modify various parameters to customize the analysis:

#### Face Detection
```python
detector = MediaPipeFaceDetector(
    model_path=str(model_path),
    min_detection_confidence=0.5,  # Adjust threshold (0.0 to 1.0)
    min_tracking_confidence=0.5,   # Adjust tracking threshold
    num_faces=1,                   # Number of faces to detect
    running_mode="VIDEO"           # "VIDEO" or "IMAGE"
)
```

#### Emotion Analysis
```python
emotion_analyzer = GeometryBasedEmotionAnalyzer(
    confidence_threshold=0.3  # Lower = more sensitive, higher = more strict
)
```

#### Temporal Smoothing
```python
temporal_analyzer = TemporalEmotionAnalyzer(
    window_size=7,          # Larger = smoother but less responsive
    min_confidence=0.3      # Minimum confidence to consider
)
```

#### Frame Processing
```python
skip_frames = 2  # Process every (skip_frames + 1)th frame
frames_to_process = list(range(0, min(metadata.total_frames, 300), skip_frames + 1))
```

## Output Examples

The notebooks generate various outputs:

### Visualizations
- Face landmark overlays
- Emotion-annotated video frames
- Distribution charts (bar, pie)
- Timeline plots
- Heatmaps
- Transition matrices
- Statistical plots

### Data
- Emotion predictions per frame
- Confidence scores
- Action Unit detections
- Temporal statistics
- Transition probabilities

### Statistics
- Emotion distributions
- Confidence statistics
- Stability metrics
- Transition frequencies
- Microexpression counts

## Further Resources

### Documentation
- [ASDRP Main README](../README.md) - Project overview
- [API Documentation](../docs/) - Detailed API reference
- [Pipeline Guide](../PIPELINE_USAGE.md) - Using the complete pipeline

### Research References
- **Facial Action Coding System (FACS)**: Ekman, P., & Friesen, W. V. (1978)
- **Basic Emotions**: Ekman, P. (1992). "An argument for basic emotions"
- **MediaPipe**: Google's ML framework for face detection

### Example Scripts
- [Pipeline Example](../examples/pipeline_example.py) - Complete pipeline usage
- [Visualization Demo](../examples/visualization_demo.py) - Visualization examples
- [Video Processing](../examples/video_processing_example.py) - Video processing

## Contributing

If you create additional notebooks or improve existing ones:

1. Follow the existing structure and style
2. Include clear explanations in markdown cells
3. Add error handling and helpful messages
4. Test with different video inputs
5. Document any new requirements

## Support

For issues or questions:
1. Check the [main project README](../README.md)
2. Review the [documentation](../docs/)
3. Examine the [example scripts](../examples/)
4. Create an issue on the project repository

## License

These notebooks are part of the ASDRP Emotion Detection project and are distributed under the MIT License. See the [LICENSE](../LICENSE) file for details.

---

**Happy Analyzing!** 🎭

Explore emotions, discover patterns, and learn about facial expression analysis through these interactive notebooks.
