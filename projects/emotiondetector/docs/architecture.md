# Emotion Detector Architecture

## System Overview

The Emotion Detector is a real-time facial emotion recognition system that analyzes video streams to detect and classify human emotions. The system uses MediaPipe's Face Landmarker for facial feature extraction and applies geometry-based rules derived from the Facial Action Coding System (FACS) to identify emotions.

### Core Capabilities

- Real-time video processing (webcam, video files, or streams)
- Face detection and landmark extraction (478 landmarks)
- Geometry-based emotion classification
- Frame-by-frame emotion tracking
- Visualization of landmarks and detected emotions
- Export of analysis results

### Supported Emotions

The system currently detects seven basic emotions:
- **Neutral** - Relaxed facial expression
- **Happy** - Smile with raised cheeks
- **Sad** - Downturned mouth, drooping features
- **Angry** - Furrowed brows, tense jaw
- **Surprised** - Raised brows, open mouth
- **Fearful** - Wide eyes, raised brows, tense mouth
- **Disgusted** - Wrinkled nose, raised upper lip

## Architecture Layers

The system follows a layered architecture pattern with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────┐
│                   Application Layer                      │
│              (Examples & Notebooks)                      │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                  Visualization Layer                     │
│          (Display, Overlay, Export)                      │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                   Business Logic Layer                   │
│          (Emotion Detection & Classification)            │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                   Data Processing Layer                  │
│          (Face Detection & Landmark Extraction)          │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                      Input Layer                         │
│          (Video Capture & Frame Processing)              │
└─────────────────────────────────────────────────────────┘
```

### Layer Details

#### 1. Input Layer (`asdrp.video`)

**Purpose**: Handle video input from various sources and provide frame-by-frame access.

**Components**:
- `VideoReader`: Abstract interface for video input
- `WebcamReader`: Real-time webcam capture
- `FileReader`: Read from video files (MP4, AVI, etc.)
- `StreamReader`: Process video streams (RTSP, HTTP, etc.)
- `FrameBuffer`: Optional buffering for smooth processing

**Key Responsibilities**:
- Video source initialization and configuration
- Frame extraction and preprocessing
- FPS management and frame timing
- Resource cleanup and error handling

#### 2. Data Processing Layer (`asdrp.face`)

**Purpose**: Detect faces and extract facial landmarks using MediaPipe.

**Components**:
- `FaceLandmarker`: Wrapper around MediaPipe Face Landmarker
- `LandmarkProcessor`: Normalize and preprocess landmarks
- `FaceDetector`: Face detection and bounding box extraction
- `LandmarkGeometry`: Geometric calculations on landmarks

**Key Responsibilities**:
- Face detection in frames
- 478-point landmark extraction
- Landmark normalization (relative to face dimensions)
- Geometric feature calculation (distances, angles, ratios)

#### 3. Business Logic Layer (`asdrp.emotion`)

**Purpose**: Classify emotions based on facial geometry and action units.

**Components**:
- `EmotionClassifier`: Main emotion detection logic
- `ActionUnitDetector`: Detect facial action units (AUs)
- `EmotionRules`: Geometry-based classification rules
- `ConfidenceCalculator`: Compute confidence scores
- `EmotionTracker`: Temporal smoothing and tracking

**Key Responsibilities**:
- Action unit detection from landmarks
- Emotion classification using rule-based system
- Confidence score calculation
- Temporal smoothing to reduce jitter
- Multi-emotion detection (primary and secondary)

#### 4. Visualization Layer (`asdrp.visualization`)

**Purpose**: Render landmarks, emotions, and analysis results.

**Components**:
- `LandmarkRenderer`: Draw facial landmarks on frames
- `EmotionOverlay`: Display detected emotions and confidence
- `VideoWriter`: Export annotated videos
- `PlotGenerator`: Create analysis charts and graphs
- `ReportGenerator`: Generate analysis reports

**Key Responsibilities**:
- Real-time visualization of landmarks
- Emotion label and confidence display
- Video export with annotations
- Timeline and statistics visualization
- PDF/HTML report generation

#### 5. Application Layer (`examples` & `notebooks`)

**Purpose**: End-user applications and demonstrations.

**Components**:
- Example scripts for common use cases
- Jupyter notebooks for exploration
- Command-line tools
- Demo applications

## Key Design Patterns

### 1. Strategy Pattern (Emotion Classification)

Different emotion detection strategies can be implemented and swapped:

```python
class EmotionClassifier(ABC):
    @abstractmethod
    def classify(self, landmarks: FaceLandmarks) -> Emotion:
        pass

class GeometryBasedClassifier(EmotionClassifier):
    """Rule-based using facial geometry"""

class MLBasedClassifier(EmotionClassifier):
    """Machine learning-based (future)"""
```

### 2. Pipeline Pattern (Processing Flow)

Video processing follows a pipeline architecture:

```
VideoReader → FaceLandmarker → EmotionClassifier → Visualizer → Output
```

Each stage is independent and can be tested/replaced separately.

### 3. Factory Pattern (Video Readers)

Create appropriate video readers based on source type:

```python
class VideoReaderFactory:
    @staticmethod
    def create(source: str) -> VideoReader:
        if source.isdigit() or source == "webcam":
            return WebcamReader(int(source))
        elif source.startswith("rtsp://"):
            return StreamReader(source)
        else:
            return FileReader(source)
```

### 4. Observer Pattern (Emotion Tracking)

Components can subscribe to emotion detection events:

```python
class EmotionObserver(ABC):
    @abstractmethod
    def on_emotion_detected(self, emotion: Emotion, timestamp: float):
        pass

# Logger, UI updater, data collector can all observe
```

### 5. Dependency Injection

Components receive dependencies through constructors:

```python
class EmotionDetector:
    def __init__(
        self,
        face_landmarker: FaceLandmarker,
        classifier: EmotionClassifier,
        visualizer: Optional[Visualizer] = None
    ):
        self.face_landmarker = face_landmarker
        self.classifier = classifier
        self.visualizer = visualizer
```

## Component Interactions

### Basic Processing Flow

```
1. VideoReader reads frame
   ↓
2. FaceLandmarker detects face and extracts landmarks
   ↓
3. LandmarkProcessor normalizes coordinates
   ↓
4. ActionUnitDetector analyzes facial geometry
   ↓
5. EmotionClassifier applies rules to classify emotion
   ↓
6. EmotionTracker smooths temporal variations
   ↓
7. Visualizer renders results on frame
   ↓
8. Output (display, save, or stream)
```

### Data Flow Diagram

```
┌──────────────┐
│ Video Source │
└──────┬───────┘
       │ raw frames
       ↓
┌──────────────┐
│ Frame Buffer │
└──────┬───────┘
       │ RGB image
       ↓
┌──────────────────┐
│ Face Landmarker  │
└──────┬───────────┘
       │ 478 landmarks + face region
       ↓
┌────────────────────┐
│ Landmark Processor │
└──────┬─────────────┘
       │ normalized landmarks
       ↓
┌──────────────────────┐
│ Action Unit Detector │
└──────┬───────────────┘
       │ AU activations
       ↓
┌───────────────────────┐
│ Emotion Classifier    │
└──────┬────────────────┘
       │ emotion + confidence
       ↓
┌──────────────────┐
│ Emotion Tracker  │
└──────┬───────────┘
       │ smoothed emotion
       ↓
┌──────────────┐
│ Visualizer   │
└──────┬───────┘
       │ annotated frame
       ↓
┌──────────────┐
│ Output Sink  │
└──────────────┘
```

### Key Data Structures

```python
@dataclass
class FaceLandmarks:
    """478 facial landmarks from MediaPipe"""
    points: np.ndarray  # Shape: (478, 3) - x, y, z
    face_bbox: BoundingBox
    confidence: float
    timestamp: float

@dataclass
class ActionUnits:
    """Facial Action Unit activations"""
    au1_inner_brow_raiser: float  # 0-1
    au2_outer_brow_raiser: float
    au4_brow_lowerer: float
    au5_upper_lid_raiser: float
    au6_cheek_raiser: float
    au9_nose_wrinkler: float
    au10_upper_lip_raiser: float
    au12_lip_corner_puller: float
    au15_lip_corner_depressor: float
    au17_chin_raiser: float
    au20_lip_stretcher: float
    au25_lips_part: float
    au26_jaw_drop: float
    # ... additional AUs

@dataclass
class Emotion:
    """Detected emotion with metadata"""
    label: str  # "happy", "sad", "angry", etc.
    confidence: float  # 0-1
    action_units: ActionUnits
    secondary_emotion: Optional[str] = None
    timestamp: float = 0.0
```

## Extensibility Points

### 1. Adding New Emotions

To add a new emotion category:

1. Add emotion label to `EmotionType` enum
2. Define detection rules in `EmotionRules`
3. Specify required action units and thresholds
4. Add visualization color/icon in `EmotionOverlay`

```python
# In asdrp/emotion/rules.py
class EmotionRules:
    @staticmethod
    def detect_contempt(aus: ActionUnits) -> float:
        """Unilateral lip corner tightener"""
        asymmetry = abs(aus.au14_left - aus.au14_right)
        if asymmetry > 0.3 and max(aus.au14_left, aus.au14_right) > 0.4:
            return min(asymmetry * 2, 1.0)
        return 0.0
```

### 2. Custom Action Units

Add domain-specific action units:

```python
class CustomActionUnits(ActionUnits):
    """Extended AUs for specific use case"""
    au_head_tilt: float
    au_gaze_direction: float
    au_pupil_dilation: float
```

### 3. Alternative Classification Methods

Replace rule-based classifier with ML model:

```python
class MLEmotionClassifier(EmotionClassifier):
    def __init__(self, model_path: str):
        self.model = load_model(model_path)

    def classify(self, landmarks: FaceLandmarks) -> Emotion:
        features = self.extract_features(landmarks)
        prediction = self.model.predict(features)
        return self.to_emotion(prediction)
```

### 4. Custom Visualizations

Extend visualization capabilities:

```python
class HeatmapVisualizer(Visualizer):
    """Show emotion intensity as heatmap"""
    def render(self, frame: np.ndarray, emotion: Emotion):
        heatmap = self.create_heatmap(emotion.action_units)
        return cv2.addWeighted(frame, 0.7, heatmap, 0.3, 0)
```

### 5. New Video Sources

Add support for custom video sources:

```python
class CameraArrayReader(VideoReader):
    """Read from multiple synchronized cameras"""
    def __init__(self, camera_ids: List[int]):
        self.cameras = [cv2.VideoCapture(id) for id in camera_ids]
```

### 6. Plugin System

Support plugins for extended functionality:

```python
class EmotionPlugin(ABC):
    @abstractmethod
    def on_frame_processed(self, frame, landmarks, emotion):
        pass

class EmotionLogger(EmotionPlugin):
    """Log emotions to database"""

class EmotionTrigger(EmotionPlugin):
    """Trigger actions based on emotions"""
```

## Utility Components (`asdrp.utils`)

### Configuration Management

```python
class Config:
    """Central configuration management"""
    MEDIAPIPE_MODEL_PATH: str
    MIN_DETECTION_CONFIDENCE: float = 0.5
    MIN_TRACKING_CONFIDENCE: float = 0.5
    EMOTION_SMOOTHING_WINDOW: int = 5
    VIDEO_OUTPUT_FPS: int = 30
```

### Logging

```python
class Logger:
    """Structured logging for debugging and monitoring"""
    - Frame processing times
    - Detection confidence
    - Classification results
    - Error tracking
```

### Performance Monitoring

```python
class PerformanceMonitor:
    """Track system performance metrics"""
    - FPS (frames per second)
    - Processing latency
    - Memory usage
    - GPU utilization
```

## Error Handling

### Exception Hierarchy

```python
class EmotionDetectorError(Exception):
    """Base exception"""

class VideoSourceError(EmotionDetectorError):
    """Video input errors"""

class FaceNotFoundError(EmotionDetectorError):
    """No face detected in frame"""

class ModelLoadError(EmotionDetectorError):
    """MediaPipe model loading failed"""

class ProcessingError(EmotionDetectorError):
    """Frame processing failed"""
```

### Graceful Degradation

- If face not detected: Skip frame or use last known position
- If landmark extraction fails: Use bounding box only
- If emotion classification uncertain: Report "neutral" with low confidence
- If processing too slow: Drop frames to maintain real-time performance

## Testing Strategy

### Unit Tests (`tests/unit/`)
- Individual component functionality
- Geometry calculations
- Action unit detection logic
- Emotion classification rules

### Integration Tests (`tests/integration/`)
- Pipeline integration
- Component interactions
- Data flow validation

### End-to-End Tests (`tests/e2e/`)
- Full system workflows
- Video file processing
- Real-time webcam processing
- Export functionality

### Test Data
- Synthetic landmark data
- Sample videos with known emotions
- Edge cases (multiple faces, occlusions, lighting)

## Performance Considerations

### Optimization Strategies

1. **Frame Skipping**: Process every Nth frame for real-time performance
2. **ROI Tracking**: Once face detected, track region instead of full frame
3. **Landmark Caching**: Reuse landmarks for multiple analyses
4. **Parallel Processing**: Process multiple frames in parallel
5. **GPU Acceleration**: Leverage MediaPipe GPU support

### Benchmarks

Target performance metrics:
- **Webcam**: 30 FPS on laptop (CPU)
- **Video File**: 2x real-time processing
- **Latency**: <50ms per frame
- **Memory**: <500MB for basic operation

## Future Enhancements

1. **Deep Learning Integration**
   - CNN-based emotion classification
   - Transfer learning from pre-trained models
   - Ensemble methods (geometry + ML)

2. **Multi-Face Support**
   - Track multiple faces simultaneously
   - Associate emotions with identities

3. **Temporal Analysis**
   - Emotion transitions over time
   - Micro-expressions detection
   - Emotional state tracking

4. **3D Analysis**
   - Head pose estimation
   - 3D facial reconstruction
   - Depth-based features

5. **Real-World Integration**
   - REST API for remote processing
   - WebSocket streaming
   - Mobile SDK
   - Cloud deployment

## References

- MediaPipe Face Landmarker: https://developers.google.com/mediapipe/solutions/vision/face_landmarker
- Facial Action Coding System (FACS): Ekman & Friesen (1978)
- Emotion Recognition: A Survey - Zhang et al. (2021)
