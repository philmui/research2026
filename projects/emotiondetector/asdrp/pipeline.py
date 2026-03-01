"""Main pipeline orchestration for emotion detection.

This module provides the EmotionDetectionPipeline class that orchestrates the
complete emotion detection workflow, including video reading, face detection,
emotion analysis, visualization, and results export.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from asdrp.emotion import GeometryBasedEmotionAnalyzer, TemporalEmotionAnalyzer
from asdrp.emotion.base import EmotionPrediction
from asdrp.face.base import FaceLandmarks
from asdrp.face.detector import MediaPipeFaceDetector
from asdrp.utils.config import PipelineConfig
from asdrp.utils.export import (
    export_analysis_summary,
    export_emotions_to_csv,
    export_emotions_to_json,
)
from asdrp.video.camera import CameraCapture
from asdrp.video.frame import FrameData
from asdrp.video.reader import VideoFileReader


class PipelineError(Exception):
    """Base exception for pipeline errors."""

    pass


class EmotionDetectionPipeline:
    """Main pipeline for emotion detection from video.

    This class orchestrates the complete workflow for detecting and analyzing
    emotions from video files or real-time camera streams. It manages all
    components including video reading, face detection, emotion analysis,
    visualization, and results export.

    The pipeline supports:
    - Video file processing with frame skipping and range selection
    - Real-time camera stream processing
    - Batch and single frame processing modes
    - Temporal smoothing and tracking
    - Multiple output formats (JSON, CSV)
    - Optional visualization overlay
    - Progress tracking with tqdm
    - Context manager for automatic resource cleanup

    Attributes:
        config: PipelineConfig containing all settings for the pipeline.
        face_detector: Face detection and landmark extraction component.
        emotion_analyzer: Emotion classification component.
        temporal_analyzer: Optional temporal smoothing component.
        results: Accumulated results from processing.

    Example:
        >>> # Process video file with default settings
        >>> config = PipelineConfig.from_defaults(
        ...     model_path="face_landmarker.task",
        ...     input_path="input.mp4",
        ...     output_path="output.mp4"
        ... )
        >>> with EmotionDetectionPipeline(config) as pipeline:
        ...     results = pipeline.process_video()
        ...     pipeline.save_results("results.json")
        >>>
        >>> # Real-time camera processing
        >>> config = PipelineConfig.for_realtime_processing(
        ...     model_path="face_landmarker.task",
        ...     input_path="0"
        ... )
        >>> with EmotionDetectionPipeline(config) as pipeline:
        ...     for result in pipeline.process_stream(max_frames=100):
        ...         print(f"Emotion: {result['emotion']}")
    """

    def __init__(self, config: PipelineConfig) -> None:
        """Initialize the emotion detection pipeline.

        Args:
            config: PipelineConfig containing all pipeline settings.

        Raises:
            PipelineError: If initialization fails or configuration is invalid.
        """
        self.config = config
        self._logger = self._setup_logging()

        # Initialize components
        self.face_detector: Optional[MediaPipeFaceDetector] = None
        self.emotion_analyzer: Optional[GeometryBasedEmotionAnalyzer] = None
        self.temporal_analyzer: Optional[TemporalEmotionAnalyzer] = None

        # Results storage
        self.results: List[Dict[str, Any]] = []
        self._frame_count = 0

        # Video writer for output
        self._video_writer: Optional[cv2.VideoWriter] = None

        try:
            self._initialize_components()
        except Exception as e:
            self._logger.error(f"Failed to initialize pipeline: {e}")
            raise PipelineError(f"Pipeline initialization failed: {e}") from e

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the pipeline.

        Returns:
            Configured logger instance.
        """
        logger = logging.getLogger(__name__)
        logger.setLevel(getattr(logging, self.config.log_level))

        # Only add handler if none exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_components(self) -> None:
        """Initialize all pipeline components.

        Raises:
            PipelineError: If component initialization fails.
        """
        self._logger.info("Initializing pipeline components...")

        # Initialize face detector
        try:
            self.face_detector = MediaPipeFaceDetector(
                model_path=self.config.face_detection.model_path,
                min_detection_confidence=self.config.face_detection.min_detection_confidence,
                min_tracking_confidence=self.config.face_detection.min_tracking_confidence,
                num_faces=self.config.face_detection.num_faces,
                running_mode=self.config.face_detection.running_mode,
            )
            self._logger.info("Face detector initialized")
        except Exception as e:
            raise PipelineError(f"Failed to initialize face detector: {e}") from e

        # Initialize emotion analyzer
        try:
            if self.config.emotion_analysis.analyzer_type == "geometric":
                self.emotion_analyzer = GeometryBasedEmotionAnalyzer(
                    emotion_threshold=self.config.emotion_analysis.confidence_threshold,
                )
            else:
                raise NotImplementedError(
                    f"Analyzer type '{self.config.emotion_analysis.analyzer_type}' not implemented"
                )
            self._logger.info("Emotion analyzer initialized")
        except Exception as e:
            raise PipelineError(f"Failed to initialize emotion analyzer: {e}") from e

        # Initialize temporal analyzer if enabled
        if self.config.emotion_analysis.enable_temporal_smoothing:
            try:
                self.temporal_analyzer = TemporalEmotionAnalyzer(
                    smoothing_window=self.config.emotion_analysis.smoothing_window_size,
                )
                self._logger.info("Temporal analyzer initialized")
            except Exception as e:
                self._logger.warning(f"Failed to initialize temporal analyzer: {e}")
                self.temporal_analyzer = None

    def process_frame(
        self, frame_data: FrameData, visualize: bool = True
    ) -> Dict[str, Any]:
        """Process a single frame through the pipeline.

        Args:
            frame_data: FrameData containing the frame to process.
            visualize: Whether to apply visualization overlay.

        Returns:
            Dictionary containing processing results with keys:
                - frame_number: Frame number
                - timestamp: Frame timestamp
                - faces: List of face detection results
                - annotated_frame: Optional annotated frame (if visualize=True)

        Raises:
            PipelineError: If processing fails.
        """
        if self.face_detector is None or self.emotion_analyzer is None:
            raise PipelineError("Pipeline components not initialized")

        result: Dict[str, Any] = {
            "frame_number": frame_data.frame_number,
            "timestamp": frame_data.timestamp,
            "faces": [],
        }

        try:
            # Detect faces and extract landmarks
            timestamp_ms = frame_data.timestamp * 1000  # Convert to milliseconds
            face_landmarks_list = self.face_detector.detect(
                frame_data.frame, timestamp_ms=timestamp_ms
            )

            # Process each detected face
            for face_idx, face_landmarks in enumerate(face_landmarks_list):
                # Analyze emotion
                emotion_prediction = self.emotion_analyzer.analyze(face_landmarks)

                # Apply temporal smoothing if enabled
                if self.temporal_analyzer is not None:
                    emotion_prediction = self.temporal_analyzer.smooth_prediction(
                        emotion_prediction
                    )

                face_result = {
                    "face_id": face_idx,
                    "emotion": emotion_prediction.emotion.value,
                    "confidence": float(emotion_prediction.confidence),
                    "probabilities": {
                        emotion.value: float(prob)
                        for emotion, prob in emotion_prediction.probabilities.items()
                    },
                    "bounding_box": None,
                }

                if face_landmarks.bounding_box is not None:
                    bbox = face_landmarks.bounding_box
                    face_result["bounding_box"] = {
                        "x_min": float(bbox.x_min),
                        "y_min": float(bbox.y_min),
                        "width": float(bbox.width),
                        "height": float(bbox.height),
                    }

                result["faces"].append(face_result)

            # Add visualization if requested
            if visualize and self.config.save_annotated_video:
                annotated_frame = self._visualize_frame(
                    frame_data.frame, face_landmarks_list, result["faces"]
                )
                result["annotated_frame"] = annotated_frame

        except Exception as e:
            self._logger.error(f"Error processing frame {frame_data.frame_number}: {e}")
            raise PipelineError(f"Frame processing failed: {e}") from e

        return result

    def process_video(self, show_progress: bool = True) -> List[Dict[str, Any]]:
        """Process entire video file through the pipeline.

        Args:
            show_progress: Whether to display progress bar.

        Returns:
            List of result dictionaries, one per processed frame.

        Raises:
            PipelineError: If video processing fails.
        """
        self._logger.info(f"Starting video processing: {self.config.video.input_path}")
        self.results = []
        self._frame_count = 0

        try:
            # Open video reader
            with VideoFileReader(self.config.video.input_path) as reader:
                metadata = reader.get_metadata()
                self._logger.info(f"Video metadata: {metadata}")

                # Initialize video writer if needed
                if self.config.save_annotated_video and self.config.video.output_path:
                    self._initialize_video_writer(metadata)

                # Determine frame range
                start_frame = self.config.video.start_frame
                end_frame = (
                    self.config.video.end_frame
                    if self.config.video.end_frame is not None
                    else metadata.total_frames
                )

                # Calculate total frames to process
                frames_to_process = end_frame - start_frame
                if self.config.video.max_frames is not None:
                    frames_to_process = min(frames_to_process, self.config.video.max_frames)

                # Skip frames with consideration for skip_frames
                effective_frames = frames_to_process // (self.config.video.skip_frames + 1)

                # Seek to start frame
                if start_frame > 0:
                    reader.seek(start_frame)

                # Process frames with progress bar
                progress_bar = None
                if show_progress:
                    progress_bar = tqdm(
                        total=effective_frames,
                        desc="Processing video",
                        unit="frame",
                    )

                frames_processed = 0
                frames_since_last_process = 0

                for frame_data in reader:
                    # Check if we've reached the end frame
                    if frame_data.frame_number >= end_frame:
                        break

                    # Check if we've processed enough frames
                    if (
                        self.config.video.max_frames is not None
                        and frames_processed >= self.config.video.max_frames
                    ):
                        break

                    # Skip frames according to configuration
                    if frames_since_last_process < self.config.video.skip_frames:
                        frames_since_last_process += 1
                        continue

                    frames_since_last_process = 0

                    # Process frame
                    result = self.process_frame(
                        frame_data,
                        visualize=self.config.save_annotated_video,
                    )

                    # Write annotated frame if available
                    if (
                        self._video_writer is not None
                        and "annotated_frame" in result
                    ):
                        self._video_writer.write(result["annotated_frame"])
                        # Remove frame from result to save memory
                        del result["annotated_frame"]

                    # Display frame in real-time if configured
                    if self.config.video.display_realtime and "annotated_frame" in result:
                        cv2.imshow("Emotion Detection", result["annotated_frame"])
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    self.results.append(result)
                    frames_processed += 1
                    self._frame_count += 1

                    if progress_bar is not None:
                        progress_bar.update(1)

                if progress_bar is not None:
                    progress_bar.close()

                self._logger.info(f"Processed {frames_processed} frames")

        except Exception as e:
            self._logger.error(f"Video processing failed: {e}")
            raise PipelineError(f"Video processing failed: {e}") from e
        finally:
            self._cleanup_video_writer()
            if self.config.video.display_realtime:
                cv2.destroyAllWindows()

        return self.results

    def process_stream(
        self,
        camera_id: int = 0,
        max_frames: Optional[int] = None,
        display: bool = True,
    ) -> Iterator[Dict[str, Any]]:
        """Process real-time camera stream through the pipeline.

        Args:
            camera_id: Camera device ID (0 for default camera).
            max_frames: Maximum number of frames to process (None for unlimited).
            display: Whether to display frames in real-time.

        Yields:
            Result dictionary for each processed frame.

        Raises:
            PipelineError: If stream processing fails.
        """
        self._logger.info(f"Starting camera stream processing (camera_id={camera_id})")
        self.results = []
        self._frame_count = 0

        try:
            with CameraCapture(camera_id=camera_id) as camera:
                metadata = camera.get_metadata()
                self._logger.info(f"Camera metadata: {metadata}")

                frame_count = 0
                while max_frames is None or frame_count < max_frames:
                    # Read frame
                    frame_data = camera.read_frame()
                    if frame_data is None:
                        break

                    # Process frame
                    result = self.process_frame(frame_data, visualize=display)

                    # Display frame if requested
                    if display and "annotated_frame" in result:
                        cv2.imshow("Emotion Detection - Press 'q' to quit", result["annotated_frame"])
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        # Remove frame from result to save memory
                        del result["annotated_frame"]

                    self.results.append(result)
                    frame_count += 1
                    self._frame_count += 1

                    yield result

                self._logger.info(f"Processed {frame_count} frames from stream")

        except Exception as e:
            self._logger.error(f"Stream processing failed: {e}")
            raise PipelineError(f"Stream processing failed: {e}") from e
        finally:
            if display:
                cv2.destroyAllWindows()

    def get_results(self) -> List[Dict[str, Any]]:
        """Get accumulated processing results.

        Returns:
            List of result dictionaries from all processed frames.
        """
        return self.results

    def save_results(self, output_path: str | Path) -> None:
        """Save processing results to file.

        Automatically determines format based on file extension and
        config.output_format setting.

        Args:
            output_path: Path to output file (.json or .csv).

        Raises:
            PipelineError: If saving fails or results are empty.
        """
        if not self.results:
            raise PipelineError("No results to save. Process video/stream first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Prepare data for export
            emotions_list = []
            frame_numbers = []
            timestamps = []

            for result in self.results:
                frame_numbers.append(result["frame_number"])
                timestamps.append(result["timestamp"])

                # Get primary face emotion (first face if multiple)
                if result["faces"]:
                    emotions_list.append(result["faces"][0]["probabilities"])
                else:
                    # No face detected - add neutral
                    emotions_list.append({"neutral": 1.0})

            # Determine output format
            extension = output_path.suffix.lower()

            if self.config.output_format == "json" or extension == ".json":
                self._logger.info(f"Saving results to JSON: {output_path}")
                export_emotions_to_json(
                    emotions_list,
                    output_path,
                    frame_numbers=frame_numbers,
                    timestamps=timestamps,
                    metadata={"config": self.config.to_dict()},
                )

            elif self.config.output_format == "csv" or extension == ".csv":
                self._logger.info(f"Saving results to CSV: {output_path}")
                export_emotions_to_csv(
                    emotions_list,
                    output_path,
                    frame_numbers=frame_numbers,
                    timestamps=timestamps,
                )

            elif self.config.output_format == "both":
                # Save both formats
                json_path = output_path.with_suffix(".json")
                csv_path = output_path.with_suffix(".csv")

                self._logger.info(f"Saving results to JSON: {json_path}")
                export_emotions_to_json(
                    emotions_list,
                    json_path,
                    frame_numbers=frame_numbers,
                    timestamps=timestamps,
                    metadata={"config": self.config.to_dict()},
                )

                self._logger.info(f"Saving results to CSV: {csv_path}")
                export_emotions_to_csv(
                    emotions_list,
                    csv_path,
                    frame_numbers=frame_numbers,
                    timestamps=timestamps,
                )

            self._logger.info("Results saved successfully")

        except Exception as e:
            self._logger.error(f"Failed to save results: {e}")
            raise PipelineError(f"Failed to save results: {e}") from e

    def _visualize_frame(
        self,
        frame: np.ndarray,
        face_landmarks_list: List[FaceLandmarks],
        face_results: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Apply visualization overlay to frame.

        Args:
            frame: Input frame to annotate.
            face_landmarks_list: List of detected face landmarks.
            face_results: List of emotion analysis results.

        Returns:
            Annotated frame.
        """
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]

        for face_landmarks, face_result in zip(face_landmarks_list, face_results):
            # Draw bounding box
            if (
                self.config.visualization.draw_bounding_box
                and face_landmarks.bounding_box is not None
            ):
                bbox = face_landmarks.bounding_box
                x1, y1, x2, y2 = bbox.to_absolute(width, height)
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1),
                    (x2, y2),
                    self.config.visualization.bbox_color,
                    self.config.visualization.line_thickness,
                )

            # Draw landmarks
            if self.config.visualization.draw_landmarks:
                landmarks_abs = face_landmarks.to_absolute(width, height)
                for landmark in landmarks_abs:
                    x, y = int(landmark[0]), int(landmark[1])
                    cv2.circle(
                        annotated_frame,
                        (x, y),
                        self.config.visualization.landmark_radius,
                        self.config.visualization.landmark_color,
                        -1,
                    )

            # Draw emotion label
            if self.config.visualization.show_emotion:
                emotion = face_result["emotion"]
                confidence = face_result["confidence"]

                # Prepare label text
                if self.config.visualization.show_confidence:
                    label = f"{emotion}: {confidence:.2f}"
                else:
                    label = emotion

                # Calculate label position (above bounding box)
                if face_landmarks.bounding_box is not None:
                    bbox = face_landmarks.bounding_box
                    x1, y1, _, _ = bbox.to_absolute(width, height)
                    label_pos = (x1, max(y1 - 10, 20))
                else:
                    label_pos = (10, 30)

                # Draw label with background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = self.config.visualization.font_scale
                thickness = self.config.visualization.line_thickness

                (text_width, text_height), baseline = cv2.getTextSize(
                    label, font, font_scale, thickness
                )

                # Draw semi-transparent background
                overlay = annotated_frame.copy()
                cv2.rectangle(
                    overlay,
                    (label_pos[0], label_pos[1] - text_height - baseline),
                    (label_pos[0] + text_width, label_pos[1] + baseline),
                    (0, 0, 0),
                    -1,
                )
                cv2.addWeighted(
                    overlay,
                    self.config.visualization.background_alpha,
                    annotated_frame,
                    1 - self.config.visualization.background_alpha,
                    0,
                    annotated_frame,
                )

                # Draw text
                cv2.putText(
                    annotated_frame,
                    label,
                    label_pos,
                    font,
                    font_scale,
                    self.config.visualization.text_color,
                    thickness,
                    cv2.LINE_AA,
                )

        # Draw timestamp if configured
        if self.config.visualization.show_timestamp:
            timestamp_text = f"Time: {self._frame_count / 30:.2f}s"  # Assume 30 fps
            cv2.putText(
                annotated_frame,
                timestamp_text,
                (10, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.config.visualization.text_color,
                1,
                cv2.LINE_AA,
            )

        return annotated_frame

    def _initialize_video_writer(self, metadata) -> None:
        """Initialize video writer for output.

        Args:
            metadata: Video metadata from input video.
        """
        if self.config.video.output_path is None:
            return

        # Get output video properties
        fourcc = cv2.VideoWriter_fourcc(*self.config.video.codec)
        fps = self.config.video.fps if self.config.video.fps is not None else metadata.fps
        resolution = (
            self.config.video.resolution
            if self.config.video.resolution is not None
            else (metadata.width, metadata.height)
        )

        self._video_writer = cv2.VideoWriter(
            str(self.config.video.output_path),
            fourcc,
            fps,
            resolution,
        )

        if not self._video_writer.isOpened():
            raise PipelineError(f"Failed to open video writer: {self.config.video.output_path}")

        self._logger.info(f"Video writer initialized: {self.config.video.output_path}")

    def _cleanup_video_writer(self) -> None:
        """Release video writer resources."""
        if self._video_writer is not None:
            self._video_writer.release()
            self._video_writer = None

    def close(self) -> None:
        """Close pipeline and release all resources.

        Should be called when the pipeline is no longer needed.
        Automatically called when using context manager.
        """
        self._logger.info("Closing pipeline...")

        if self.face_detector is not None:
            self.face_detector.close()
            self.face_detector = None

        self._cleanup_video_writer()
        cv2.destroyAllWindows()

        self._logger.info("Pipeline closed")

    def __enter__(self) -> "EmotionDetectionPipeline":
        """Context manager entry.

        Returns:
            Self for use in with statement.
        """
        return self

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: object) -> None:
        """Context manager exit and cleanup.

        Args:
            exc_type: Exception type if an exception occurred.
            exc_val: Exception value if an exception occurred.
            exc_tb: Exception traceback if an exception occurred.
        """
        self.close()

    def __repr__(self) -> str:
        """String representation of the pipeline."""
        return (
            f"EmotionDetectionPipeline("
            f"input={self.config.video.input_path}, "
            f"frames_processed={self._frame_count})"
        )
