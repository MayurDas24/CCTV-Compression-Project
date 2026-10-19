"""
Phase 2 - AI Perception Layer

Main orchestration engine for:
    - video / RTSP ingestion
    - configurable frame sampling
    - YOLO object detection
    - importance scoring
    - ROI generation
    - perception metadata generation
    - final processing summary

This phase DOES NOT perform adaptive video compression.
It produces perception metadata that Phase 3 will consume.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import yaml

from detector import Detection, ObjectDetector
from frame_sampler import FrameSampler
from metadata_writer import MetadataWriter


logger = logging.getLogger("phase_2.perception")


# ============================================================
# DATA MODELS
# ============================================================


@dataclass
class ROI:
    """Region of Interest generated from an object detection."""

    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    class_name: str
    importance_score: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert ROI into JSON-safe dictionary."""

        return {
            "bbox": [
                float(round(self.x1, 2)),
                float(round(self.y1, 2)),
                float(round(self.x2, 2)),
                float(round(self.y2, 2)),
            ],
            "dimensions": [
                float(round(self.width, 2)),
                float(round(self.height, 2)),
            ],
            "class_name": str(self.class_name),
            "importance_score": float(
                round(self.importance_score, 4)
            ),
        }


# ============================================================
# IMPORTANCE SCORING
# ============================================================


class ImportanceScorer:
    """
    Calculate a transparent heuristic importance score.

    Phase 2 only generates importance metadata.

    Phase 3 will later translate this information into actual
    encoder / compression decisions.
    """

    def __init__(self, config: Dict[str, Any]) -> None:

        importance_config = config.get(
            "importance",
            {},
        )

        self.class_weights: Dict[str, float] = {
            str(key).lower(): float(value)
            for key, value in importance_config.get(
                "class_weights",
                {},
            ).items()
        }

        self.default_class_weight = float(
            importance_config.get(
                "default_class_weight",
                0.3,
            )
        )

        self.confidence_weight = float(
            importance_config.get(
                "confidence_weight",
                0.3,
            )
        )

        self.area_weight = float(
            importance_config.get(
                "area_weight",
                0.3,
            )
        )

        self.center_weight = float(
            importance_config.get(
                "center_weight",
                0.4,
            )
        )

        total = (
            self.confidence_weight
            + self.area_weight
            + self.center_weight
        )

        if total <= 0:
            raise ValueError(
                "Importance scoring weights must sum "
                "to a value greater than zero."
            )

        # Normalize weights.
        self.confidence_weight /= total
        self.area_weight /= total
        self.center_weight /= total

    def calculate_score(
        self,
        detection: Detection,
    ) -> float:
        """
        Return importance score in range [0.0, 1.0].

        Factors:
            - object class priority
            - detector confidence
            - relative object size
            - proximity to frame center
        """

        class_weight = self.class_weights.get(
            detection.class_name.lower(),
            self.default_class_weight,
        )

        class_weight = float(
            np.clip(class_weight, 0.0, 1.0)
        )

        confidence_factor = float(
            np.clip(
                detection.confidence,
                0.0,
                1.0,
            )
        )

        normalized_area = (
            detection.norm_width
            * detection.norm_height
        )

        # A huge object should matter more than a very tiny one,
        # but raw area percentages are often too small.
        #
        # Scale the area slightly before clamping.
        area_factor = float(
            np.clip(
                normalized_area * 5.0,
                0.0,
                1.0,
            )
        )

        object_center_x = (
            detection.norm_x1
            + detection.norm_x2
        ) / 2.0

        object_center_y = (
            detection.norm_y1
            + detection.norm_y2
        ) / 2.0

        distance = float(
            np.sqrt(
                (object_center_x - 0.5) ** 2
                + (object_center_y - 0.5) ** 2
            )
        )

        maximum_distance = float(
            np.sqrt(
                (0.5 ** 2)
                + (0.5 ** 2)
            )
        )

        center_factor = 1.0 - (
            distance / maximum_distance
        )

        center_factor = float(
            np.clip(
                center_factor,
                0.0,
                1.0,
            )
        )

        spatial_score = (
            self.confidence_weight
            * confidence_factor
            + self.area_weight
            * area_factor
            + self.center_weight
            * center_factor
        )

        final_score = (
            class_weight
            * spatial_score
        )

        return float(
            np.clip(
                final_score,
                0.0,
                1.0,
            )
        )


# ============================================================
# PERCEPTION ENGINE
# ============================================================


class PerceptionEngine:
    """Main Phase 2 processing pipeline."""

    def __init__(
        self,
        config_path: str = "config.yaml",
    ) -> None:

        self.script_directory = (
            Path(__file__).resolve().parent
        )

        self.config_path = (
            self.script_directory
            / config_path
        ).resolve()

        self.config = self._load_config(
            self.config_path
        )

        self._setup_logging()

        logger.info(
            "=============================================="
        )
        logger.info(
            "Starting Phase 2 AI Perception Layer"
        )
        logger.info(
            "=============================================="
        )

        self._normalize_sampling_config()

        self.source = self._resolve_source()

        detector_config = self.config.get(
            "detector",
            {},
        )

        self.detector = ObjectDetector(
            model_name=detector_config.get(
                "model",
                "yolov8n.pt",
            ),
            confidence_threshold=float(
                detector_config.get(
                    "confidence_threshold",
                    0.25,
                )
            ),
            iou_threshold=float(
                detector_config.get(
                    "iou_threshold",
                    0.45,
                )
            ),
            device=str(
                detector_config.get(
                    "device",
                    "auto",
                )
            ),
        )

        self.importance_scorer = (
            ImportanceScorer(
                self.config
            )
        )

        output_config = self.config.get(
            "output",
            {},
        )

        metadata_path = self._resolve_output_path(
            output_config.get(
                "metadata_file",
                "outputs/perception.jsonl",
            )
        )

        summary_path = self._resolve_output_path(
            output_config.get(
                "summary_file",
                "outputs/summary.json",
            )
        )

        self.writer = MetadataWriter(
            metadata_file=str(metadata_path),
            summary_file=str(summary_path),
        )

        self.sampler: Optional[
            FrameSampler
        ] = None

    # ========================================================
    # CONFIGURATION
    # ========================================================

    @staticmethod
    def _load_config(
        path: Path,
    ) -> Dict[str, Any]:
        """Load and validate YAML configuration."""

        if not path.exists():
            raise FileNotFoundError(
                f"Configuration file not found: "
                f"{path}"
            )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as config_file:

                config = yaml.safe_load(
                    config_file
                )

        except yaml.YAMLError as error:
            raise RuntimeError(
                f"Invalid YAML configuration: "
                f"{error}"
            ) from error

        except OSError as error:
            raise RuntimeError(
                f"Unable to read configuration: "
                f"{error}"
            ) from error

        if not isinstance(config, dict):
            raise ValueError(
                "config.yaml must contain a "
                "top-level YAML mapping."
            )

        return config

    def _setup_logging(self) -> None:
        """Configure application logging."""

        logging_config = self.config.get(
            "logging",
            {},
        )

        requested_level = str(
            logging_config.get(
                "level",
                "INFO",
            )
        ).upper()

        level = getattr(
            logging,
            requested_level,
            logging.INFO,
        )

        logging.basicConfig(
            level=level,
            format=(
                "%(asctime)s "
                "[%(levelname)s] "
                "%(name)s: "
                "%(message)s"
            ),
            handlers=[
                logging.StreamHandler(
                    sys.stdout
                )
            ],
            force=True,
        )

    def _normalize_sampling_config(
        self,
    ) -> None:
        """
        Translate the current config.yaml naming into the naming
        expected by the existing FrameSampler implementation.

        Current config:

            mode: fps
            fps: 2.0

        becomes:

            mode: fixed_fps
            target_fps: 2.0

        And:

            mode: interval
            frame_interval: 15

        becomes:

            mode: every_n
            step_interval: 15
        """

        sampling = self.config.setdefault(
            "sampling",
            {},
        )

        mode = str(
            sampling.get(
                "mode",
                "fps",
            )
        ).strip().lower()

        if mode in {
            "fps",
            "fixed_fps",
        }:

            sampling["mode"] = (
                "fixed_fps"
            )

            if (
                "target_fps"
                not in sampling
            ):
                sampling["target_fps"] = float(
                    sampling.get(
                        "fps",
                        2.0,
                    )
                )

        elif mode in {
            "interval",
            "every_n",
        }:

            sampling["mode"] = (
                "every_n"
            )

            if (
                "step_interval"
                not in sampling
            ):
                sampling[
                    "step_interval"
                ] = int(
                    sampling.get(
                        "frame_interval",
                        15,
                    )
                )

        elif mode == "all":
            sampling["mode"] = "all"

        else:
            raise ValueError(
                "Unsupported sampling mode "
                f"'{mode}'. Expected "
                "'fps', 'interval', or 'all'."
            )

    def _resolve_source(self) -> str:
        """
        Resolve local video source robustly.

        RTSP URLs are returned unchanged.

        For local files, several sensible locations are checked so
        that both of these work:

            sample_videos/test1.mp4

        and:

            ../sample_videos/test1.mp4
        """

        input_config = self.config.get(
            "input",
            {},
        )

        source = str(
            input_config.get(
                "source",
                "../sample_videos/test1.mp4",
            )
        ).strip()

        input_type = str(
            input_config.get(
                "type",
                "file",
            )
        ).lower()

        if (
            input_type == "rtsp"
            or source.lower().startswith(
                "rtsp://"
            )
        ):
            return source

        configured_path = Path(
            source
        )

        candidates = []

        if configured_path.is_absolute():
            candidates.append(
                configured_path
            )
        else:
            # Relative to phase_2 directory.
            candidates.append(
                self.script_directory
                / configured_path
            )

            # Relative to project root.
            candidates.append(
                self.script_directory.parent
                / configured_path
            )

            # Common project layout fallback.
            candidates.append(
                self.script_directory.parent
                / "sample_videos"
                / configured_path.name
            )

        for candidate in candidates:

            candidate = candidate.resolve()

            if candidate.exists():
                return str(candidate)

        candidate_list = "\n".join(
            f"  - {path.resolve()}"
            for path in candidates
        )

        raise FileNotFoundError(
            "Input video could not be found.\n"
            f"Configured source: {source}\n"
            "Checked:\n"
            f"{candidate_list}"
        )

    def _resolve_output_path(
        self,
        value: str,
    ) -> Path:
        """Resolve output path relative to phase_2."""

        path = Path(value)

        if path.is_absolute():
            return path

        return (
            self.script_directory
            / path
        ).resolve()

    # ========================================================
    # ROI GENERATION
    # ========================================================

    @staticmethod
    def _create_roi(
        detection: Detection,
        importance_score: float,
        frame_width: int,
        frame_height: int,
    ) -> ROI:
        """Generate a valid ROI from a detection."""

        x1 = float(
            np.clip(
                detection.x1,
                0,
                frame_width,
            )
        )

        y1 = float(
            np.clip(
                detection.y1,
                0,
                frame_height,
            )
        )

        x2 = float(
            np.clip(
                detection.x2,
                0,
                frame_width,
            )
        )

        y2 = float(
            np.clip(
                detection.y2,
                0,
                frame_height,
            )
        )

        width = max(
            0.0,
            x2 - x1,
        )

        height = max(
            0.0,
            y2 - y1,
        )

        return ROI(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            width=width,
            height=height,
            class_name=detection.class_name,
            importance_score=importance_score,
        )

    # ========================================================
    # MAIN PIPELINE
    # ========================================================

    def process_video(self) -> None:
        """Execute Phase 2 perception processing."""

        logger.info(
            "Input source: %s",
            self.source,
        )

        logger.info(
            "Loading YOLO detector..."
        )

        self.detector.load()

        logger.info(
            "Detector ready | Model: %s | Device: %s",
            self.detector.model_name,
            self.detector.device,
        )

        capture = cv2.VideoCapture(
            self.source
        )

        if not capture.isOpened():

            self.writer.close()

            raise RuntimeError(
                f"Unable to open video source: "
                f"{self.source}"
            )

        frame_width = int(
            capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

        frame_height = int(
            capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

        source_fps = float(
            capture.get(
                cv2.CAP_PROP_FPS
            )
        )

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if (
            source_fps <= 0
            or np.isnan(source_fps)
        ):
            logger.warning(
                "Invalid FPS reported by input. "
                "Using 30 FPS fallback."
            )

            source_fps = 30.0

        if (
            frame_width <= 0
            or frame_height <= 0
        ):
            capture.release()
            self.writer.close()

            raise RuntimeError(
                "Invalid video dimensions "
                f"{frame_width}x{frame_height}"
            )

        logger.info(
            "Video properties | "
            "Resolution: %dx%d | "
            "FPS: %.2f | "
            "Frames: %s",
            frame_width,
            frame_height,
            source_fps,
            (
                total_frames
                if total_frames > 0
                else "unknown"
            ),
        )

        self.sampler = (
            FrameSampler.from_config(
                self.config,
                video_fps=source_fps,
            )
        )

        logger.info(
            "Sampling configuration | "
            "Mode: %s | "
            "Target FPS: %.2f | "
            "Frame interval: %d | "
            "Effective FPS: %.2f",
            self.sampler.mode,
            self.sampler.target_fps,
            self.sampler.step_interval,
            self.sampler.effective_fps(
                source_fps
            ),
        )

        frame_index = 0
        sampled_frames = 0

        total_detections = 0

        confidence_sum = 0.0

        class_distribution: Dict[
            str,
            int
        ] = {}

        maximum_importance = 0.0

        start_time = time.perf_counter()

        processing_error: Optional[
            Exception
        ] = None

        try:

            while True:

                success, frame = (
                    capture.read()
                )

                if (
                    not success
                    or frame is None
                ):

                    logger.info(
                        "End of video / stream "
                        "at frame %d.",
                        frame_index,
                    )

                    break

                if not self.sampler.should_sample(
                    frame_index
                ):
                    frame_index += 1
                    continue

                sampled_frames += 1

                timestamp_seconds = (
                    frame_index
                    / source_fps
                )

                inference_start = (
                    time.perf_counter()
                )

                detections = (
                    self.detector.detect(
                        frame
                    )
                )

                inference_ms = (
                    time.perf_counter()
                    - inference_start
                ) * 1000.0

                detection_records: List[
                    Dict[str, Any]
                ] = []

                roi_records: List[
                    ROI
                ] = []

                classes_present = set()

                frame_importance = 0.0

                for detection in detections:

                    importance = (
                        self.importance_scorer
                        .calculate_score(
                            detection
                        )
                    )

                    roi = self._create_roi(
                        detection,
                        importance,
                        frame_width,
                        frame_height,
                    )

                    roi_records.append(
                        roi
                    )

                    record = (
                        detection.to_dict()
                    )

                    record[
                        "importance_score"
                    ] = float(
                        round(
                            importance,
                            4,
                        )
                    )

                    record[
                        "area"
                    ] = float(
                        round(
                            detection.width
                            * detection.height,
                            2,
                        )
                    )

                    record[
                        "normalized_area"
                    ] = float(
                        round(
                            detection.norm_width
                            * detection.norm_height,
                            6,
                        )
                    )

                    detection_records.append(
                        record
                    )

                    classes_present.add(
                        detection.class_name
                    )

                    total_detections += 1

                    confidence_sum += (
                        detection.confidence
                    )

                    class_distribution[
                        detection.class_name
                    ] = (
                        class_distribution.get(
                            detection.class_name,
                            0,
                        )
                        + 1
                    )

                    frame_importance = max(
                        frame_importance,
                        importance,
                    )

                    maximum_importance = max(
                        maximum_importance,
                        importance,
                    )

                frame_metadata = {
                    "schema_version": "1.0",
                    "frame_index": int(
                        frame_index
                    ),
                    "timestamp_seconds": float(
                        round(
                            timestamp_seconds,
                            4,
                        )
                    ),
                    "frame_width": int(
                        frame_width
                    ),
                    "frame_height": int(
                        frame_height
                    ),
                    "source_fps": float(
                        round(
                            source_fps,
                            4,
                        )
                    ),
                    "inference_latency_ms": float(
                        round(
                            inference_ms,
                            2,
                        )
                    ),
                    "number_of_objects": int(
                        len(detections)
                    ),
                    "classes_present": sorted(
                        classes_present
                    ),
                    "frame_importance_score": float(
                        round(
                            frame_importance,
                            4,
                        )
                    ),
                    "detections": detection_records,
                    "rois": [
                        roi.to_dict()
                        for roi in roi_records
                    ],
                }

                self.writer.write_record(
                    frame_metadata
                )

                # Keep logs readable.
                if (
                    sampled_frames == 1
                    or sampled_frames % 10 == 0
                ):

                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    processing_rate = (
                        sampled_frames
                        / elapsed
                        if elapsed > 0
                        else 0.0
                    )

                    logger.info(
                        "[Metrics] "
                        "Frame=%d | "
                        "Sampled=%d | "
                        "Objects=%d | "
                        "Inference=%.1f ms | "
                        "Processing=%.2f sampled FPS",
                        frame_index,
                        sampled_frames,
                        len(detections),
                        inference_ms,
                        processing_rate,
                    )

                frame_index += 1

        except KeyboardInterrupt:

            logger.warning(
                "Phase 2 interrupted by user."
            )

        except Exception as error:

            processing_error = error

            logger.error(
                "Unexpected Phase 2 processing "
                "error: %s",
                error,
                exc_info=True,
            )

        finally:

            capture.release()

            # Flush and close frame metadata before
            # generating summary.
            self.writer.flush()

        elapsed_seconds = (
            time.perf_counter()
            - start_time
        )

        average_confidence = (
            confidence_sum
            / total_detections
            if total_detections > 0
            else 0.0
        )

        average_detections = (
            total_detections
            / sampled_frames
            if sampled_frames > 0
            else 0.0
        )

        processing_fps = (
            sampled_frames
            / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )

        effective_sampling_fps = (
            self.sampler.effective_fps(
                source_fps
            )
            if self.sampler
            else 0.0
        )

        summary = {
            "schema_version": "1.0",
            "phase": "phase_2_perception",
            "status": (
                "completed"
                if processing_error is None
                else "completed_with_error"
            ),
            "input_source": self.source,
            "input_resolution": (
                f"{frame_width}x"
                f"{frame_height}"
            ),
            "source_fps": float(
                round(
                    source_fps,
                    4,
                )
            ),
            "total_frames_read": int(
                frame_index
            ),
            "sampled_frames_analyzed": int(
                sampled_frames
            ),
            "sampling": {
                "mode": (
                    self.sampler.mode
                    if self.sampler
                    else "unknown"
                ),
                "target_fps": float(
                    self.sampler.target_fps
                    if self.sampler
                    else 0.0
                ),
                "step_interval": int(
                    self.sampler.step_interval
                    if self.sampler
                    else 1
                ),
                "effective_fps": float(
                    round(
                        effective_sampling_fps,
                        4,
                    )
                ),
            },
            "model": {
                "name": self.detector.model_name,
                "device": self.detector.device,
                "confidence_threshold": (
                    self.detector
                    .confidence_threshold
                ),
                "iou_threshold": (
                    self.detector
                    .iou_threshold
                ),
            },
            "total_detections": int(
                total_detections
            ),
            "detection_counts_by_class": (
                dict(
                    sorted(
                        class_distribution.items()
                    )
                )
            ),
            "average_detections_per_sampled_frame": float(
                round(
                    average_detections,
                    4,
                )
            ),
            "average_confidence": float(
                round(
                    average_confidence,
                    4,
                )
            ),
            "maximum_importance_score": float(
                round(
                    maximum_importance,
                    4,
                )
            ),
            "processing_fps": float(
                round(
                    processing_fps,
                    4,
                )
            ),
            "total_processing_duration_seconds": float(
                round(
                    elapsed_seconds,
                    3,
                )
            ),
        }

        if processing_error is not None:
            summary["error"] = str(
                processing_error
            )

        try:
            self.writer.write_summary(
                summary
            )

        finally:
            self.writer.close()

        logger.info(
            "=============================================="
        )

        logger.info(
            "Phase 2 Perception Completed"
        )

        logger.info(
            "Frames read      : %d",
            frame_index,
        )

        logger.info(
            "Frames analyzed  : %d",
            sampled_frames,
        )

        logger.info(
            "Detections       : %d",
            total_detections,
        )

        logger.info(
            "Average confidence: %.4f",
            average_confidence,
        )

        logger.info(
            "Processing speed : %.2f sampled FPS",
            processing_fps,
        )

        logger.info(
            "Duration         : %.2f sec",
            elapsed_seconds,
        )

        logger.info(
            "=============================================="
        )

        if processing_error is not None:
            raise processing_error


# ============================================================
# ENTRY POINT
# ============================================================


def main() -> None:
    """Phase 2 command-line entry point."""

    parser = argparse.ArgumentParser(
        description=(
            "Run the Phase 2 AI perception pipeline."
        )
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help=(
            "Configuration file path. Absolute paths are used "
            "as provided; relative paths are resolved from the "
            "phase_2 directory. (default: config.yaml)"
        ),
    )
    arguments = parser.parse_args()

    try:

        engine = PerceptionEngine(
            config_path=arguments.config
        )

        engine.process_video()

    except KeyboardInterrupt:

        logger.warning(
            "Application stopped by user."
        )

    except Exception as error:

        logger.critical(
            "Phase 2 execution failed: %s",
            error,
            exc_info=True,
        )

        sys.exit(1)


if __name__ == "__main__":
    main()
