"""Object detection module utilizing Ultralytics YOLO models."""
"""Detector.py"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Structured object detection result."""

    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    norm_x1: float
    norm_y1: float
    norm_x2: float
    norm_y2: float
    norm_width: float
    norm_height: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert detection data to JSON-serializable dictionary."""
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "confidence": float(round(self.confidence, 4)),
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
            "normalized_bbox": [
                float(round(self.norm_x1, 4)),
                float(round(self.norm_y1, 4)),
                float(round(self.norm_x2, 4)),
                float(round(self.norm_y2, 4)),
            ],
            "normalized_dimensions": [
                float(round(self.norm_width, 4)),
                float(round(self.norm_height, 4)),
            ],
        }


class ObjectDetector:
    """Wrapper around Ultralytics YOLO for object detection with fallback mechanisms."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "auto",
    ) -> None:
        """Initialize ObjectDetector configuration without loading weights immediately."""
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device_setting = device
        self.device = self._resolve_device(device)
        self.model: Optional[YOLO] = None

    def _resolve_device(self, requested_device: str) -> str:
        """Determine hardware device based on availability and requested setting."""
        req = requested_device.lower().strip()
        if req == "cuda":
            if torch.cuda.is_available():
                return "cuda"
            logger.warning("CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        elif req == "cpu":
            return "cpu"
        else:
            resolved = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info("Auto-detected processing device: %s", resolved)
            return resolved

    def load(self) -> None:
        """Load YOLO model into memory on specified compute device."""
        if self.model is not None:
            return

        logger.info(
            "Loading detector model '%s' onto device '%s'...",
            self.model_name,
            self.device,
        )
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            logger.info("Detector model successfully loaded.")
        except Exception as err:
            logger.error("Failed to load YOLO model '%s': %s", self.model_name, err)
            raise RuntimeError(f"Detector initialization failed: {err}") from err

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Perform object detection on an input BGR image frame.

        Args:
            frame: Input numpy array representing BGR image frame.

        Returns:
            List of structured Detection objects.
        """
        if self.model is None:
            self.load()

        if frame is None or frame.size == 0:
            logger.warning("Received invalid or empty frame for detection.")
            return []

        frame_height, frame_width = frame.shape[:2]
        if frame_height == 0 or frame_width == 0:
            logger.warning("Frame dimensions are zero.")
            return []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
            )
        except Exception as err:
            logger.error("Error occurred during model inference: %s", err)
            return []

        detections: List[Detection] = []
        if not results or len(results) == 0:
            return detections

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return detections

        # Extract numpy representations of prediction tensors
        xyxy_tensor = boxes.xyxy.cpu().numpy()
        conf_tensor = boxes.conf.cpu().numpy()
        cls_tensor = boxes.cls.cpu().numpy()

        names = self.model.names if hasattr(self.model, "names") else {}

        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy_tensor[i]
            conf = float(conf_tensor[i])
            cls_id = int(cls_tensor[i])
            cls_name = str(names.get(cls_id, f"class_{cls_id}"))

            # Clamp bounding boxes to image dimensions
            x1_clamped = max(0.0, min(float(x1), float(frame_width)))
            y1_clamped = max(0.0, min(float(y1), float(frame_height)))
            x2_clamped = max(0.0, min(float(x2), float(frame_width)))
            y2_clamped = max(0.0, min(float(y2), float(frame_height)))

            width = max(0.0, x2_clamped - x1_clamped)
            height = max(0.0, y2_clamped - y1_clamped)

            # Compute normalized bounding box coordinates
            norm_x1 = x1_clamped / frame_width
            norm_y1 = y1_clamped / frame_height
            norm_x2 = x2_clamped / frame_width
            norm_y2 = y2_clamped / frame_height
            norm_width = width / frame_width
            norm_height = height / frame_height

            detections.append(
                Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    x1=x1_clamped,
                    y1=y1_clamped,
                    x2=x2_clamped,
                    y2=y2_clamped,
                    width=width,
                    height=height,
                    norm_x1=norm_x1,
                    norm_y1=norm_y1,
                    norm_x2=norm_x2,
                    norm_y2=norm_y2,
                    norm_width=norm_width,
                    norm_height=norm_height,
                )
            )

        return detections