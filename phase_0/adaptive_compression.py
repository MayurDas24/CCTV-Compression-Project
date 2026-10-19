import argparse
import logging
import pathlib
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("AdaptiveCompression")


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int
    class_id: int
    confidence: float


class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_thresh: float = 0.35, target_classes: List[int] = None):
        self.conf_thresh = conf_thresh
        self.target_classes = target_classes or [0, 2, 3, 5, 7]
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_name)
        except ImportError:
            logger.error("Ultralytics library missing. Install via `pip install ultralytics`.")
            raise

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        results = self.model(frame, verbose=False)[0]
        boxes = []
        for r in results.boxes:
            cls_id = int(r.cls[0])
            conf = float(r.conf[0])
            if conf >= self.conf_thresh and cls_id in self.target_classes:
                coords = r.xyxy[0].cpu().numpy().astype(int)
                boxes.append(BoundingBox(coords[0], coords[1], coords[2], coords[3], cls_id, conf))
        return boxes


class ROIMapGenerator:
    @staticmethod
    def generate_mask(frame_shape: Tuple[int, int, int], boxes: List[BoundingBox], padding: int = 10) -> np.ndarray:
        h, w, _ = frame_shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for box in boxes:
            x1 = max(0, box.xmin - padding)
            y1 = max(0, box.ymin - padding)
            x2 = min(w, box.xmax + padding)
            y2 = min(h, box.ymax + padding)
            mask[y1:y2, x1:x2] = 255
        return mask


class AdaptiveCompressor:
    def __init__(self, config_path: pathlib.Path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)["adaptive"]
        
        self.detector = ObjectDetector(
            model_name=self.config.get("model_name", "yolov8n.pt"),
            conf_thresh=self.config.get("confidence_threshold", 0.35),
            target_classes=self.config.get("target_classes", [0, 2, 3, 5, 7])
        )
        self.kernel_size = 7  # Optimized kernel size for fast spatial filtering

    def process_video(self, input_path: pathlib.Path, output_path: pathlib.Path):
        if not input_path.exists():
            raise FileNotFoundError(f"Input file non-existent: {input_path}")
            
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        temp_raw_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        temp_raw_path = pathlib.Path(temp_raw_file.name)
        temp_raw_file.close()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(temp_raw_path), fourcc, fps, (width, height))

        logger.info(f"Processing {total_frames} frames with fast ROI policy from {input_path}")
        
        frame_idx = 0
        boxes = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection every 5th frame for max speed; reuse boxes in between
            if frame_idx % 5 == 0:
                boxes = self.detector.detect(frame)
                
            mask = ROIMapGenerator.generate_mask(frame.shape, boxes)
            
            # Fast spatial frequency attenuation on non-ROI regions
            blurred = cv2.GaussianBlur(frame, (self.kernel_size, self.kernel_size), 0)
            
            # Composite ROI and background using fast bitwise operations
            mask_3d = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            inv_mask_3d = cv2.bitwise_not(mask_3d)
            
            roi_part = cv2.bitwise_and(frame, mask_3d)
            bg_part = cv2.bitwise_and(blurred, inv_mask_3d)
            processed_frame = cv2.add(roi_part, bg_part)

            out.write(processed_frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames...")

        cap.release()
        out.release()
        
        logger.info("Spatial filtering completed. Encoding via FFmpeg...")
        
        cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_raw_path),
            "-c:v", "libx265",
            "-crf", str(self.config["quality_policy"].get("background_crf", 35)),
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-an",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        if temp_raw_path.exists():
            temp_raw_path.unlink()
        logger.info(f"Adaptive output successfully saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Adaptive Compression Engine")
    parser.add_argument("--input", required=True, type=pathlib.Path, help="Input video file")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Output target path")
    parser.add_argument("--config", default="config.yaml", type=pathlib.Path, help="Path to config file")
    args = parser.parse_args()

    compressor = AdaptiveCompressor(args.config)
    compressor.process_video(args.input, args.output)


if __name__ == "__main__":
    main()