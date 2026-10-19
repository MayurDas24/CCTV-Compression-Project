import logging
import os
import queue
import threading
import time
from typing import Optional, Tuple

import cv2
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("RTSPIngestion")


class RTSPStreamReader:
    """
    RTSP stream reader with:
    - automatic connection/reconnection
    - bounded frame buffer
    - dropped-frame handling
    - stream metadata detection
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            raw_yaml = f.read()

        expanded_yaml = os.path.expandvars(raw_yaml)
        full_config = yaml.safe_load(expanded_yaml)

        self.config = full_config["stream"]

        self.rtsp_url = self.config["rtsp_url"]
        self.reconnect_attempts = self.config.get("reconnect_attempts", 10)
        self.reconnect_delay = self.config.get("reconnect_delay_seconds", 3)
        self.max_queue_size = self.config.get("frame_buffer_size", 60)

        self.frame_queue = queue.Queue(maxsize=self.max_queue_size)

        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.cap: Optional[cv2.VideoCapture] = None

        self.total_frames_read = 0
        self.dropped_frames = 0

        self.width: Optional[int] = None
        self.height: Optional[int] = None
        self.fps: Optional[float] = None

    def start(self):
        """Start the background frame-capture thread."""
        if self.is_running:
            logger.warning("RTSP ingestion is already running.")
            return

        self.is_running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="RTSPCaptureThread",
        )

        self.thread.start()

        logger.info("RTSP ingestion thread initialized.")

    def _connect(self) -> bool:
        """Attempt to establish an RTSP connection."""
        attempts = 0

        while attempts < self.reconnect_attempts and self.is_running:
            sanitized_url = (
                self.rtsp_url.split("@")[-1]
                if "@" in self.rtsp_url
                else self.rtsp_url
            )

            logger.info(
                "Attempting connection to stream endpoint: ...%s "
                "(Attempt %d/%d)",
                sanitized_url,
                attempts + 1,
                self.reconnect_attempts,
            )

            self.cap = cv2.VideoCapture(self.rtsp_url)

            if self.cap.isOpened():
                self._read_stream_metadata()

                logger.info(
                    "RTSP connection established. "
                    "Resolution: %sx%s | FPS: %.2f",
                    self.width,
                    self.height,
                    self.fps if self.fps else 0.0,
                )

                return True

            attempts += 1

            if attempts < self.reconnect_attempts:
                time.sleep(self.reconnect_delay)

        return False

    def _read_stream_metadata(self):
        """Read metadata reported by OpenCV."""
        if self.cap is None:
            return

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.width = width if width > 0 else None
        self.height = height if height > 0 else None
        self.fps = fps if fps > 0 else None

    def _capture_loop(self):
        """Continuously capture frames from the RTSP stream."""
        while self.is_running:

            if self.cap is None or not self.cap.isOpened():

                if not self._connect():
                    logger.error(
                        "Failed to establish RTSP connection "
                        "after maximum retry attempts."
                    )

                    self.is_running = False
                    break

            ret, frame = self.cap.read()

            if not ret or frame is None:
                logger.warning(
                    "Frame read failure / stream interrupted. "
                    "Attempting reconnection..."
                )

                if self.cap:
                    self.cap.release()

                self.cap = None

                time.sleep(self.reconnect_delay)
                continue

            # Detect actual frame dimensions from the received frame.
            if self.width is None or self.height is None:
                self.height, self.width = frame.shape[:2]

            self.total_frames_read += 1

            try:
                self.frame_queue.put_nowait(frame)

            except queue.Full:
                # Drop the oldest frame rather than allowing
                # latency to grow indefinitely.
                try:
                    self.frame_queue.get_nowait()
                    self.dropped_frames += 1

                    self.frame_queue.put_nowait(frame)

                except queue.Empty:
                    pass

    def read_frame(
        self,
        timeout: float = 1.0,
    ) -> Tuple[bool, Optional[object]]:
        """Return the next available frame."""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return True, frame

        except queue.Empty:
            return False, None

    def get_metadata(self) -> dict:
        """Return detected stream metadata."""
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }

    def stop(self):
        """Stop capture and release resources."""
        self.is_running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        logger.info(
            "Stream ingestion stopped. Read: %d | Dropped: %d",
            self.total_frames_read,
            self.dropped_frames,
        )


if __name__ == "__main__":
    reader = RTSPStreamReader("config.yaml")

    reader.start()

    try:
        start_time = time.time()

        while time.time() - start_time < 10:
            status, frame = reader.read_frame()

            if status:
                logger.info(
                    "Frame received | Shape: %s",
                    frame.shape,
                )

    finally:
        reader.stop()