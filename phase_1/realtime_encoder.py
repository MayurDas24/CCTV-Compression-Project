import logging
import pathlib
import subprocess
import time
from typing import Optional

import psutil
import yaml

from rtsp_ingestion import RTSPStreamReader


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("RealtimeEncoder")


class RealtimePipeline:
    """
    Real-time RTSP -> frame processing -> H.265 encoding pipeline.

    Stream resolution and FPS are detected automatically from the
    incoming stream. The pipeline uses an intermediate MKV file
    during encoding and remuxes it to MP4 after successful shutdown.
    """

    def __init__(self, config_path: str = "config.yaml"):

        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.stream_reader = RTSPStreamReader(config_path)

        self.encoding_config = self.config["encoding"]
        self.monitoring_config = self.config["monitoring"]

        self.output_dir = pathlib.Path(
            self.encoding_config.get(
                "output_directory",
                "./logs/streams",
            )
        )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.process: Optional[subprocess.Popen] = None

        self.is_running = False

        self.input_width: Optional[int] = None
        self.input_height: Optional[int] = None
        self.input_fps: Optional[float] = None

        self.processed_frames = 0

        self.output_mkv: Optional[pathlib.Path] = None
        self.output_mp4: Optional[pathlib.Path] = None

    def _determine_stream_properties(self, first_frame):
        """Detect actual resolution and FPS."""

        frame_height, frame_width = first_frame.shape[:2]

        self.input_width = frame_width
        self.input_height = frame_height

        metadata = self.stream_reader.get_metadata()

        detected_fps = metadata.get("fps")

        configured_fps = self.encoding_config.get("fps")

        if detected_fps and detected_fps > 0:
            self.input_fps = float(detected_fps)

        elif configured_fps and configured_fps > 0:
            self.input_fps = float(configured_fps)

        else:
            self.input_fps = 25.0

        logger.info(
            "Detected input properties: %dx%d @ %.2f FPS",
            self.input_width,
            self.input_height,
            self.input_fps,
        )

    def _init_ffmpeg_pipe(
        self,
        output_filepath: pathlib.Path,
    ) -> subprocess.Popen:
        """Start FFmpeg H.265 encoder."""

        if not self.input_width or not self.input_height:
            raise RuntimeError(
                "Input resolution is unavailable."
            )

        if not self.input_fps:
            raise RuntimeError(
                "Input FPS is unavailable."
            )

        codec = self.encoding_config.get(
            "codec",
            "libx265",
        )

        preset = self.encoding_config.get(
            "preset",
            "ultrafast",
        )

        crf = self.encoding_config.get(
            "crf",
            28,
        )

        command = [
            "ffmpeg",

            "-hide_banner",

            "-loglevel",
            "warning",

            "-y",

            # Raw frames arrive through stdin.
            "-f",
            "rawvideo",

            "-pix_fmt",
            "bgr24",

            "-s",
            f"{self.input_width}x{self.input_height}",

            "-r",
            str(self.input_fps),

            "-i",
            "-",

            # Video only.
            "-an",

            # H.265 encoder.
            "-c:v",
            codec,

            "-preset",
            preset,

            "-crf",
            str(crf),

            "-pix_fmt",
            "yuv420p",

            # MKV is used while encoding so the recording
            # can be finalized reliably.
            "-f",
            "matroska",

            str(output_filepath),
        ]

        logger.info(
            "Starting FFmpeg encoder: "
            "%dx%d @ %.2f FPS | codec=%s | CRF=%s | preset=%s",
            self.input_width,
            self.input_height,
            self.input_fps,
            codec,
            crf,
            preset,
        )

        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _write_frame(self, frame):
        """Send one frame to FFmpeg."""

        if (
            self.process is None
            or self.process.stdin is None
        ):
            raise RuntimeError(
                "FFmpeg encoder is not running."
            )

        expected_shape = (
            self.input_height,
            self.input_width,
        )

        actual_shape = frame.shape[:2]

        if actual_shape != expected_shape:

            logger.warning(
                "Frame resolution changed from %s to %s. "
                "Resizing.",
                expected_shape,
                actual_shape,
            )

            import cv2

            frame = cv2.resize(
                frame,
                (
                    self.input_width,
                    self.input_height,
                ),
                interpolation=cv2.INTER_AREA,
            )

        try:

            self.process.stdin.write(
                frame.tobytes()
            )

            self.processed_frames += 1

        except (
            BrokenPipeError,
            OSError,
            ValueError,
        ) as error:

            logger.error(
                "FFmpeg input pipe failure: %s",
                error,
            )

            self.is_running = False

    def _finalize_output(self):
        """
        Wait for FFmpeg to finalize the MKV and then
        remux it into a standard MP4 container.
        """

        if self.process is None:
            return

        logger.info(
            "Finalizing FFmpeg output..."
        )

        try:

            return_code = self.process.wait(
                timeout=30
            )

        except subprocess.TimeoutExpired:

            logger.warning(
                "FFmpeg is taking longer than expected "
                "to finalize."
            )

            self.process.terminate()

            try:
                return_code = self.process.wait(
                    timeout=10
                )

            except subprocess.TimeoutExpired:

                logger.error(
                    "FFmpeg did not terminate. "
                    "Killing process."
                )

                self.process.kill()

                return_code = self.process.wait()

        if return_code != 0:

            logger.error(
                "FFmpeg exited with return code %s.",
                return_code,
            )

            return

        logger.info(
            "FFmpeg encoding completed successfully."
        )

        # Convert/remux MKV → MP4.
        if (
            self.output_mkv
            and self.output_mkv.exists()
        ):

            logger.info(
                "Creating finalized MP4: %s",
                self.output_mp4,
            )

            remux_command = [
                "ffmpeg",

                "-hide_banner",

                "-loglevel",
                "warning",

                "-y",

                "-i",
                str(self.output_mkv),

                "-c",
                "copy",

                "-movflags",
                "+faststart",

                str(self.output_mp4),
            ]

            result = subprocess.run(
                remux_command,
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:

                logger.info(
                    "MP4 finalized successfully: %s",
                    self.output_mp4,
                )

                # Remove intermediate MKV.
                try:
                    self.output_mkv.unlink()

                except OSError as error:

                    logger.warning(
                        "Could not remove intermediate MKV: %s",
                        error,
                    )

            else:

                logger.error(
                    "MP4 remux failed: %s",
                    result.stderr,
                )

    def run(
        self,
        duration_seconds: Optional[int] = None,
    ):
        """Run the real-time processing pipeline."""

        self.stream_reader.start()

        self.is_running = True

        logger.info(
            "Waiting for first frame to determine "
            "stream properties..."
        )

        status, first_frame = (
            self.stream_reader.read_frame(
                timeout=15.0
            )
        )

        if not status or first_frame is None:

            logger.error(
                "Unable to receive first frame "
                "from RTSP stream."
            )

            self.stop()

            return

        self._determine_stream_properties(
            first_frame
        )

        timestamp = int(time.time())

        self.output_mkv = (
            self.output_dir
            / f"stream_{timestamp}.mkv"
        )

        self.output_mp4 = (
            self.output_dir
            / f"stream_{timestamp}.mp4"
        )

        self.process = self._init_ffmpeg_pipe(
            self.output_mkv
        )

        start_time = time.time()

        last_metric_time = start_time

        logger.info(
            "Real-time processing pipeline started."
        )

        logger.info(
            "Output: %s",
            self.output_mp4,
        )

        try:

            # Process first frame.
            self._write_frame(
                first_frame
            )

            while self.is_running:

                status, frame = (
                    self.stream_reader.read_frame(
                        timeout=1.0
                    )
                )

                if (
                    not status
                    or frame is None
                ):
                    continue

                self._write_frame(frame)

                now = time.time()

                interval = (
                    self.monitoring_config.get(
                        "metrics_interval_seconds",
                        5,
                    )
                )

                if (
                    now - last_metric_time
                    >= interval
                ):

                    elapsed = (
                        now - start_time
                    )

                    encoded_fps = (
                        self.processed_frames
                        / elapsed
                        if elapsed > 0
                        else 0
                    )

                    cpu_percent = (
                        psutil.cpu_percent()
                    )

                    memory_percent = (
                        psutil.virtual_memory()
                        .percent
                    )

                    logger.info(
                        "[Metrics Monitor] "
                        "Elapsed: %ds | "
                        "Encoded FPS: %.2f | "
                        "CPU: %.1f%% | "
                        "RAM: %.1f%% | "
                        "Frames: %d",
                        int(elapsed),
                        encoded_fps,
                        cpu_percent,
                        memory_percent,
                        self.processed_frames,
                    )

                    last_metric_time = now

                if (
                    duration_seconds
                    and (
                        now - start_time
                        >= duration_seconds
                    )
                ):

                    logger.info(
                        "Target duration reached."
                    )

                    break

        except KeyboardInterrupt:

            logger.info(
                "Shutdown requested by user."
            )

        except Exception as error:

            logger.exception(
                "Pipeline error: %s",
                error,
            )

        finally:

            self.stop()

    def stop(self):
        """Gracefully stop ingestion and encoder."""

        if not self.is_running and self.process is None:
            return

        self.is_running = False

        logger.info(
            "Stopping RTSP ingestion..."
        )

        self.stream_reader.stop()

        if self.process:

            if self.process.stdin:

                try:

                    # Closing stdin tells FFmpeg that
                    # no more frames are coming.
                    self.process.stdin.close()

                except Exception:
                    pass

            # Allow FFmpeg to finish and finalize
            # the recording.
            self._finalize_output()

            self.process = None

        logger.info(
            "Pipeline stopped. "
            "Total frames encoded: %d",
            self.processed_frames,
        )


if __name__ == "__main__":

    pipeline = RealtimePipeline(
        "config.yaml"
    )

    pipeline.run(
        duration_seconds=15
    )
