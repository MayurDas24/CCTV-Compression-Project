"""Segment-based FFmpeg/libx265 encoder preserving every source frame."""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from frame_selector import Priority, TemporalPlan


LOGGER = logging.getLogger("VideoEncoder")


class VideoEncoderError(RuntimeError):
    """Raised when media probing, encoding, or validation fails."""


@dataclass(frozen=True, slots=True)
class EncodingConfig:
    """Video and audio encoding configuration."""

    codec: str
    preset: str
    pixel_format: str
    container: str
    priority_crf: Mapping[Priority, int]
    audio_enabled: bool
    audio_mode: str
    audio_bitrate: str | None
    movflags: str | None


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """External tool and output behavior."""

    ffmpeg_path: str
    ffprobe_path: str
    overwrite: bool
    keep_temporary_files: bool
    probe_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class OutputValidationConfig:
    """Allowed differences between source and final output."""

    fps_tolerance: float
    duration_tolerance_seconds: float


@dataclass(frozen=True, slots=True)
class VideoProperties:
    """Properties measured using ffprobe."""

    path: Path
    width: int
    height: int
    average_fps: float
    nominal_fps: float | None
    duration_seconds: float
    frame_count: int
    bitrate_bps: int | None
    codec: str | None
    pixel_format: str | None
    has_audio: bool
    audio_codec: str | None


@dataclass(frozen=True, slots=True)
class SegmentResult:
    """One encoded temporal quality segment."""

    index: int
    priority: Priority
    crf: int
    start_frame: int
    end_frame_exclusive: int
    frame_count: int
    output_path: Path


@dataclass(frozen=True, slots=True)
class EncodingResult:
    """Measured result of the complete Phase 4 encode."""

    source: VideoProperties
    output: VideoProperties
    segments: tuple[SegmentResult, ...]
    encoding_duration_seconds: float
    physically_removed_frames: int
    temporary_directory: str | None


class VideoEncoder:
    """Encode quality segments and concatenate them into one smooth video."""

    def __init__(
        self,
        encoding: EncodingConfig,
        runtime: RuntimeConfig,
        validation: OutputValidationConfig,
    ) -> None:
        self.encoding = encoding
        self.runtime = runtime
        self.validation = validation

        self.ffmpeg = _resolve_executable(
            runtime.ffmpeg_path,
            "FFmpeg executable not found. Install FFmpeg and ensure it is "
            "available on PATH.",
        )
        self.ffprobe = _resolve_executable(
            runtime.ffprobe_path,
            "FFprobe executable not found. Install FFmpeg and ensure ffprobe "
            "is available on PATH.",
        )

    def probe(self, path: Path) -> VideoProperties:
        """Inspect media and obtain a reliable frame count."""

        media_path = path.expanduser().resolve()

        if not media_path.is_file():
            raise VideoEncoderError(f"Video file not found: {media_path}")

        if media_path.stat().st_size <= 0:
            raise VideoEncoderError(f"Video file is empty: {media_path}")

        payload = self._run_ffprobe(media_path, count_frames=True)
        properties = _parse_probe_payload(payload, media_path)

        if properties.frame_count is None:
            raise VideoEncoderError(
                f"Unable to determine a reliable frame count for {media_path}"
            )

        return VideoProperties(
            path=properties.path,
            width=properties.width,
            height=properties.height,
            average_fps=properties.average_fps,
            nominal_fps=properties.nominal_fps,
            duration_seconds=properties.duration_seconds,
            frame_count=properties.frame_count,
            bitrate_bps=properties.bitrate_bps,
            codec=properties.codec,
            pixel_format=properties.pixel_format,
            has_audio=properties.has_audio,
            audio_codec=properties.audio_codec,
        )

    def encode(
        self,
        source: VideoProperties,
        output_path: Path,
        plan: TemporalPlan,
    ) -> EncodingResult:
        """Encode all plan intervals and atomically publish the final MP4."""

        output = output_path.expanduser().resolve()

        if source.path == output:
            raise VideoEncoderError(
                "Output video must not overwrite source video"
            )

        if output.exists() and not self.runtime.overwrite:
            raise VideoEncoderError(
                f"Output exists and overwrite is disabled: {output}"
            )

        try:
            output.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VideoEncoderError(
                f"Unable to create output directory {output.parent}: {exc}"
            ) from exc

        self._validate_encoder()

        temporary_context: tempfile.TemporaryDirectory[str] | None = None

        if self.runtime.keep_temporary_files:
            temporary_directory = Path(
                tempfile.mkdtemp(
                    prefix="phase4_",
                    dir=output.parent,
                )
            )
        else:
            temporary_context = tempfile.TemporaryDirectory(
                prefix="phase4_",
                dir=output.parent,
            )
            temporary_directory = Path(temporary_context.name)

        started_at = time.perf_counter()

        try:
            segment_results: list[SegmentResult] = []

            for index, interval in enumerate(plan.intervals):
                crf = self.encoding.priority_crf[interval.priority]
                segment_path = (
                    temporary_directory / f"segment_{index:06d}.ts"
                )

                LOGGER.info(
                    "Encoding segment %d/%d frames=%d-%d priority=%s CRF=%d",
                    index + 1,
                    len(plan.intervals),
                    interval.start_frame,
                    interval.end_frame_exclusive - 1,
                    interval.priority.value,
                    crf,
                )

                self._encode_segment(
                    source=source.path,
                    destination=segment_path,
                    start_frame=interval.start_frame,
                    end_frame_exclusive=interval.end_frame_exclusive,
                    crf=crf,
                    source_fps=source.average_fps,
                )

                segment_results.append(
                    SegmentResult(
                        index=index,
                        priority=interval.priority,
                        crf=crf,
                        start_frame=interval.start_frame,
                        end_frame_exclusive=interval.end_frame_exclusive,
                        frame_count=interval.frame_count,
                        output_path=segment_path,
                    )
                )

            concat_file = temporary_directory / "segments.ffconcat"
            _write_concat_file(concat_file, segment_results)

            final_temp = temporary_directory / (
                f"final{output.suffix or '.' + self.encoding.container}"
            )

            self._concatenate(
                concat_file=concat_file,
                source=source.path,
                source_has_audio=source.has_audio,
                destination=final_temp,
            )

            final_properties = self.probe(final_temp)
            self._validate_output(source, final_properties)

            os.replace(final_temp, output)

            duration = time.perf_counter() - started_at

            LOGGER.info(
                "Encoding completed in %.3f seconds",
                duration,
            )

            return EncodingResult(
                source=source,
                output=VideoProperties(
                    path=output,
                    width=final_properties.width,
                    height=final_properties.height,
                    average_fps=final_properties.average_fps,
                    nominal_fps=final_properties.nominal_fps,
                    duration_seconds=final_properties.duration_seconds,
                    frame_count=final_properties.frame_count,
                    bitrate_bps=final_properties.bitrate_bps,
                    codec=final_properties.codec,
                    pixel_format=final_properties.pixel_format,
                    has_audio=final_properties.has_audio,
                    audio_codec=final_properties.audio_codec,
                ),
                segments=tuple(segment_results),
                encoding_duration_seconds=duration,
                physically_removed_frames=(
                    source.frame_count - final_properties.frame_count
                ),
                temporary_directory=(
                    str(temporary_directory)
                    if self.runtime.keep_temporary_files
                    else None
                ),
            )
        except OSError as exc:
            raise VideoEncoderError(
                f"Encoding file operation failed: {exc}"
            ) from exc
        finally:
            if temporary_context is not None:
                temporary_context.cleanup()
            elif self.runtime.keep_temporary_files:
                LOGGER.info(
                    "Temporary files retained at %s",
                    temporary_directory,
                )

    def _encode_segment(
        self,
        *,
        source: Path,
        destination: Path,
        start_frame: int,
        end_frame_exclusive: int,
        crf: int,
        source_fps: float,
    ) -> None:
        fps_text = str(
            Fraction(source_fps).limit_denominator(1_000_000)
        )

        filter_expression = (
            f"trim=start_frame={start_frame}:"
            f"end_frame={end_frame_exclusive},"
            f"settb=AVTB,setpts=N/({fps_text}*TB)"
        )

        command = [
            self.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "warning",
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            filter_expression,
            "-c:v",
            self.encoding.codec,
            "-crf",
            str(crf),
            "-preset",
            self.encoding.preset,
            "-pix_fmt",
            self.encoding.pixel_format,
            "-r",
            fps_text,
            "-fps_mode:v",
            "cfr",
            "-map_metadata",
            "-1",
            "-f",
            "mpegts",
            "-y",
            str(destination),
        ]

        result = _run_process(command, timeout=None)

        if result.returncode != 0:
            raise VideoEncoderError(
                f"FFmpeg segment encoding failed for frames "
                f"{start_frame}-{end_frame_exclusive - 1}: "
                f"{_diagnostic_tail(result.stderr)}"
            )

        if not destination.is_file() or destination.stat().st_size <= 0:
            raise VideoEncoderError(
                f"Segment encoding produced no output for frames "
                f"{start_frame}-{end_frame_exclusive - 1}"
            )

    def _concatenate(
        self,
        *,
        concat_file: Path,
        source: Path,
        source_has_audio: bool,
        destination: Path,
    ) -> None:
        command = [
            self.ffmpeg,
            "-hide_banner",
            "-nostdin",
            "-loglevel",
            "warning",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_file),
            "-i",
            str(source),
            "-map",
            "0:v:0",
        ]

        if self.encoding.audio_enabled and source_has_audio:
            command.extend(["-map", "1:a:0?"])
        else:
            command.append("-an")

        command.extend(["-c:v", "copy"])

        if self.encoding.audio_enabled and source_has_audio:
            if self.encoding.audio_mode == "copy":
                command.extend(["-c:a", "copy"])
            else:
                command.extend(["-c:a", "aac"])

                if self.encoding.audio_bitrate:
                    command.extend(
                        ["-b:a", self.encoding.audio_bitrate]
                    )

        if (
            self.encoding.codec == "libx265"
            and self.encoding.container in {"mp4", "mov"}
        ):
            command.extend(["-tag:v", "hvc1"])

        if (
            self.encoding.movflags
            and self.encoding.container in {"mp4", "mov"}
        ):
            command.extend(
                ["-movflags", self.encoding.movflags]
            )

        command.extend(
            [
                "-map_metadata",
                "1",
                "-f",
                self.encoding.container,
                "-y",
                str(destination),
            ]
        )

        result = _run_process(command, timeout=None)

        if result.returncode != 0:
            raise VideoEncoderError(
                "FFmpeg segment concatenation failed: "
                f"{_diagnostic_tail(result.stderr)}"
            )

        if not destination.is_file() or destination.stat().st_size <= 0:
            raise VideoEncoderError(
                "FFmpeg produced no non-empty final output"
            )

    def _validate_output(
        self,
        source: VideoProperties,
        output: VideoProperties,
    ) -> None:
        errors: list[str] = []

        if (source.width, source.height) != (
            output.width,
            output.height,
        ):
            errors.append(
                "output resolution differs from source"
            )

        fps_difference = abs(
            source.average_fps - output.average_fps
        )

        if fps_difference > self.validation.fps_tolerance:
            errors.append(
                f"FPS difference {fps_difference:.6f} exceeds tolerance "
                f"{self.validation.fps_tolerance}"
            )

        duration_difference = abs(
            source.duration_seconds - output.duration_seconds
        )

        if (
            duration_difference
            > self.validation.duration_tolerance_seconds
        ):
            errors.append(
                f"duration difference {duration_difference:.6f}s exceeds "
                f"tolerance "
                f"{self.validation.duration_tolerance_seconds}s"
            )

        if source.frame_count != output.frame_count:
            errors.append(
                f"output frame count {output.frame_count} differs from "
                f"source frame count {source.frame_count}"
            )

        if errors:
            raise VideoEncoderError(
                "Output validation failed: " + "; ".join(errors)
            )

    def _validate_encoder(self) -> None:
        result = _run_process(
            [self.ffmpeg, "-hide_banner", "-encoders"],
            timeout=self.runtime.probe_timeout_seconds,
        )

        if result.returncode != 0:
            raise VideoEncoderError(
                "Unable to query FFmpeg encoders: "
                f"{_diagnostic_tail(result.stderr)}"
            )

        encoders = {
            fields[1]
            for line in result.stdout.splitlines()
            if len(fields := line.split()) >= 2
            and fields[0].startswith("V")
        }

        if self.encoding.codec not in encoders:
            raise VideoEncoderError(
                f"FFmpeg encoder '{self.encoding.codec}' is unavailable"
            )

    def _run_ffprobe(
        self,
        path: Path,
        *,
        count_frames: bool,
    ) -> Mapping[str, Any]:
        command = [
            self.ffprobe,
            "-v",
            "error",
        ]

        if count_frames:
            command.append("-count_frames")

        command.extend(
            [
                "-show_streams",
                "-show_format",
                "-print_format",
                "json",
                str(path),
            ]
        )

        result = _run_process(
            command,
            timeout=self.runtime.probe_timeout_seconds,
        )

        if result.returncode != 0:
            raise VideoEncoderError(
                f"ffprobe failed for {path}: "
                f"{_diagnostic_tail(result.stderr)}"
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise VideoEncoderError(
                f"ffprobe returned invalid JSON for {path}: {exc}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise VideoEncoderError(
                f"Invalid ffprobe response for {path}"
            )

        return payload


@dataclass(frozen=True, slots=True)
class _ProbeProperties:
    path: Path
    width: int
    height: int
    average_fps: float
    nominal_fps: float | None
    duration_seconds: float
    frame_count: int | None
    bitrate_bps: int | None
    codec: str | None
    pixel_format: str | None
    has_audio: bool
    audio_codec: str | None


def _parse_probe_payload(
    payload: Mapping[str, Any],
    path: Path,
) -> _ProbeProperties:
    streams = payload.get("streams")

    if not isinstance(streams, list):
        raise VideoEncoderError(
            f"ffprobe found no streams in {path}"
        )

    video = next(
        (
            item
            for item in streams
            if isinstance(item, Mapping)
            and item.get("codec_type") == "video"
        ),
        None,
    )

    if video is None:
        raise VideoEncoderError(
            f"No video stream found in {path}"
        )

    audio = next(
        (
            item
            for item in streams
            if isinstance(item, Mapping)
            and item.get("codec_type") == "audio"
        ),
        None,
    )

    width = _positive_integer(video.get("width"))
    height = _positive_integer(video.get("height"))
    average_fps = _parse_rate(video.get("avg_frame_rate"))

    if width is None or height is None or average_fps is None:
        raise VideoEncoderError(
            f"Invalid video stream metadata for {path}"
        )

    format_data = payload.get("format")

    if not isinstance(format_data, Mapping):
        format_data = {}

    duration = _positive_number(video.get("duration"))

    if duration is None:
        duration = _positive_number(format_data.get("duration"))

    if duration is None:
        raise VideoEncoderError(
            f"Unable to determine duration for {path}"
        )

    frame_count = _positive_integer(
        video.get("nb_read_frames")
    )

    bitrate = _positive_integer(format_data.get("bit_rate"))

    if bitrate is None:
        bitrate = _positive_integer(video.get("bit_rate"))

    return _ProbeProperties(
        path=path,
        width=width,
        height=height,
        average_fps=average_fps,
        nominal_fps=_parse_rate(video.get("r_frame_rate")),
        duration_seconds=duration,
        frame_count=frame_count,
        bitrate_bps=bitrate,
        codec=_text(video.get("codec_name")),
        pixel_format=_text(video.get("pix_fmt")),
        has_audio=audio is not None,
        audio_codec=(
            _text(audio.get("codec_name"))
            if audio is not None
            else None
        ),
    )


def _write_concat_file(
    path: Path,
    segments: list[SegmentResult],
) -> None:
    lines = ["ffconcat version 1.0"]

    for segment in segments:
        escaped = (
            segment.output_path.resolve()
            .as_posix()
            .replace("'", "'\\''")
        )
        lines.append(f"file '{escaped}'")

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _resolve_executable(value: str, message: str) -> str:
    candidate = Path(value).expanduser()

    if candidate.is_absolute() or candidate.parent != Path("."):
        if candidate.is_file():
            return str(candidate.resolve())

        raise VideoEncoderError(message)

    resolved = shutil.which(value)

    if resolved is None:
        raise VideoEncoderError(message)

    return resolved


def _run_process(
    command: list[str],
    *,
    timeout: float | None,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            shell=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise VideoEncoderError(
            f"Command timed out after {timeout} seconds: "
            f"{command[0]}"
        ) from exc
    except OSError as exc:
        raise VideoEncoderError(
            f"Unable to execute {command[0]}: {exc}"
        ) from exc


def _parse_rate(value: Any) -> float | None:
    if (
        not isinstance(value, str)
        or not value
        or value == "0/0"
    ):
        return None

    try:
        result = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None

    if not math.isfinite(result) or result <= 0.0:
        return None

    return result


def _positive_number(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result) or result <= 0.0:
        return None

    return result


def _positive_integer(value: Any) -> int | None:
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError):
        return None

    return result if result > 0 else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _diagnostic_tail(
    value: str,
    limit: int = 4000,
) -> str:
    cleaned = value.strip()

    return (
        cleaned[-limit:]
        if cleaned
        else "No diagnostic output was provided."
    )
