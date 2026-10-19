"""Input, schema, media, and cross-phase validation for Phase 5."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping


LOGGER = logging.getLogger("Validation")


class ValidationError(RuntimeError):
    """Fatal validation error that prevents trustworthy reporting."""


@dataclass(frozen=True, slots=True)
class DecisionStatistics:
    """Validated aggregate facts from Phase 3 decision JSONL."""

    total_lines: int
    nonempty_lines: int
    total_decision_records: int
    retained_frames: int
    dropped_frames: int
    malformed_records: int
    first_frame_index: int | None
    last_frame_index: int | None
    minimum_timestamp_seconds: float | None
    maximum_timestamp_seconds: float | None
    priority_distribution: dict[str, int]
    reason_distribution: dict[str, int]

    @property
    def retention_ratio(self) -> float:
        if self.total_decision_records == 0:
            return 0.0
        return self.retained_frames / self.total_decision_records

    @property
    def drop_ratio(self) -> float:
        if self.total_decision_records == 0:
            return 0.0
        return self.dropped_frames / self.total_decision_records


@dataclass(frozen=True, slots=True)
class MediaInformation:
    """Media properties measured by ffprobe."""

    path: Path
    size_bytes: int
    width: int
    height: int
    average_fps: float | None
    nominal_fps: float | None
    duration_seconds: float | None
    frame_count: int | None
    codec: str | None
    pixel_format: str | None
    bitrate_bps: int | None
    has_audio: bool
    audio_codec: str | None
    frame_count_method: str | None = None


@dataclass(frozen=True, slots=True)
class Phase4Summary:
    """Validated Phase 4 summary with its raw compatible payload."""

    path: Path
    schema_version: str
    phase: str
    status: str
    payload: Mapping[str, Any]
    encoding_duration_seconds: float | None


class DecisionValidator:
    """Stream and validate Phase 3 decisions using bounded memory."""

    def __init__(self, *, fail_on_malformed: bool) -> None:
        self.fail_on_malformed = fail_on_malformed

    def validate(self, path: Path) -> DecisionStatistics:
        decision_path = _required_file(path, "Phase 3 decision file")
        total_lines = 0
        nonempty_lines = 0
        valid_records = 0
        retained = 0
        dropped = 0
        malformed = 0
        first_index: int | None = None
        previous_index: int | None = None
        minimum_timestamp: float | None = None
        maximum_timestamp: float | None = None
        priority_distribution: dict[str, int] = {}
        reason_distribution: dict[str, int] = {}

        try:
            with decision_path.open(
                "r",
                encoding="utf-8-sig",
                errors="strict",
            ) as stream:
                for line_number, line in enumerate(stream, start=1):
                    total_lines += 1
                    stripped = line.strip()

                    if not stripped:
                        continue

                    nonempty_lines += 1

                    try:
                        record = json.loads(stripped)
                        frame_index, decision, timestamp = (
                            self._validate_record(record)
                        )
                    except (json.JSONDecodeError, ValueError) as exc:
                        if self.fail_on_malformed:
                            raise ValidationError(
                                f"Invalid Phase 3 decision at line "
                                f"{line_number}: {exc}"
                            ) from exc

                        malformed += 1
                        LOGGER.warning(
                            "Skipping malformed Phase 3 record at line %d: %s",
                            line_number,
                            exc,
                        )
                        continue

                    if previous_index is not None:
                        if frame_index == previous_index:
                            raise ValidationError(
                                "Duplicate Phase 3 decision for frame_index "
                                f"{frame_index} at line {line_number}"
                            )

                        if frame_index < previous_index:
                            raise ValidationError(
                                "Phase 3 decisions are out of order at line "
                                f"{line_number}: {frame_index} follows "
                                f"{previous_index}"
                            )

                    if first_index is None:
                        first_index = frame_index

                    previous_index = frame_index
                    valid_records += 1

                    if decision == "retain":
                        retained += 1
                    else:
                        dropped += 1

                    if timestamp is not None:
                        minimum_timestamp = (
                            timestamp
                            if minimum_timestamp is None
                            else min(minimum_timestamp, timestamp)
                        )
                        maximum_timestamp = (
                            timestamp
                            if maximum_timestamp is None
                            else max(maximum_timestamp, timestamp)
                        )

                    priority = _optional_nonempty_text(
                        record.get("priority")
                    )
                    if priority is not None:
                        priority_distribution[priority] = (
                            priority_distribution.get(priority, 0) + 1
                        )

                    reason = _optional_nonempty_text(
                        record.get("reason")
                    )
                    if reason is not None:
                        reason_distribution[reason] = (
                            reason_distribution.get(reason, 0) + 1
                        )
        except UnicodeError as exc:
            raise ValidationError(
                f"Phase 3 decision file is not valid UTF-8: {decision_path}"
            ) from exc
        except OSError as exc:
            raise ValidationError(
                f"Unable to read Phase 3 decisions {decision_path}: {exc}"
            ) from exc

        if nonempty_lines == 0:
            raise ValidationError(
                f"Phase 3 decision file is empty: {decision_path}"
            )

        if valid_records == 0:
            raise ValidationError(
                "Phase 3 decision file contains no valid records"
            )

        if retained + dropped != valid_records:
            raise ValidationError(
                "Internal Phase 3 decision counts are inconsistent"
            )

        result = DecisionStatistics(
            total_lines=total_lines,
            nonempty_lines=nonempty_lines,
            total_decision_records=valid_records,
            retained_frames=retained,
            dropped_frames=dropped,
            malformed_records=malformed,
            first_frame_index=first_index,
            last_frame_index=previous_index,
            minimum_timestamp_seconds=minimum_timestamp,
            maximum_timestamp_seconds=maximum_timestamp,
            priority_distribution=dict(
                sorted(priority_distribution.items())
            ),
            reason_distribution=dict(
                sorted(reason_distribution.items())
            ),
        )

        LOGGER.info(
            "Validated %d Phase 3 decisions: retain=%d drop=%d malformed=%d",
            result.total_decision_records,
            result.retained_frames,
            result.dropped_frames,
            result.malformed_records,
        )

        return result

    @staticmethod
    def _validate_record(
        record: Any,
    ) -> tuple[int, str, float | None]:
        if not isinstance(record, Mapping):
            raise ValueError("JSON value must be an object")

        frame_index = _nonnegative_integer(
            record.get("frame_index"),
            "frame_index",
        )

        raw_decision = record.get("decision")
        if not isinstance(raw_decision, str):
            raise ValueError("decision must be 'retain' or 'drop'")

        decision = raw_decision.strip().lower()
        if decision not in {"retain", "drop"}:
            raise ValueError("decision must be 'retain' or 'drop'")

        timestamp: float | None = None
        if "timestamp_seconds" in record:
            timestamp = _nonnegative_number(
                record.get("timestamp_seconds"),
                "timestamp_seconds",
            )

        return frame_index, decision, timestamp


class Phase4SummaryValidator:
    """Load the existing Phase 4 summary without changing its schema."""

    def load(self, path: Path) -> Phase4Summary:
        summary_path = _required_file(
            path,
            "Phase 4 compression summary",
        )

        try:
            with summary_path.open(
                "r",
                encoding="utf-8-sig",
            ) as stream:
                payload = json.load(stream)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"Invalid Phase 4 summary JSON {summary_path}: {exc}"
            ) from exc
        except (OSError, UnicodeError) as exc:
            raise ValidationError(
                f"Unable to read Phase 4 summary {summary_path}: {exc}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise ValidationError(
                "Phase 4 summary root must be a JSON object"
            )

        schema_version = _required_text(payload, "schema_version")
        phase = _required_text(payload, "phase")
        status = _required_text(payload, "status")

        if schema_version != "1.0":
            raise ValidationError(
                f"Unsupported Phase 4 schema_version: {schema_version}"
            )

        if phase != "phase_4_intelligent_compression":
            raise ValidationError(
                f"Unexpected Phase 4 phase identifier: {phase}"
            )

        if status not in {
            "completed",
            "completed_with_warnings",
        }:
            raise ValidationError(
                "Phase 4 summary does not describe a completed run: "
                f"{status}"
            )

        performance = payload.get("performance")
        encoding_duration: float | None = None

        if isinstance(performance, Mapping):
            encoding_duration = _optional_positive_number(
                performance.get("encoding_duration_seconds")
            )

        LOGGER.info(
            "Validated Phase 4 summary: %s",
            summary_path,
        )

        return Phase4Summary(
            path=summary_path,
            schema_version=schema_version,
            phase=phase,
            status=status,
            payload=payload,
            encoding_duration_seconds=encoding_duration,
        )


class MediaInspector:
    """Inspect media and measure decoded frame counts with ffprobe."""

    def __init__(
        self,
        ffprobe_path: str,
        timeout_seconds: float,
    ) -> None:
        self.ffprobe = _resolve_executable(
            ffprobe_path,
            "FFprobe executable not found. Install FFmpeg and ensure "
            "ffprobe is available on PATH.",
        )
        self.timeout_seconds = timeout_seconds

    def inspect(
        self,
        path: Path,
        label: str,
    ) -> MediaInformation:
        media_path = _required_file(path, label)

        normal_payload = self._run_ffprobe(
            [
                self.ffprobe,
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-print_format",
                "json",
                str(media_path),
            ],
            media_path=media_path,
            label=label,
            operation="reading stream metadata",
        )

        information = _parse_media_payload(
            normal_payload,
            media_path,
        )

        metadata_frame_count = information.frame_count
        frame_count, count_error = self._count_video_frames(
            media_path,
            label,
        )

        if frame_count is not None:
            if (
                metadata_frame_count is not None
                and metadata_frame_count != frame_count
            ):
                LOGGER.warning(
                    "%s container frame count %d differs from "
                    "decoded frame count %d; using decoded count",
                    label,
                    metadata_frame_count,
                    frame_count,
                )

            information = replace(
                information,
                frame_count=frame_count,
                frame_count_method="ffprobe_count_frames",
            )
        elif metadata_frame_count is not None:
            information = replace(
                information,
                frame_count_method="container_metadata_fallback",
            )
            LOGGER.warning(
                "%s decoded frame count is unavailable; retaining "
                "container metadata for reporting only: %s",
                label,
                count_error
                or "no measured frame count was returned",
            )
        else:
            information = replace(
                information,
                frame_count_method="unavailable",
            )
            LOGGER.warning(
                "%s frame count is unavailable from container "
                "metadata and decoded frame counting: %s",
                label,
                count_error
                or "no measured frame count was returned",
            )

        LOGGER.info(
            "%s: %dx%d fps=%s duration=%s frames=%s "
            "frame_count_method=%s codec=%s",
            label,
            information.width,
            information.height,
            _display(information.average_fps),
            _display(information.duration_seconds),
            _display_integer(information.frame_count),
            information.frame_count_method or "unavailable",
            information.codec or "unknown",
        )

        return information

    def _count_video_frames(
        self,
        media_path: Path,
        label: str,
    ) -> tuple[int | None, str | None]:
        """Count decoded video frames without estimating from duration."""

        command = [
            self.ffprobe,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-print_format",
            "json",
            str(media_path),
        ]

        try:
            payload = self._run_ffprobe(
                command,
                media_path=media_path,
                label=label,
                operation="counting decoded video frames",
            )
        except ValidationError as exc:
            return None, str(exc)

        streams = payload.get("streams")

        if not isinstance(streams, list) or not streams:
            return (
                None,
                "ffprobe frame-count output contains no video stream",
            )

        video_stream = streams[0]

        if not isinstance(video_stream, Mapping):
            return (
                None,
                "ffprobe frame-count video stream is malformed",
            )

        frame_count = _optional_positive_integer(
            video_stream.get("nb_read_frames")
        )

        if frame_count is None:
            return (
                None,
                "ffprobe did not return a positive "
                "nb_read_frames value",
            )

        return frame_count, None

    def _run_ffprobe(
        self,
        command: list[str],
        *,
        media_path: Path,
        label: str,
        operation: str,
    ) -> Mapping[str, Any]:
        """Execute one controlled ffprobe operation."""

        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise ValidationError(
                f"ffprobe timed out after {self.timeout_seconds} "
                f"seconds while {operation} for {label}: {media_path}"
            ) from exc
        except OSError as exc:
            raise ValidationError(
                f"Unable to execute ffprobe for {label}: {exc}"
            ) from exc

        if result.returncode != 0:
            raise ValidationError(
                f"ffprobe failed while {operation} for {label} "
                f"{media_path}: {_diagnostic_tail(result.stderr)}"
            )

        if not result.stdout.strip():
            raise ValidationError(
                f"ffprobe returned empty output while {operation} "
                f"for {label}"
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise ValidationError(
                f"ffprobe returned invalid JSON while {operation} "
                f"for {label}: {exc}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise ValidationError(
                "ffprobe returned a non-object JSON value while "
                f"{operation} for {label}"
            )

        return payload


def validate_cross_phase_consistency(
    decisions: DecisionStatistics,
    phase4: Phase4Summary,
    source: MediaInformation,
    compressed: MediaInformation,
    configured_compressed_path: Path,
) -> list[str]:
    """Return nonfatal discrepancies between independently measured data."""

    warnings: list[str] = []
    phase4_decisions = phase4.payload.get("decisions")

    if isinstance(phase4_decisions, Mapping):
        comparisons = (
            (
                "frames_analyzed",
                decisions.total_decision_records,
            ),
            (
                "frames_retained",
                decisions.retained_frames,
            ),
            (
                "frames_dropped",
                decisions.dropped_frames,
            ),
        )

        for key, actual in comparisons:
            stated = _optional_nonnegative_integer(
                phase4_decisions.get(key)
            )

            if stated is None:
                warnings.append(
                    f"Phase 4 summary does not provide decisions.{key}"
                )
            elif stated != actual:
                warnings.append(
                    f"Phase 4 decisions.{key}={stated}, but Phase 3 "
                    f"validation measured {actual}"
                )
    else:
        warnings.append(
            "Phase 4 summary has no usable decisions object"
        )

    outputs = phase4.payload.get("outputs")

    if isinstance(outputs, Mapping):
        stated_output = _optional_nonempty_text(
            outputs.get("video")
        )

        if stated_output is not None:
            stated_path = Path(stated_output).expanduser()

            if not stated_path.is_absolute():
                stated_path = phase4.path.parent / stated_path

            if (
                stated_path.resolve()
                != configured_compressed_path.resolve()
            ):
                warnings.append(
                    "Configured compressed video differs from the "
                    "video path recorded in the Phase 4 summary"
                )

    storage = phase4.payload.get("storage")

    if isinstance(storage, Mapping):
        stated_original = _optional_nonnegative_integer(
            storage.get("original_bytes")
        )
        stated_compressed = _optional_nonnegative_integer(
            storage.get("compressed_bytes")
        )

        if (
            stated_original is not None
            and stated_original != source.size_bytes
        ):
            warnings.append(
                f"Phase 4 original_bytes={stated_original}, but "
                f"current source file size is {source.size_bytes}"
            )

        if (
            stated_compressed is not None
            and stated_compressed != compressed.size_bytes
        ):
            warnings.append(
                f"Phase 4 compressed_bytes={stated_compressed}, but "
                f"current compressed file size is "
                f"{compressed.size_bytes}"
            )

    if source.frame_count is None:
        warnings.append(
            "Source frame count is unavailable"
        )
    elif source.frame_count_method != "ffprobe_count_frames":
        warnings.append(
            "Source frame count uses container metadata fallback; "
            "decoded frame count is unavailable"
        )

    if compressed.frame_count is None:
        warnings.append(
            "Compressed frame count is unavailable"
        )
    elif compressed.frame_count_method != "ffprobe_count_frames":
        warnings.append(
            "Compressed frame count uses container metadata fallback; "
            "decoded frame count is unavailable"
        )

    return warnings


def _parse_media_payload(
    payload: Any,
    path: Path,
) -> MediaInformation:
    if not isinstance(payload, Mapping):
        raise ValidationError(
            f"Invalid ffprobe response for {path}"
        )

    streams = payload.get("streams")

    if not isinstance(streams, list):
        raise ValidationError(
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
        raise ValidationError(
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

    width = _optional_positive_integer(video.get("width"))
    height = _optional_positive_integer(video.get("height"))

    if width is None or height is None:
        raise ValidationError(
            f"Invalid video dimensions reported for {path}"
        )

    format_data = payload.get("format")

    if not isinstance(format_data, Mapping):
        format_data = {}

    duration = _optional_positive_number(
        video.get("duration")
    )

    if duration is None:
        duration = _optional_positive_number(
            format_data.get("duration")
        )

    bitrate = _optional_positive_integer(
        format_data.get("bit_rate")
    )

    if bitrate is None:
        bitrate = _optional_positive_integer(
            video.get("bit_rate")
        )

    frame_count = _optional_positive_integer(
        video.get("nb_frames")
    )

    return MediaInformation(
        path=path,
        size_bytes=path.stat().st_size,
        width=width,
        height=height,
        average_fps=_parse_rate(
            video.get("avg_frame_rate")
        ),
        nominal_fps=_parse_rate(
            video.get("r_frame_rate")
        ),
        duration_seconds=duration,
        frame_count=frame_count,
        codec=_optional_nonempty_text(
            video.get("codec_name")
        ),
        pixel_format=_optional_nonempty_text(
            video.get("pix_fmt")
        ),
        bitrate_bps=bitrate,
        has_audio=audio is not None,
        audio_codec=(
            _optional_nonempty_text(
                audio.get("codec_name")
            )
            if audio is not None
            else None
        ),
        frame_count_method=(
            "container_metadata"
            if frame_count is not None
            else None
        ),
    )


def _required_file(
    path: Path,
    label: str,
) -> Path:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise ValidationError(
            f"{label} not found: {resolved}"
        )

    try:
        size = resolved.stat().st_size
    except OSError as exc:
        raise ValidationError(
            f"Unable to inspect {label}: {exc}"
        ) from exc

    if size <= 0:
        raise ValidationError(
            f"{label} is empty: {resolved}"
        )

    return resolved


def _resolve_executable(
    value: str,
    message: str,
) -> str:
    candidate = Path(value).expanduser()

    if candidate.is_absolute() or candidate.parent != Path("."):
        if candidate.is_file():
            return str(candidate.resolve())

        raise ValidationError(message)

    resolved = shutil.which(value)

    if resolved is None:
        raise ValidationError(message)

    return resolved


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


def _required_text(
    mapping: Mapping[str, Any],
    key: str,
) -> str:
    value = mapping.get(key)

    if not isinstance(value, str) or not value.strip():
        raise ValidationError(
            f"Phase 4 summary field '{key}' is required"
        )

    return value.strip()


def _optional_nonempty_text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def _nonnegative_integer(
    value: Any,
    field: str,
) -> int:
    result = _optional_nonnegative_integer(value)

    if result is None:
        raise ValueError(
            f"{field} must be a nonnegative integer"
        )

    return result


def _optional_nonnegative_integer(
    value: Any,
) -> int | None:
    if isinstance(value, bool):
        return None

    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        not math.isfinite(number)
        or not number.is_integer()
        or number < 0
    ):
        return None

    return int(number)


def _optional_positive_integer(
    value: Any,
) -> int | None:
    result = _optional_nonnegative_integer(value)

    if result is None or result <= 0:
        return None

    return result


def _nonnegative_number(
    value: Any,
    field: str,
) -> float:
    if isinstance(value, bool):
        raise ValueError(
            f"{field} must be a nonnegative number"
        )

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a nonnegative number"
        ) from exc

    if not math.isfinite(result) or result < 0.0:
        raise ValueError(
            f"{field} must be a nonnegative number"
        )

    return result


def _optional_positive_number(
    value: Any,
) -> float | None:
    if isinstance(value, bool):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result) or result <= 0.0:
        return None

    return result


def _diagnostic_tail(
    value: str,
    limit: int = 4000,
) -> str:
    cleaned = value.strip()

    if not cleaned:
        return "No diagnostic output was provided."

    return cleaned[-limit:]


def _display(value: float | None) -> str:
    if value is None:
        return "unavailable"

    return f"{value:.6g}"


def _display_integer(value: int | None) -> str:
    if value is None:
        return "unavailable"

    return str(value)
