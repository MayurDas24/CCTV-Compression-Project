"""Read-only cross-phase loading, media inspection, and consistency checks."""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping


LOGGER = logging.getLogger("Phase6Integration")


class IntegrationError(RuntimeError):
    """Raised when a required artifact cannot be safely inspected."""


@dataclass(frozen=True, slots=True)
class IntegrationPaths:
    source_video: Path
    phase3_decisions: Path
    phase4_compressed_video: Path
    phase4_summary: Path
    phase5_report_json: Path
    phase5_report_text: Path


@dataclass(frozen=True, slots=True)
class ValidationTolerances:
    fps: float
    duration_seconds: float
    compression_ratio: float
    storage_reduction_percent: float


@dataclass(frozen=True, slots=True)
class Phase3Statistics:
    path: Path
    total_records: int
    retained: int
    dropped: int
    malformed: int
    priority_distribution: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_file": str(self.path),
            "total_decision_records": self.total_records,
            "retained_decisions": self.retained,
            "dropped_decisions": self.dropped,
            "malformed_records": self.malformed,
            "priority_distribution": self.priority_distribution,
        }


@dataclass(frozen=True, slots=True)
class MediaMetadata:
    path: Path
    size_bytes: int
    width: int
    height: int
    average_fps: float
    duration_seconds: float
    frame_count: int
    frame_count_method: str
    codec: str | None
    pixel_format: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "average_fps": round(self.average_fps, 6),
            "duration_seconds": round(self.duration_seconds, 6),
            "frame_count": self.frame_count,
            "frame_count_method": self.frame_count_method,
            "codec": self.codec,
            "pixel_format": self.pixel_format,
        }


@dataclass(frozen=True, slots=True)
class ConsistencyCheck:
    name: str
    category: str
    status: str
    message: str
    expected: Any = None
    actual: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "category": self.category,
            "status": self.status,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True, slots=True)
class LoadedPipelineData:
    paths: IntegrationPaths
    phase3: Phase3Statistics
    phase4: Mapping[str, Any]
    phase5: Mapping[str, Any]
    source_media: MediaMetadata
    compressed_media: MediaMetadata


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    data: LoadedPipelineData
    checks: tuple[ConsistencyCheck, ...]

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(
            check.message
            for check in self.checks
            if check.status == "error"
        )

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(
            check.message
            for check in self.checks
            if check.status == "warning"
        )

    @property
    def successful_checks(self) -> int:
        return sum(
            check.status == "passed"
            for check in self.checks
        )

    @property
    def valid(self) -> bool:
        return not self.errors


class MediaInspector:
    """Measure final media properties with ffprobe."""

    def __init__(
        self,
        ffprobe_path: str,
        timeout_seconds: float,
    ) -> None:
        self.ffprobe = _resolve_executable(ffprobe_path)
        self.timeout_seconds = timeout_seconds

    def inspect(
        self,
        path: Path,
        label: str,
    ) -> MediaMetadata:
        media_path = _required_file(path, label)

        payload = self._probe(
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
            label,
        )

        streams = payload.get("streams")

        if not isinstance(streams, list):
            raise IntegrationError(
                f"ffprobe found no streams in {label}"
            )

        video = next(
            (
                stream
                for stream in streams
                if isinstance(stream, Mapping)
                and stream.get("codec_type") == "video"
            ),
            None,
        )

        if video is None:
            raise IntegrationError(
                f"No video stream found in {label}"
            )

        width = _positive_int(video.get("width"))
        height = _positive_int(video.get("height"))
        fps = _rate(video.get("avg_frame_rate"))

        format_data = payload.get("format")
        if not isinstance(format_data, Mapping):
            format_data = {}

        duration = _positive_float(video.get("duration"))
        if duration is None:
            duration = _positive_float(
                format_data.get("duration")
            )

        if (
            width is None
            or height is None
            or fps is None
            or duration is None
        ):
            raise IntegrationError(
                f"Incomplete video metadata for {label}"
            )

        frame_count = self._count_frames(
            media_path,
            label,
        )
        method = "ffprobe_count_frames"

        return MediaMetadata(
            path=media_path,
            size_bytes=media_path.stat().st_size,
            width=width,
            height=height,
            average_fps=fps,
            duration_seconds=duration,
            frame_count=frame_count,
            frame_count_method=method,
            codec=_text(video.get("codec_name")),
            pixel_format=_text(video.get("pix_fmt")),
        )

    def _count_frames(
        self,
        path: Path,
        label: str,
    ) -> int:
        payload = self._probe(
            [
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
                str(path),
            ],
            f"{label} frame count",
        )

        streams = payload.get("streams")

        if (
            not isinstance(streams, list)
            or not streams
            or not isinstance(streams[0], Mapping)
        ):
            raise IntegrationError(
                f"Unable to count frames in {label}"
            )

        count = _positive_int(
            streams[0].get("nb_read_frames")
        )

        if count is None:
            raise IntegrationError(
                f"ffprobe returned no measured frame count for {label}"
            )

        return count

    def _probe(
        self,
        command: list[str],
        label: str,
    ) -> Mapping[str, Any]:
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
            raise IntegrationError(
                f"ffprobe timed out while inspecting {label}"
            ) from exc
        except OSError as exc:
            raise IntegrationError(
                f"Unable to execute ffprobe for {label}: {exc}"
            ) from exc

        if result.returncode != 0:
            raise IntegrationError(
                f"ffprobe failed for {label}: "
                f"{_tail(result.stderr)}"
            )

        if not result.stdout.strip():
            raise IntegrationError(
                f"ffprobe returned empty output for {label}"
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise IntegrationError(
                f"ffprobe returned invalid JSON for {label}: {exc}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise IntegrationError(
                f"ffprobe returned invalid data for {label}"
            )

        return payload


class IntegrationValidator:
    """Load stable phase outputs and perform consistency checks."""

    def __init__(
        self,
        paths: IntegrationPaths,
        tolerances: ValidationTolerances,
        ffprobe_path: str,
        probe_timeout_seconds: float,
    ) -> None:
        self.paths = paths
        self.tolerances = tolerances
        self.inspector = MediaInspector(
            ffprobe_path,
            probe_timeout_seconds,
        )

    def validate(self) -> IntegrationResult:
        self._validate_required_files()

        phase3 = _load_phase3(
            self.paths.phase3_decisions
        )
        phase4 = _load_report(
            self.paths.phase4_summary,
            "Phase 4 summary",
            "phase_4_intelligent_compression",
        )
        phase5 = _load_report(
            self.paths.phase5_report_json,
            "Phase 5 report",
            "phase_5_quality_forensic_validation",
        )

        source = self.inspector.inspect(
            self.paths.source_video,
            "source video",
        )
        compressed = self.inspector.inspect(
            self.paths.phase4_compressed_video,
            "Phase 4 compressed video",
        )

        data = LoadedPipelineData(
            paths=self.paths,
            phase3=phase3,
            phase4=phase4,
            phase5=phase5,
            source_media=source,
            compressed_media=compressed,
        )

        checks: list[ConsistencyCheck] = []

        self._check_statuses(data, checks)
        self._check_decisions(data, checks)
        self._check_media(data, checks)
        self._check_compression(data, checks)
        self._check_quality(data, checks)

        result = IntegrationResult(
            data=data,
            checks=tuple(checks),
        )

        LOGGER.info(
            "Integration checks: passed=%d warnings=%d errors=%d",
            result.successful_checks,
            len(result.warnings),
            len(result.errors),
        )

        return result

    def _validate_required_files(self) -> None:
        for label, path in (
            ("Source video", self.paths.source_video),
            (
                "Phase 3 decisions",
                self.paths.phase3_decisions,
            ),
            (
                "Phase 4 compressed video",
                self.paths.phase4_compressed_video,
            ),
            (
                "Phase 4 summary",
                self.paths.phase4_summary,
            ),
            (
                "Phase 5 JSON report",
                self.paths.phase5_report_json,
            ),
            (
                "Phase 5 text report",
                self.paths.phase5_report_text,
            ),
        ):
            _required_file(path, label)

    @staticmethod
    def _check_statuses(
        data: LoadedPipelineData,
        checks: list[ConsistencyCheck],
    ) -> None:
        _add(
            checks,
            "phase3_valid",
            "phase_status",
            (
                data.phase3.total_records > 0
                and data.phase3.malformed == 0
            ),
            "Phase 3 decision output is valid",
            "Phase 3 decisions are missing or malformed",
        )

        for name, payload in (
            ("phase4", data.phase4),
            ("phase5", data.phase5),
        ):
            status = payload.get("status")
            completed = status in {
                "completed",
                "completed_with_warnings",
            }

            _add(
                checks,
                f"{name}_completed",
                "phase_status",
                completed,
                f"{name.title()} completed successfully",
                f"{name.title()} status is not completed",
                "completed",
                status,
            )

            if status == "completed_with_warnings":
                checks.append(
                    ConsistencyCheck(
                        name=f"{name}_warnings",
                        category="phase_status",
                        status="warning",
                        message=(
                            f"{name.title()} completed with warnings"
                        ),
                        actual=status,
                    )
                )

    @staticmethod
    def _check_decisions(
        data: LoadedPipelineData,
        checks: list[ConsistencyCheck],
    ) -> None:
        phase4_decisions = _mapping(
            data.phase4.get("decisions")
        )

        for key, actual in (
            (
                "frames_analyzed",
                data.phase3.total_records,
            ),
            (
                "frames_retained",
                data.phase3.retained,
            ),
            (
                "frames_dropped",
                data.phase3.dropped,
            ),
        ):
            _equal(
                checks,
                f"phase3_phase4_{key}",
                "decision_statistics",
                actual,
                phase4_decisions.get(key),
            )

    def _check_media(
        self,
        data: LoadedPipelineData,
        checks: list[ConsistencyCheck],
    ) -> None:
        phase4_video = _mapping(
            data.phase4.get("video")
        )
        phase4_source = _mapping(
            phase4_video.get("source")
        )
        phase4_output = _mapping(
            phase4_video.get("output")
        )
        phase5_source = _mapping(
            data.phase5.get("source_video")
        )
        phase5_output = _mapping(
            data.phase5.get("compressed_video")
        )

        self._compare_media(
            "source_phase4",
            data.source_media,
            phase4_source,
            checks,
        )
        self._compare_media(
            "source_phase5",
            data.source_media,
            phase5_source,
            checks,
        )
        self._compare_media(
            "compressed_phase4",
            data.compressed_media,
            phase4_output,
            checks,
        )
        self._compare_media(
            "compressed_phase5",
            data.compressed_media,
            phase5_output,
            checks,
        )

        _equal(
            checks,
            "source_compressed_resolution",
            "temporal_integrity",
            [
                data.source_media.width,
                data.source_media.height,
            ],
            [
                data.compressed_media.width,
                data.compressed_media.height,
            ],
        )
        _equal(
            checks,
            "source_compressed_frame_count",
            "temporal_integrity",
            data.source_media.frame_count,
            data.compressed_media.frame_count,
        )
        _close(
            checks,
            "source_compressed_fps",
            "temporal_integrity",
            data.source_media.average_fps,
            data.compressed_media.average_fps,
            self.tolerances.fps,
        )
        _close(
            checks,
            "source_compressed_duration",
            "temporal_integrity",
            data.source_media.duration_seconds,
            data.compressed_media.duration_seconds,
            self.tolerances.duration_seconds,
        )

    def _compare_media(
        self,
        prefix: str,
        measured: MediaMetadata,
        reported: Mapping[str, Any],
        checks: list[ConsistencyCheck],
    ) -> None:
        _equal(
            checks,
            f"{prefix}_resolution",
            "media_metadata",
            [measured.width, measured.height],
            [
                reported.get("width"),
                reported.get("height"),
            ],
        )
        _equal(
            checks,
            f"{prefix}_frame_count",
            "media_metadata",
            measured.frame_count,
            reported.get("frame_count"),
        )
        _close(
            checks,
            f"{prefix}_fps",
            "media_metadata",
            measured.average_fps,
            reported.get("average_fps"),
            self.tolerances.fps,
        )
        _close(
            checks,
            f"{prefix}_duration",
            "media_metadata",
            measured.duration_seconds,
            reported.get("duration_seconds"),
            self.tolerances.duration_seconds,
        )
        _equal(
            checks,
            f"{prefix}_codec",
            "media_metadata",
            measured.codec,
            reported.get("codec"),
        )
        _equal(
            checks,
            f"{prefix}_pixel_format",
            "media_metadata",
            measured.pixel_format,
            reported.get("pixel_format"),
        )

    def _check_compression(
        self,
        data: LoadedPipelineData,
        checks: list[ConsistencyCheck],
    ) -> None:
        phase4_storage = _mapping(
            data.phase4.get("storage")
        )
        phase5_storage = _mapping(
            data.phase5.get("storage_metrics")
        )

        for key in (
            "original_bytes",
            "compressed_bytes",
            "bytes_saved",
        ):
            _equal(
                checks,
                f"phase4_phase5_{key}",
                "compression",
                phase4_storage.get(key),
                phase5_storage.get(key),
            )

        _close(
            checks,
            "phase4_phase5_compression_ratio",
            "compression",
            phase4_storage.get("compression_ratio"),
            phase5_storage.get("compression_ratio"),
            self.tolerances.compression_ratio,
        )
        _close(
            checks,
            "phase4_phase5_storage_reduction",
            "compression",
            phase4_storage.get(
                "storage_reduction_percent"
            ),
            phase5_storage.get(
                "storage_reduction_percent"
            ),
            self.tolerances.storage_reduction_percent,
        )
        _equal(
            checks,
            "measured_original_size",
            "compression",
            data.source_media.size_bytes,
            phase4_storage.get("original_bytes"),
        )
        _equal(
            checks,
            "measured_compressed_size",
            "compression",
            data.compressed_media.size_bytes,
            phase4_storage.get("compressed_bytes"),
        )

    @staticmethod
    def _check_quality(
        data: LoadedPipelineData,
        checks: list[ConsistencyCheck],
    ) -> None:
        quality = _mapping(
            data.phase5.get("quality_analysis")
        )
        alignment = _mapping(
            quality.get("timeline_alignment")
        )

        _add(
            checks,
            "timeline_alignment",
            "quality",
            alignment.get("proven") is True,
            "Phase 5 proved one-to-one timeline alignment",
            "Phase 5 did not prove one-to-one timeline alignment",
            True,
            alignment.get("proven"),
        )

        metrics = quality.get("metrics")
        metric_items = (
            metrics
            if isinstance(metrics, list)
            else []
        )
        by_name = {
            item.get("name"): item
            for item in metric_items
            if isinstance(item, Mapping)
        }

        for name in ("psnr", "ssim", "vmaf"):
            item = by_name.get(name)

            if item is None:
                checks.append(
                    ConsistencyCheck(
                        name=name,
                        category="quality",
                        status="warning",
                        message=(
                            f"{name.upper()} is not present "
                            "in Phase 5"
                        ),
                    )
                )
            elif (
                item.get("status") == "completed"
                and (
                    _finite(item.get("value"))
                    or item.get("infinite") is True
                )
            ):
                checks.append(
                    ConsistencyCheck(
                        name=name,
                        category="quality",
                        status="passed",
                        message=(
                            f"{name.upper()} value was "
                            "preserved from Phase 5"
                        ),
                        actual=item.get("value"),
                    )
                )
            else:
                checks.append(
                    ConsistencyCheck(
                        name=name,
                        category="quality",
                        status="warning",
                        message=(
                            f"{name.upper()} is unavailable: "
                            f"{item.get('reason') or item.get('status')}"
                        ),
                        expected="completed",
                        actual=item.get("status"),
                    )
                )


def _load_phase3(path: Path) -> Phase3Statistics:
    decision_path = _required_file(
        path,
        "Phase 3 decisions",
    )
    total = 0
    retained = 0
    dropped = 0
    malformed = 0
    priorities: dict[str, int] = {}
    seen: set[int] = set()

    try:
        with decision_path.open(
            "r",
            encoding="utf-8-sig",
            errors="strict",
        ) as stream:
            for line_number, line in enumerate(
                stream,
                start=1,
            ):
                if not line.strip():
                    continue

                try:
                    item = json.loads(line)

                    if not isinstance(item, Mapping):
                        raise ValueError(
                            "record is not an object"
                        )

                    index = _nonnegative_int(
                        item.get("frame_index")
                    )
                    decision = item.get("decision")

                    if (
                        index is None
                        or decision not in {"retain", "drop"}
                    ):
                        raise ValueError(
                            "invalid frame_index or decision"
                        )

                    if index in seen:
                        raise ValueError(
                            f"duplicate frame_index {index}"
                        )

                    seen.add(index)
                    total += 1

                    if decision == "retain":
                        retained += 1
                    else:
                        dropped += 1

                    priority = _text(
                        item.get("priority")
                    )
                    if priority:
                        priorities[priority] = (
                            priorities.get(priority, 0) + 1
                        )
                except (
                    json.JSONDecodeError,
                    ValueError,
                ) as exc:
                    malformed += 1
                    LOGGER.warning(
                        "Invalid Phase 3 record at line %d: %s",
                        line_number,
                        exc,
                    )
    except (OSError, UnicodeError) as exc:
        raise IntegrationError(
            f"Unable to read Phase 3 decisions: {exc}"
        ) from exc

    if total == 0 or malformed:
        raise IntegrationError(
            "Phase 3 decisions are not valid: "
            f"records={total}, malformed={malformed}"
        )

    return Phase3Statistics(
        path=decision_path,
        total_records=total,
        retained=retained,
        dropped=dropped,
        malformed=malformed,
        priority_distribution=dict(
            sorted(priorities.items())
        ),
    )


def _load_report(
    path: Path,
    label: str,
    expected_phase: str,
) -> Mapping[str, Any]:
    report_path = _required_file(path, label)

    try:
        with report_path.open(
            "r",
            encoding="utf-8-sig",
        ) as stream:
            payload = json.load(stream)
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
    ) as exc:
        raise IntegrationError(
            f"Unable to load {label}: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise IntegrationError(
            f"{label} root must be an object"
        )

    if (
        payload.get("schema_version") != "1.0"
        or payload.get("phase") != expected_phase
    ):
        raise IntegrationError(
            f"{label} has an unsupported schema "
            "or phase identifier"
        )

    return payload


def _required_file(
    path: Path,
    label: str,
) -> Path:
    resolved = path.expanduser().resolve()

    if not resolved.is_file():
        raise IntegrationError(
            f"{label} not found: {resolved}"
        )

    try:
        if resolved.stat().st_size <= 0:
            raise IntegrationError(
                f"{label} is empty: {resolved}"
            )
    except OSError as exc:
        raise IntegrationError(
            f"Unable to inspect {label}: {exc}"
        ) from exc

    return resolved


def _resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()

    if (
        candidate.is_absolute()
        or candidate.parent != Path(".")
    ):
        if candidate.is_file():
            return str(candidate.resolve())

        raise IntegrationError(
            f"FFprobe executable not found: {candidate}"
        )

    resolved = shutil.which(value)

    if resolved is None:
        raise IntegrationError(
            "FFprobe executable not found on PATH"
        )

    return resolved


def _add(
    checks: list[ConsistencyCheck],
    name: str,
    category: str,
    condition: bool,
    passed: str,
    failed: str,
    expected: Any = True,
    actual: Any = None,
) -> None:
    checks.append(
        ConsistencyCheck(
            name=name,
            category=category,
            status="passed" if condition else "error",
            message=passed if condition else failed,
            expected=expected,
            actual=actual,
        )
    )


def _equal(
    checks: list[ConsistencyCheck],
    name: str,
    category: str,
    expected: Any,
    actual: Any,
) -> None:
    _add(
        checks,
        name,
        category,
        expected == actual,
        f"{name} matches",
        (
            f"{name} mismatch: expected {expected!r}, "
            f"got {actual!r}"
        ),
        expected,
        actual,
    )


def _close(
    checks: list[ConsistencyCheck],
    name: str,
    category: str,
    expected: Any,
    actual: Any,
    tolerance: float,
) -> None:
    left = _number(expected)
    right = _number(actual)

    valid = (
        left is not None
        and right is not None
        and abs(left - right) <= tolerance
    )

    _add(
        checks,
        name,
        category,
        valid,
        f"{name} is within tolerance",
        (
            f"{name} mismatch: expected {expected!r}, "
            f"got {actual!r}, tolerance {tolerance}"
        ),
        expected,
        actual,
    )


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    return result if math.isfinite(result) else None


def _finite(value: Any) -> bool:
    return _number(value) is not None


def _positive_float(value: Any) -> float | None:
    result = _number(value)

    if result is None or result <= 0:
        return None

    return result


def _nonnegative_int(value: Any) -> int | None:
    result = _number(value)

    if (
        result is None
        or result < 0
        or not result.is_integer()
    ):
        return None

    return int(result)


def _positive_int(value: Any) -> int | None:
    result = _nonnegative_int(value)

    if result is None or result <= 0:
        return None

    return result


def _rate(value: Any) -> float | None:
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

    if not math.isfinite(result) or result <= 0:
        return None

    return result


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()

    return None


def _tail(
    value: str,
    limit: int = 2000,
) -> str:
    cleaned = value.strip()

    return (
        cleaned[-limit:]
        if cleaned
        else "no diagnostic output"
    )
