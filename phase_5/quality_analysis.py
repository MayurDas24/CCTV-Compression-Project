"""Optional, alignment-aware FFmpeg quality metrics for Phase 5."""

from __future__ import annotations

import logging
import math
import re
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from validation import MediaInformation


LOGGER = logging.getLogger("QualityAnalysis")


class QualityMetric(str, Enum):
    """Supported objective full-reference quality metrics."""

    PSNR = "psnr"
    SSIM = "ssim"
    VMAF = "vmaf"


@dataclass(frozen=True, slots=True)
class QualityConfig:
    """Configuration for optional objective quality analysis."""

    enabled: bool
    metrics: tuple[QualityMetric, ...]
    maximum_fps_difference: float
    maximum_duration_difference_seconds: float
    timeout_seconds: float | None


@dataclass(frozen=True, slots=True)
class AlignmentAssessment:
    """Whether direct frame-paired quality comparison is defensible."""

    aligned: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class QualityMetricResult:
    """Result or explicit unavailability reason for one metric."""

    name: str
    status: str
    value: float | None
    infinite: bool
    methodology: str
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "value": self.value,
            "infinite": self.infinite,
            "methodology": self.methodology,
            "reason": self.reason,
        }


def assess_timeline_alignment(
    source: MediaInformation,
    compressed: MediaInformation,
    config: QualityConfig,
) -> AlignmentAssessment:
    """Require evidence of one-to-one timeline correspondence."""

    reasons: list[str] = []

    if (
        source.width,
        source.height,
    ) != (
        compressed.width,
        compressed.height,
    ):
        reasons.append(
            "Source and compressed resolutions differ"
        )

    if (
        source.average_fps is None
        or compressed.average_fps is None
    ):
        reasons.append(
            "Comparable average FPS metadata is unavailable"
        )
    elif (
        abs(
            source.average_fps
            - compressed.average_fps
        )
        > config.maximum_fps_difference
    ):
        reasons.append(
            "FPS difference exceeds the configured "
            "alignment tolerance"
        )

    if (
        source.duration_seconds is None
        or compressed.duration_seconds is None
    ):
        reasons.append(
            "Comparable duration metadata is unavailable"
        )
    elif (
        abs(
            source.duration_seconds
            - compressed.duration_seconds
        )
        > config.maximum_duration_difference_seconds
    ):
        reasons.append(
            "Duration difference exceeds the configured "
            "alignment tolerance"
        )

    if (
        source.frame_count is None
        or compressed.frame_count is None
    ):
        reasons.append(
            "Frame counts are unavailable, so pairing "
            "cannot be proven"
        )
    elif (
        source.frame_count_method != "ffprobe_count_frames"
        or compressed.frame_count_method != "ffprobe_count_frames"
    ):
        reasons.append(
            "Decoded frame counts are unavailable for one or both "
            "videos, so pairing cannot be proven"
        )
    elif source.frame_count != compressed.frame_count:
        reasons.append(
            "Source and compressed frame counts differ"
        )

    return AlignmentAssessment(
        aligned=not reasons,
        reasons=tuple(reasons),
    )


class QualityAnalyzer:
    """Calculate optional metrics only when pairing is trustworthy."""

    _METHODOLOGY = {
        QualityMetric.PSNR: (
            "FFmpeg psnr filter; compressed video is the distorted "
            "input and source video is the full-reference input"
        ),
        QualityMetric.SSIM: (
            "FFmpeg ssim filter; compressed video is the distorted "
            "input and source video is the full-reference input"
        ),
        QualityMetric.VMAF: (
            "FFmpeg libvmaf filter using its configured default "
            "model; compressed video is the distorted input and "
            "source video is the reference"
        ),
    }

    def __init__(
        self,
        ffmpeg_path: str,
        config: QualityConfig,
    ) -> None:
        self.config = config
        self.ffmpeg = _find_executable(ffmpeg_path)

    def analyze(
        self,
        source: MediaInformation,
        compressed: MediaInformation,
        alignment: AlignmentAssessment,
    ) -> dict[str, Any]:
        """Return completed, disabled, or unavailable metrics."""

        if not self.config.enabled:
            results = [
                self._unavailable(
                    metric,
                    "Quality analysis is disabled",
                )
                for metric in self.config.metrics
            ]

            return self._result_payload(
                source,
                compressed,
                alignment,
                results,
            )

        if not alignment.aligned:
            reason = (
                "Direct full-reference comparison is not valid "
                "because timeline alignment was not proven: "
                + "; ".join(alignment.reasons)
            )
            results = [
                self._unavailable(metric, reason)
                for metric in self.config.metrics
            ]
            LOGGER.warning(reason)

            return self._result_payload(
                source,
                compressed,
                alignment,
                results,
            )

        if self.ffmpeg is None:
            reason = (
                "FFmpeg executable is unavailable; required "
                "validation and non-quality metrics remain valid"
            )
            results = [
                self._unavailable(metric, reason)
                for metric in self.config.metrics
            ]
            LOGGER.warning(reason)

            return self._result_payload(
                source,
                compressed,
                alignment,
                results,
            )

        filters, filter_error = self._available_filters()
        results: list[QualityMetricResult] = []

        for metric in self.config.metrics:
            required_filter = (
                "libvmaf"
                if metric is QualityMetric.VMAF
                else metric.value
            )

            if filter_error is not None:
                results.append(
                    self._unavailable(
                        metric,
                        filter_error,
                    )
                )
                continue

            if required_filter not in filters:
                results.append(
                    self._unavailable(
                        metric,
                        "Configured FFmpeg build does not provide "
                        f"the '{required_filter}' filter",
                    )
                )
                continue

            results.append(
                self._run_metric(
                    metric,
                    source.path,
                    compressed.path,
                )
            )

        return self._result_payload(
            source,
            compressed,
            alignment,
            results,
        )

    def _run_metric(
        self,
        metric: QualityMetric,
        source: Path,
        compressed: Path,
    ) -> QualityMetricResult:
        filter_name = (
            "libvmaf"
            if metric is QualityMetric.VMAF
            else metric.value
        )

        command = [
            self.ffmpeg or "ffmpeg",
            "-hide_banner",
            "-nostdin",
            "-v",
            "info",
            "-i",
            str(compressed),
            "-i",
            str(source),
            "-lavfi",
            f"[0:v:0][1:v:0]{filter_name}",
            "-an",
            "-f",
            "null",
            "-",
        ]

        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.config.timeout_seconds,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return self._unavailable(
                metric,
                "Metric calculation exceeded the configured "
                f"timeout of {self.config.timeout_seconds} seconds",
            )
        except OSError as exc:
            return self._unavailable(
                metric,
                f"Unable to execute FFmpeg: {exc}",
            )

        if result.returncode != 0:
            return self._unavailable(
                metric,
                "FFmpeg metric process failed: "
                f"{_diagnostic_tail(result.stderr)}",
            )

        parsed = _parse_metric(
            metric,
            result.stderr,
        )

        if parsed is None:
            return self._unavailable(
                metric,
                "FFmpeg completed but its metric summary "
                "could not be parsed",
            )

        value, infinite = parsed

        LOGGER.info(
            "Quality metric %s=%s",
            metric.value,
            "infinite" if infinite else f"{value:.6f}",
        )

        return QualityMetricResult(
            name=metric.value,
            status="completed",
            value=None if infinite else value,
            infinite=infinite,
            methodology=self._METHODOLOGY[metric],
            reason=None,
        )

    def _available_filters(
        self,
    ) -> tuple[set[str], str | None]:
        command = [
            self.ffmpeg or "ffmpeg",
            "-hide_banner",
            "-filters",
        ]

        try:
            result = subprocess.run(
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30.0,
                check=False,
                shell=False,
            )
        except (
            OSError,
            subprocess.TimeoutExpired,
        ) as exc:
            return (
                set(),
                f"Unable to query FFmpeg filters: {exc}",
            )

        if result.returncode != 0:
            return (
                set(),
                "Unable to query FFmpeg filters: "
                + _diagnostic_tail(result.stderr),
            )

        filters: set[str] = set()

        for line in result.stdout.splitlines():
            fields = line.split()

            if (
                len(fields) >= 2
                and fields[0]
                and fields[0][0] in ".TSC"
            ):
                filters.add(fields[1])

        return filters, None

    def _unavailable(
        self,
        metric: QualityMetric,
        reason: str,
    ) -> QualityMetricResult:
        return QualityMetricResult(
            name=metric.value,
            status=(
                "unavailable"
                if self.config.enabled
                else "disabled"
            ),
            value=None,
            infinite=False,
            methodology=self._METHODOLOGY[metric],
            reason=reason,
        )

    @staticmethod
    def _result_payload(
        source: MediaInformation,
        compressed: MediaInformation,
        alignment: AlignmentAssessment,
        results: list[QualityMetricResult],
    ) -> dict[str, Any]:
        return {
            "timeline_alignment": {
                "proven": alignment.aligned,
                "reasons": list(alignment.reasons),
                "source_frame_count": source.frame_count,
                "compressed_frame_count": (
                    compressed.frame_count
                ),
                "frame_count_difference": (
                    compressed.frame_count
                    - source.frame_count
                    if source.frame_count is not None
                    and compressed.frame_count is not None
                    else None
                ),
                "source_frame_count_method": (
                    source.frame_count_method
                ),
                "compressed_frame_count_method": (
                    compressed.frame_count_method
                ),
                "resolution_matches": (
                    (
                        source.width,
                        source.height,
                    )
                    == (
                        compressed.width,
                        compressed.height,
                    )
                ),
            },
            "metrics": [
                result.to_dict()
                for result in results
            ],
        }


def _find_executable(value: str) -> str | None:
    candidate = Path(value).expanduser()

    if candidate.is_absolute() or candidate.parent != Path("."):
        if candidate.is_file():
            return str(candidate.resolve())

        return None

    return shutil.which(value)


def _parse_metric(
    metric: QualityMetric,
    stderr: str,
) -> tuple[float, bool] | None:
    patterns = {
        QualityMetric.PSNR: (
            r"average:([+\-0-9.eEinfINF]+)"
        ),
        QualityMetric.SSIM: (
            r"All:([+\-0-9.eE]+)"
        ),
        QualityMetric.VMAF: (
            r"VMAF score:\s*([+\-0-9.eE]+)"
        ),
    }

    matches = re.findall(
        patterns[metric],
        stderr,
    )

    if not matches:
        return None

    raw = matches[-1]

    if raw.lower() in {"inf", "+inf"}:
        return 0.0, True

    try:
        value = float(raw)
    except ValueError:
        return None

    if not math.isfinite(value):
        return None

    return value, False


def _diagnostic_tail(
    value: str,
    limit: int = 1200,
) -> str:
    cleaned = value.strip()

    if not cleaned:
        return "No diagnostic output was provided."

    return cleaned[-limit:]
