"""Phase 5 validation, quality analysis, and reporting."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from metrics import calculate_metrics
from quality_analysis import (
    QualityAnalyzer,
    QualityConfig,
    QualityMetric,
    assess_timeline_alignment,
)
from reporting import (
    ReportWriter,
    ReportingError,
)
from validation import (
    DecisionValidator,
    MediaInspector,
    Phase4SummaryValidator,
    ValidationError,
    validate_cross_phase_consistency,
)


LOGGER = logging.getLogger("Phase5Pipeline")


class Phase5FatalError(RuntimeError):
    """Fatal configuration or orchestration error."""


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configurable validation strictness."""

    fail_on_malformed_decisions: bool
    fail_on_phase4_count_mismatch: bool


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """External tool configuration."""

    ffmpeg_path: str
    ffprobe_path: str
    ffprobe_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class Phase5Config:
    """Validated Phase 5 configuration."""

    source_video: Path
    decision_file: Path
    compressed_video: Path
    phase4_summary: Path
    json_report: Path
    text_report: Path
    validation: ValidationConfig
    quality: QualityConfig
    runtime: RuntimeConfig
    logging_level: str


def load_config(
    config_path: Path,
) -> Phase5Config:
    """Load configuration relative to the YAML file."""

    path = config_path.expanduser().resolve()

    if not path.is_file():
        raise Phase5FatalError(
            f"Configuration file not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Phase5FatalError(
            f"Invalid YAML in {path}: {exc}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise Phase5FatalError(
            "Unable to read configuration "
            f"{path}: {exc}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise Phase5FatalError(
            "Configuration root must be "
            "a YAML mapping"
        )

    inputs = _section(raw, "inputs")
    output = _section(raw, "output")

    validation_section = _section(
        raw,
        "validation",
    )

    quality_section = _section(
        raw,
        "quality",
    )

    runtime_section = _section(
        raw,
        "runtime",
    )

    logging_section = raw.get(
        "logging",
        {},
    )

    if not isinstance(logging_section, Mapping):
        raise Phase5FatalError(
            "Configuration section 'logging' "
            "must be a mapping"
        )

    base_directory = path.parent

    source_video = _resolve_path(
        base_directory,
        _string(
            inputs,
            "source_video",
        ),
    )

    decision_file = _resolve_path(
        base_directory,
        _string(
            inputs,
            "phase3_decisions",
        ),
    )

    compressed_video = _resolve_path(
        base_directory,
        _string(
            inputs,
            "phase4_compressed_video",
        ),
    )

    phase4_summary = _resolve_path(
        base_directory,
        _string(
            inputs,
            "phase4_compression_summary",
        ),
    )

    output_directory = _resolve_path(
        base_directory,
        _string(
            output,
            "directory",
        ),
    )

    json_report = _resolve_output_path(
        base_directory,
        output_directory,
        _string(
            output,
            "json_report",
        ),
    )

    text_report = _resolve_output_path(
        base_directory,
        output_directory,
        _string(
            output,
            "text_report",
        ),
    )

    required_output_root = (
        base_directory / "outputs"
    ).resolve()

    for label, report_path in (
        ("output.json_report", json_report),
        ("output.text_report", text_report),
    ):
        if not report_path.is_relative_to(
            required_output_root
        ):
            raise Phase5FatalError(
                f"{label} must be inside the Phase 5 "
                "outputs directory: "
                f"{required_output_root}"
            )

    if json_report == text_report:
        raise Phase5FatalError(
            "JSON and text report paths "
            "must be different"
        )

    input_paths = {
        source_video,
        decision_file,
        compressed_video,
        phase4_summary,
    }

    if (
        json_report in input_paths
        or text_report in input_paths
    ):
        raise Phase5FatalError(
            "Phase 5 reports must not overwrite "
            "input files"
        )

    validation_config = ValidationConfig(
        fail_on_malformed_decisions=_boolean(
            validation_section,
            "fail_on_malformed_decisions",
        ),
        fail_on_phase4_count_mismatch=_boolean(
            validation_section,
            "fail_on_phase4_count_mismatch",
        ),
    )

    raw_metrics = quality_section.get("metrics")

    if (
        not isinstance(raw_metrics, Sequence)
        or isinstance(
            raw_metrics,
            (str, bytes, bytearray),
        )
    ):
        raise Phase5FatalError(
            "quality.metrics must be a list"
        )

    quality_metrics: list[QualityMetric] = []

    for raw_metric in raw_metrics:
        if not isinstance(raw_metric, str):
            raise Phase5FatalError(
                "quality.metrics contains "
                "a non-string value"
            )

        try:
            metric = QualityMetric(
                raw_metric.strip().lower()
            )
        except ValueError as exc:
            raise Phase5FatalError(
                "Unsupported quality metric: "
                f"{raw_metric}"
            ) from exc

        if metric not in quality_metrics:
            quality_metrics.append(metric)

    metric_timeout = _number(
        quality_section,
        "timeout_seconds",
    )

    if metric_timeout < 0.0:
        raise Phase5FatalError(
            "quality.timeout_seconds "
            "cannot be negative"
        )

    quality_config = QualityConfig(
        enabled=_boolean(
            quality_section,
            "enabled",
        ),
        metrics=tuple(quality_metrics),
        maximum_fps_difference=(
            _nonnegative_number(
                quality_section,
                "maximum_fps_difference",
            )
        ),
        maximum_duration_difference_seconds=(
            _nonnegative_number(
                quality_section,
                "maximum_duration_difference_seconds",
            )
        ),
        timeout_seconds=(
            metric_timeout
            if metric_timeout > 0.0
            else None
        ),
    )

    ffprobe_timeout = _number(
        runtime_section,
        "ffprobe_timeout_seconds",
    )

    if ffprobe_timeout <= 0.0:
        raise Phase5FatalError(
            "runtime.ffprobe_timeout_seconds "
            "must be positive"
        )

    runtime_config = RuntimeConfig(
        ffmpeg_path=_resolve_executable_value(
            base_directory,
            _string(
                runtime_section,
                "ffmpeg_path",
            ),
        ),
        ffprobe_path=_resolve_executable_value(
            base_directory,
            _string(
                runtime_section,
                "ffprobe_path",
            ),
        ),
        ffprobe_timeout_seconds=(
            ffprobe_timeout
        ),
    )

    logging_level = (
        _optional_string(
            logging_section,
            "level",
        )
        or "INFO"
    ).upper()

    if (
        logging_level
        not in logging.getLevelNamesMapping()
    ):
        raise Phase5FatalError(
            f"Invalid logging level: {logging_level}"
        )

    return Phase5Config(
        source_video=source_video,
        decision_file=decision_file,
        compressed_video=compressed_video,
        phase4_summary=phase4_summary,
        json_report=json_report,
        text_report=text_report,
        validation=validation_config,
        quality=quality_config,
        runtime=runtime_config,
        logging_level=logging_level,
    )


class Phase5Pipeline:
    """Coordinate validation, metrics, quality, and reports."""

    def __init__(
        self,
        config: Phase5Config,
    ) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        """Run Phase 5 and return the final report."""

        started_at = time.perf_counter()
        config = self.config

        LOGGER.info(
            "Starting Phase 5 validation "
            "and quality analysis"
        )

        self._prepare_output_directory()

        decisions = DecisionValidator(
            fail_on_malformed=(
                config.validation
                .fail_on_malformed_decisions
            )
        ).validate(
            config.decision_file
        )

        phase4 = (
            Phase4SummaryValidator().load(
                config.phase4_summary
            )
        )

        inspector = MediaInspector(
            config.runtime.ffprobe_path,
            config.runtime.ffprobe_timeout_seconds,
        )

        source = inspector.inspect(
            config.source_video,
            "Source video",
        )

        compressed = inspector.inspect(
            config.compressed_video,
            "Compressed video",
        )

        warnings = (
            validate_cross_phase_consistency(
                decisions,
                phase4,
                source,
                compressed,
                config.compressed_video,
            )
        )

        count_mismatches = [
            warning
            for warning in warnings
            if (
                warning.startswith(
                    "Phase 4 decisions."
                )
                and (
                    "but Phase 3 validation measured"
                    in warning
                )
            )
        ]

        if (
            count_mismatches
            and (
                config.validation
                .fail_on_phase4_count_mismatch
            )
        ):
            raise ValidationError(
                "; ".join(count_mismatches)
            )

        calculated = calculate_metrics(
            source,
            compressed,
            decisions,
            phase4,
        )

        alignment = assess_timeline_alignment(
            source,
            compressed,
            config.quality,
        )

        quality_payload = QualityAnalyzer(
            config.runtime.ffmpeg_path,
            config.quality,
        ).analyze(
            source,
            compressed,
            alignment,
        )

        limitations = self._limitations(
            source,
            compressed,
            alignment.reasons,
        )

        if config.quality.enabled:
            unavailable_quality = [
                item
                for item in quality_payload["metrics"]
                if item["status"] == "unavailable"
            ]

            if unavailable_quality:
                warnings.append(
                    "One or more optional objective "
                    "quality metrics are unavailable"
                )

        if decisions.malformed_records:
            warnings.append(
                "Skipped "
                f"{decisions.malformed_records} "
                "malformed decision records"
            )

        analysis_duration = (
            time.perf_counter() - started_at
        )

        performance = dict(
            calculated.performance
        )

        performance[
            "phase5_analysis_duration_seconds"
        ] = round(
            analysis_duration,
            6,
        )

        performance[
            "phase5_decision_validation_"
            "records_per_second"
        ] = (
            round(
                decisions.total_decision_records
                / analysis_duration,
                6,
            )
            if analysis_duration > 0.0
            else None
        )

        report: dict[str, Any] = {
            "schema_version": "1.0",
            "phase": (
                "phase_5_quality_forensic_validation"
            ),
            "status": (
                "completed_with_warnings"
                if warnings
                else "completed"
            ),
            "inputs": {
                "source_video": str(
                    config.source_video
                ),
                "phase3_decisions": str(
                    config.decision_file
                ),
                "phase4_compressed_video": str(
                    config.compressed_video
                ),
                "phase4_compression_summary": str(
                    config.phase4_summary
                ),
            },
            "outputs": {
                "json_report": str(
                    config.json_report
                ),
                "text_report": str(
                    config.text_report
                ),
            },
            "validation": {
                "required_inputs_valid": True,
                "phase3_decisions_valid": True,
                "phase4_summary_valid": True,
                "source_media_readable": True,
                "compressed_media_readable": True,
                "phase4_count_consistency_valid": (
                    not count_mismatches
                ),
                "warnings": sorted(
                    set(warnings)
                ),
            },
            "source_video": (
                calculated.source_video
            ),
            "compressed_video": (
                calculated.compressed_video
            ),
            "frame_decisions": (
                calculated.frame_decisions
            ),
            "storage_metrics": (
                calculated.storage
            ),
            "integrity_metrics": (
                calculated.differences
            ),
            "performance_metrics": performance,
            "quality_analysis": quality_payload,
            "phase4_summary_reference": {
                "schema_version": (
                    phase4.schema_version
                ),
                "phase": phase4.phase,
                "status": phase4.status,
            },
            "limitations": limitations,
        }

        ReportWriter(
            config.json_report,
            config.text_report,
        ).write(report)

        LOGGER.info(
            "Storage reduction=%s%% "
            "compression_ratio=%sx",
            _display(
                calculated.storage.get(
                    "storage_reduction_percent"
                )
            ),
            _display(
                calculated.storage.get(
                    "compression_ratio"
                )
            ),
        )

        LOGGER.info(
            "Phase 5 report: %s",
            config.json_report,
        )

        LOGGER.info(
            "Phase 5 completed successfully"
        )

        return report

    def _prepare_output_directory(self) -> None:
        """Create the configured report directories."""

        try:
            self.config.json_report.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.config.text_report.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as exc:
            raise Phase5FatalError(
                "Unable to create Phase 5 output "
                f"directory: {exc}"
            ) from exc

    @staticmethod
    def _limitations(
        source: Any,
        compressed: Any,
        alignment_reasons: tuple[str, ...],
    ) -> list[str]:
        """Build limitations from measured conditions."""

        limitations: list[str] = []

        if alignment_reasons:
            limitations.append(
                "Direct PSNR/SSIM/VMAF comparison "
                "is not reported unless one-to-one "
                "timeline and frame alignment can "
                "be proven."
            )

        if (
            source.average_fps is not None
            and compressed.average_fps is not None
            and compressed.average_fps
            < source.average_fps * 0.9
        ):
            limitations.append(
                "The compressed average FPS is "
                "materially below the source FPS. "
                "This is consistent with the known "
                "sampled-frame Phase 4 output and may "
                "produce temporal lag or stuttering."
            )

        if (
            source.frame_count is None
            or compressed.frame_count is None
        ):
            limitations.append(
                "At least one container did not expose "
                "a reliable frame count; Phase 5 does "
                "not perform an additional full decode "
                "to estimate it."
            )

        limitations.append(
            "Phase 5 validates media and metadata but "
            "does not rerun object detection on the "
            "compressed video, so machine-vision "
            "accuracy preservation is not measured."
        )

        return limitations


def _section(
    root: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    """Return a required YAML mapping section."""

    value = root.get(key)

    if not isinstance(value, Mapping):
        raise Phase5FatalError(
            f"Configuration section '{key}' "
            "must be a mapping"
        )

    return value


def _string(
    section: Mapping[str, Any],
    key: str,
) -> str:
    """Return a required nonempty string."""

    value = section.get(key)

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise Phase5FatalError(
            f"Configuration value '{key}' "
            "must be a string"
        )

    return value.strip()


def _optional_string(
    section: Mapping[str, Any],
    key: str,
) -> str | None:
    """Return an optional nonempty string."""

    value = section.get(key)

    if value is None:
        return None

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise Phase5FatalError(
            f"Configuration value '{key}' must "
            "be a nonempty string or null"
        )

    return value.strip()


def _boolean(
    section: Mapping[str, Any],
    key: str,
) -> bool:
    """Return a required boolean."""

    value = section.get(key)

    if not isinstance(value, bool):
        raise Phase5FatalError(
            f"Configuration value '{key}' "
            "must be a boolean"
        )

    return value


def _number(
    section: Mapping[str, Any],
    key: str,
) -> float:
    """Return a required finite number."""

    value = section.get(key)

    if isinstance(value, bool):
        raise Phase5FatalError(
            f"Configuration value '{key}' "
            "must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError):
        raise Phase5FatalError(
            f"Configuration value '{key}' "
            "must be numeric"
        ) from None

    if not math.isfinite(result):
        raise Phase5FatalError(
            f"Configuration value '{key}' "
            "must be finite"
        )

    return result


def _nonnegative_number(
    section: Mapping[str, Any],
    key: str,
) -> float:
    """Return a required nonnegative number."""

    value = _number(
        section,
        key,
    )

    if value < 0.0:
        raise Phase5FatalError(
            f"Configuration value '{key}' "
            "cannot be negative"
        )

    return value


def _resolve_path(
    base: Path,
    configured: str,
) -> Path:
    """Resolve a path relative to config.yaml."""

    path = Path(configured).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        base / path
    ).resolve()


def _resolve_output_path(
    base: Path,
    output_directory: Path,
    configured: str,
) -> Path:
    """Resolve an output path."""

    path = Path(configured).expanduser()

    if path.is_absolute():
        return path.resolve()

    if path.parent != Path("."):
        return (
            base / path
        ).resolve()

    return (
        output_directory / path
    ).resolve()


def _resolve_executable_value(
    base: Path,
    configured: str,
) -> str:
    """Resolve explicit relative executable paths."""

    path = Path(configured).expanduser()

    if path.is_absolute():
        return str(path.resolve())

    if path.parent != Path("."):
        return str(
            (base / path).resolve()
        )

    return configured


def _display(value: Any) -> str:
    """Format an optional log value."""

    if value is None:
        return "unavailable"

    return str(value)


def _configure_logging(level: str) -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=getattr(logging, level),
        format=(
            "%(asctime)s [%(levelname)s] "
            "%(name)s: %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )


def _parse_arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate and report Phase 4 "
            "CCTV compression results."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "config.yaml"
        ),
        help=(
            "Path to Phase 5 config.yaml "
            "(default: beside this script)"
        ),
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> int:
    """Run Phase 5 and return its process exit code."""

    arguments = _parse_arguments(argv)

    try:
        config = load_config(
            arguments.config
        )

        _configure_logging(
            config.logging_level
        )

        Phase5Pipeline(config).run()
        return 0

    except (
        Phase5FatalError,
        ValidationError,
        ReportingError,
        ValueError,
    ) as exc:
        if not logging.getLogger().handlers:
            _configure_logging("ERROR")

        LOGGER.error(
            "Phase 5 failed: %s",
            exc,
        )

        return 1

    except Exception:
        if not logging.getLogger().handlers:
            _configure_logging("ERROR")

        LOGGER.exception(
            "Phase 5 failed due to an "
            "unexpected error"
        )

        return 1


if __name__ == "__main__":
    sys.exit(main())