"""Streaming orchestrator for the Phase 3 frame decision layer."""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import yaml

from compression_policy import (
    CompressionPolicy,
    PolicyConfig,
)
from decision_writer import (
    DecisionWriter,
    DecisionWriterError,
    SummaryAccumulator,
)
from importance_engine import (
    ImportanceConfig,
    ImportanceEngine,
)
from temporal_analyzer import (
    TemporalAnalyzer,
    TemporalConfig,
)


LOGGER = logging.getLogger("Phase3Pipeline")

SCHEMA_VERSION = "1.0"
PHASE_NAME = "phase_3_frame_decision"


class PipelineFatalError(RuntimeError):
    """Fatal configuration, input, or output pipeline error."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Validated and path-resolved Phase 3 configuration."""

    metadata_file: Path
    decisions_file: Path
    summary_file: Path
    importance: ImportanceConfig
    temporal: TemporalConfig
    policy: PolicyConfig
    logging_level: str


@dataclass(frozen=True, slots=True)
class InputRecord:
    """Minimally validated Phase 2 JSONL record."""

    line_number: int
    frame_index: int
    timestamp_seconds: float
    metadata: Mapping[str, Any]


@dataclass(slots=True)
class InputCounters:
    """Counters updated while streaming the input JSONL."""

    total_lines: int = 0
    nonempty_lines: int = 0
    malformed_records: int = 0


def load_config(config_path: Path) -> PipelineConfig:
    """Load, validate, and resolve the Phase 3 YAML configuration."""

    path = config_path.expanduser().resolve()

    if not path.is_file():
        raise PipelineFatalError(
            f"Configuration file not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise PipelineFatalError(
            f"Invalid YAML in {path}: {exc}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise PipelineFatalError(
            f"Unable to read configuration {path}: {exc}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise PipelineFatalError(
            "Configuration root must be a YAML mapping"
        )

    input_section = _mapping_section(
        raw,
        "input",
    )

    decision_section = _mapping_section(
        raw,
        "decision",
    )

    output_section = _mapping_section(
        raw,
        "output",
    )

    logging_section = raw.get("logging", {})

    if not isinstance(logging_section, Mapping):
        raise PipelineFatalError(
            "Configuration section 'logging' "
            "must be a mapping"
        )

    base_directory = path.parent

    metadata_file = _resolve_config_path(
        base_directory,
        _required_string(
            input_section,
            "metadata_file",
        ),
    )

    output_directory = _resolve_config_path(
        base_directory,
        _optional_string(
            output_section,
            "directory",
            "outputs",
        ),
    )

    decisions_file = _resolve_output_path(
        base_directory,
        output_directory,
        _optional_string(
            output_section,
            "decisions_file",
            "frame_decisions.jsonl",
        ),
    )

    summary_file = _resolve_output_path(
        base_directory,
        output_directory,
        _optional_string(
            output_section,
            "summary_file",
            "summary.json",
        ),
    )

    if decisions_file == summary_file:
        raise PipelineFatalError(
            "decisions_file and summary_file must "
            "resolve to different paths"
        )

    if metadata_file in {
        decisions_file,
        summary_file,
    }:
        raise PipelineFatalError(
            "Output paths must not overwrite "
            "the input metadata"
        )

    max_consecutive_drops = _required_int(
        decision_section,
        "max_consecutive_drops",
    )

    importance_config = ImportanceConfig(
        high_importance_threshold=_required_float(
            decision_section,
            "high_importance_threshold",
        ),
        low_importance_threshold=_required_float(
            decision_section,
            "low_importance_threshold",
        ),
        minimum_detection_confidence=_required_float(
            decision_section,
            "minimum_detection_confidence",
        ),
        critical_classes=_class_set(
            decision_section,
            "critical_classes",
        ),
        important_classes=_class_set(
            decision_section,
            "important_classes",
        ),
    )

    temporal_config = TemporalConfig(
        max_consecutive_drops=max_consecutive_drops,
    )

    policy_config = PolicyConfig(
        retain_empty_frames=_required_bool(
            decision_section,
            "retain_empty_frames",
        ),
        max_consecutive_drops=max_consecutive_drops,
    )

    logging_level = _optional_string(
        logging_section,
        "level",
        "INFO",
    ).upper()

    if (
        logging_level
        not in logging.getLevelNamesMapping()
    ):
        raise PipelineFatalError(
            f"Invalid logging level: {logging_level}"
        )

    return PipelineConfig(
        metadata_file=metadata_file,
        decisions_file=decisions_file,
        summary_file=summary_file,
        importance=importance_config,
        temporal=temporal_config,
        policy=policy_config,
        logging_level=logging_level,
    )


def run_pipeline(
    config: PipelineConfig,
) -> dict[str, Any]:
    """Execute the Phase 3 streaming decision pipeline."""

    if not config.metadata_file.is_file():
        raise PipelineFatalError(
            "Phase 2 perception metadata not found: "
            f"{config.metadata_file}"
        )

    LOGGER.info(
        "Starting Phase 3 intelligent frame decision pipeline"
    )
    LOGGER.info(
        "Input metadata: %s",
        config.metadata_file,
    )

    importance_engine = ImportanceEngine(
        config.importance
    )

    temporal_analyzer = TemporalAnalyzer(
        config.temporal
    )

    compression_policy = CompressionPolicy(
        config.policy
    )

    accumulator = SummaryAccumulator()
    counters = InputCounters()

    metadata_warnings = 0
    started_at = time.perf_counter()

    try:
        with DecisionWriter(
            config.decisions_file,
            config.summary_file,
        ) as writer:
            input_records = _iter_input_records(
                config.metadata_file,
                counters,
            )

            for input_record in input_records:
                assessment = importance_engine.assess(
                    input_record.metadata
                )

                if assessment.issues:
                    metadata_warnings += 1

                    LOGGER.warning(
                        "Line=%d frame=%d metadata "
                        "normalized: %s",
                        input_record.line_number,
                        input_record.frame_index,
                        "; ".join(assessment.issues),
                    )

                temporal_context = (
                    temporal_analyzer.analyze(
                        assessment
                    )
                )

                result = compression_policy.decide(
                    assessment,
                    temporal_context,
                )

                if result.retained:
                    consecutive_drops = 0
                else:
                    consecutive_drops = (
                        temporal_context.consecutive_drops
                        + 1
                    )

                output_record = {
                    "schema_version": SCHEMA_VERSION,
                    "phase": PHASE_NAME,
                    "frame_index": (
                        input_record.frame_index
                    ),
                    "timestamp_seconds": round(
                        input_record.timestamp_seconds,
                        6,
                    ),
                    "importance_score": assessment.score,
                    "number_of_objects": (
                        assessment.number_of_objects
                    ),
                    "classes_present": list(
                        assessment.classes_present
                    ),
                    "decision": result.decision.value,
                    "priority": result.priority.value,
                    "reason": result.reason.value,
                    "critical_classes_present": list(
                        assessment
                        .critical_classes_present
                    ),
                    "important_classes_present": list(
                        assessment
                        .important_classes_present
                    ),
                    "consecutive_drops": (
                        consecutive_drops
                    ),
                }

                writer.write_decision(output_record)
                accumulator.update(output_record)

                temporal_analyzer.update(
                    assessment,
                    retained=result.retained,
                )

                LOGGER.info(
                    "Frame=%d decision=%s "
                    "priority=%s reason=%s",
                    input_record.frame_index,
                    result.decision.value,
                    result.priority.value,
                    result.reason.value,
                )

            if counters.nonempty_lines == 0:
                raise PipelineFatalError(
                    "Perception metadata file is empty: "
                    f"{config.metadata_file}"
                )

            if accumulator.total_frames == 0:
                raise PipelineFatalError(
                    "No valid perception records were "
                    "found in the input file"
                )

            processing_duration = (
                time.perf_counter() - started_at
            )

            summary = accumulator.build_summary(
                input_file=config.metadata_file,
                total_input_lines=counters.total_lines,
                malformed_records_skipped=(
                    counters.malformed_records
                ),
                metadata_warnings=metadata_warnings,
                configured_max_consecutive_drops=(
                    config.temporal
                    .max_consecutive_drops
                ),
                processing_duration_seconds=(
                    processing_duration
                ),
            )

            writer.finalize(summary)

    except DecisionWriterError as exc:
        raise PipelineFatalError(str(exc)) from exc
    except OSError as exc:
        raise PipelineFatalError(
            f"Input/output failure: {exc}"
        ) from exc

    LOGGER.info(
        "Completed Phase 3: analyzed=%d retained=%d "
        "dropped=%d skipped=%d",
        accumulator.total_frames,
        accumulator.frames_retained,
        accumulator.frames_dropped,
        counters.malformed_records,
    )

    LOGGER.info(
        "Decision output: %s",
        config.decisions_file,
    )

    LOGGER.info(
        "Summary output: %s",
        config.summary_file,
    )

    return summary


def _iter_input_records(
    path: Path,
    counters: InputCounters,
) -> Iterator[InputRecord]:
    """Stream valid records from a Phase 2 JSONL file."""

    with path.open(
        "r",
        encoding="utf-8-sig",
        errors="replace",
    ) as stream:
        for line_number, line in enumerate(
            stream,
            start=1,
        ):
            counters.total_lines += 1

            stripped = line.strip()

            if not stripped:
                continue

            counters.nonempty_lines += 1

            try:
                raw = json.loads(stripped)
            except json.JSONDecodeError as exc:
                counters.malformed_records += 1

                LOGGER.warning(
                    "Skipping invalid JSON at line %d: %s",
                    line_number,
                    exc.msg,
                )
                continue

            if not isinstance(raw, Mapping):
                counters.malformed_records += 1

                LOGGER.warning(
                    "Skipping line %d because the JSON "
                    "value is not an object",
                    line_number,
                )
                continue

            frame_index = _record_nonnegative_int(
                raw.get("frame_index")
            )

            timestamp_seconds = (
                _record_nonnegative_float(
                    raw.get("timestamp_seconds")
                )
            )

            if (
                frame_index is None
                or timestamp_seconds is None
            ):
                counters.malformed_records += 1

                LOGGER.warning(
                    "Skipping line %d due to invalid "
                    "frame_index or timestamp_seconds",
                    line_number,
                )
                continue

            yield InputRecord(
                line_number=line_number,
                frame_index=frame_index,
                timestamp_seconds=timestamp_seconds,
                metadata=raw,
            )


def _mapping_section(
    root: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    """Return a required YAML mapping section."""

    value = root.get(name)

    if not isinstance(value, Mapping):
        raise PipelineFatalError(
            f"Configuration section '{name}' "
            "must be a mapping"
        )

    return value


def _required_string(
    section: Mapping[str, Any],
    key: str,
) -> str:
    """Read a required nonempty string setting."""

    value = section.get(key)

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise PipelineFatalError(
            f"Configuration value '{key}' "
            "must be a string"
        )

    return value.strip()


def _optional_string(
    section: Mapping[str, Any],
    key: str,
    default: str,
) -> str:
    """Read an optional nonempty string setting."""

    value = section.get(key, default)

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise PipelineFatalError(
            f"Configuration value '{key}' "
            "must be a string"
        )

    return value.strip()


def _required_float(
    section: Mapping[str, Any],
    key: str,
) -> float:
    """Read a required finite numeric setting."""

    value = section.get(key)

    if isinstance(value, bool):
        raise PipelineFatalError(
            f"Configuration value '{key}' "
            "must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError):
        raise PipelineFatalError(
            f"Configuration value '{key}' "
            "must be numeric"
        ) from None

    if not math.isfinite(result):
        raise PipelineFatalError(
            f"Configuration value '{key}' "
            "must be finite"
        )

    return result


def _required_int(
    section: Mapping[str, Any],
    key: str,
) -> int:
    """Read a required integer setting."""

    value = section.get(key)

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
    ):
        raise PipelineFatalError(
            f"Configuration value '{key}' "
            "must be an integer"
        )

    return value


def _required_bool(
    section: Mapping[str, Any],
    key: str,
) -> bool:
    """Read a required boolean setting."""

    value = section.get(key)

    if not isinstance(value, bool):
        raise PipelineFatalError(
            f"Configuration value '{key}' "
            "must be a boolean"
        )

    return value


def _class_set(
    section: Mapping[str, Any],
    key: str,
) -> frozenset[str]:
    """Read and normalize a configured class-name list."""

    value = section.get(key)

    if not isinstance(value, list):
        raise PipelineFatalError(
            f"Configuration value '{key}' "
            "must be a list"
        )

    normalized: set[str] = set()

    for item in value:
        if (
            not isinstance(item, str)
            or not item.strip()
        ):
            raise PipelineFatalError(
                f"Configuration value '{key}' "
                "contains an invalid class name"
            )

        normalized.add(
            item.strip().lower()
        )

    return frozenset(normalized)


def _resolve_config_path(
    base: Path,
    configured_path: str,
) -> Path:
    """Resolve a path relative to the config file."""

    path = Path(
        configured_path
    ).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (
        base / path
    ).resolve()


def _resolve_output_path(
    base: Path,
    output_directory: Path,
    configured_path: str,
) -> Path:
    """Resolve an output path using config-directory semantics."""

    path = Path(
        configured_path
    ).expanduser()

    if path.is_absolute():
        return path.resolve()

    # A value containing a directory, such as
    # outputs/frame_decisions.jsonl, is relative to config.yaml.
    # A bare filename is placed inside output.directory.
    if path.parent != Path("."):
        return (
            base / path
        ).resolve()

    return (
        output_directory / path
    ).resolve()


def _record_nonnegative_int(
    value: Any,
) -> int | None:
    """Validate an input frame index."""

    if isinstance(value, bool):
        return None

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None

    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
    ):
        return None

    result = int(numeric)

    if result < 0:
        return None

    return result


def _record_nonnegative_float(
    value: Any,
) -> float | None:
    """Validate an input timestamp."""

    if isinstance(value, bool):
        return None

    try:
        result = float(value)
    except (TypeError, ValueError):
        return None

    if (
        not math.isfinite(result)
        or result < 0.0
    ):
        return None

    return result


def _configure_logging(level: str) -> None:
    """Configure deterministic console logging."""

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

    default_config = (
        Path(__file__).resolve().parent
        / "config.yaml"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Generate Phase 3 retention decisions "
            "from Phase 2 metadata."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=default_config,
        help=(
            "Path to config.yaml "
            "(default: beside phase3_pipeline.py)"
        ),
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> int:
    """Run the CLI and return its process exit code."""

    arguments = _parse_arguments(argv)

    try:
        config = load_config(arguments.config)

        _configure_logging(
            config.logging_level
        )

        run_pipeline(config)
        return 0

    except (PipelineFatalError, ValueError) as exc:
        if not logging.getLogger().handlers:
            _configure_logging("ERROR")

        LOGGER.error(
            "Phase 3 failed: %s",
            exc,
        )
        return 1

    except Exception:
        if not logging.getLogger().handlers:
            _configure_logging("ERROR")

        LOGGER.exception(
            "Phase 3 failed due to an unexpected error"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())