"""Phase 4 temporal-preserving intelligent compression pipeline."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from compression_metrics import build_compression_summary
from frame_selector import (
    FrameSelector,
    FrameSelectorError,
    Priority,
)
from video_encoder import (
    EncodingConfig,
    OutputValidationConfig,
    RuntimeConfig,
    VideoEncoder,
    VideoEncoderError,
)


LOGGER = logging.getLogger("Phase4Pipeline")


class Phase4FatalError(RuntimeError):
    """Fatal configuration or orchestration error."""


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    """Required temporal-preservation behavior."""

    preserve_source_fps: bool
    preserve_duration: bool
    physically_drop_frames: bool


@dataclass(frozen=True, slots=True)
class Phase4Config:
    """Validated and path-resolved Phase 4 configuration."""

    source_video: Path
    decision_file: Path
    output_video: Path
    summary_file: Path
    encoding: EncodingConfig
    temporal: TemporalConfig
    validation: OutputValidationConfig
    fail_on_malformed_decisions: bool
    runtime: RuntimeConfig
    logging_level: str


def load_config(config_path: Path) -> Phase4Config:
    """Load Phase 4 configuration relative to config.yaml."""

    path = config_path.expanduser().resolve()

    if not path.is_file():
        raise Phase4FatalError(
            f"Configuration file not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Phase4FatalError(
            f"Invalid YAML in {path}: {exc}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise Phase4FatalError(
            f"Unable to read configuration {path}: {exc}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise Phase4FatalError(
            "Configuration root must be a YAML mapping"
        )

    paths = _section(raw, "paths")
    encoding_section = _section(raw, "encoding")
    audio_section = _section(raw, "audio")
    temporal_section = _section(raw, "temporal")
    validation_section = _section(raw, "validation")
    runtime_section = _section(raw, "runtime")

    logging_section = raw.get("logging", {})

    if not isinstance(logging_section, Mapping):
        raise Phase4FatalError(
            "Configuration section 'logging' must be a mapping"
        )

    base = path.parent

    source_video = _resolve_path(
        base,
        _string(paths, "source_video"),
    )
    decision_file = _resolve_path(
        base,
        _string(paths, "phase3_decisions"),
    )
    output_video = _resolve_path(
        base,
        _string(paths, "output_video"),
    )
    summary_file = _resolve_path(
        base,
        _string(paths, "compression_summary"),
    )

    output_root = (base / "outputs").resolve()

    for label, output_path in (
        ("paths.output_video", output_video),
        ("paths.compression_summary", summary_file),
    ):
        if not output_path.is_relative_to(output_root):
            raise Phase4FatalError(
                f"{label} must be inside the Phase 4 outputs "
                f"directory: {output_root}"
            )

    if output_video == summary_file:
        raise Phase4FatalError(
            "Output video and summary paths must be different"
        )

    if source_video in {
        output_video,
        summary_file,
    } or decision_file in {
        output_video,
        summary_file,
    }:
        raise Phase4FatalError(
            "Phase 4 outputs must not overwrite input files"
        )

    priority_section = _section(
        encoding_section,
        "priority_crf",
    )

    priority_crf: dict[Priority, int] = {}

    for priority in Priority:
        crf = _integer(
            priority_section,
            priority.value,
        )

        if not 0 <= crf <= 51:
            raise Phase4FatalError(
                f"encoding.priority_crf.{priority.value} "
                f"must be between 0 and 51"
            )

        priority_crf[priority] = crf

    audio_mode = _string(
        audio_section,
        "mode",
    ).lower()

    if audio_mode not in {"copy", "aac"}:
        raise Phase4FatalError(
            "audio.mode must be 'copy' or 'aac'"
        )

    codec = _string(
        encoding_section,
        "codec",
    )
    pixel_format = _string(
        encoding_section,
        "pixel_format",
    )
    container = _string(
        encoding_section,
        "container",
    ).lower()

    if codec != "libx265":
        raise Phase4FatalError(
            "encoding.codec must be 'libx265'"
        )

    if pixel_format != "yuv420p":
        raise Phase4FatalError(
            "encoding.pixel_format must be 'yuv420p'"
        )

    if container != "mp4":
        raise Phase4FatalError(
            "encoding.container must be 'mp4'"
        )

    if output_video.suffix.lower() != ".mp4":
        raise Phase4FatalError(
            "paths.output_video must use the .mp4 extension"
        )

    if summary_file.suffix.lower() != ".json":
        raise Phase4FatalError(
            "paths.compression_summary must use the .json extension"
        )

    encoding = EncodingConfig(
        codec=codec,
        preset=_string(
            encoding_section,
            "preset",
        ),
        pixel_format=pixel_format,
        container=container,
        priority_crf=priority_crf,
        audio_enabled=_boolean(
            audio_section,
            "enabled",
        ),
        audio_mode=audio_mode,
        audio_bitrate=_optional_string(
            audio_section,
            "bitrate",
        ),
        movflags=_optional_string(
            encoding_section,
            "movflags",
        ),
    )

    temporal = TemporalConfig(
        preserve_source_fps=_boolean(
            temporal_section,
            "preserve_source_fps",
        ),
        preserve_duration=_boolean(
            temporal_section,
            "preserve_duration",
        ),
        physically_drop_frames=_boolean(
            temporal_section,
            "physically_drop_frames",
        ),
    )

    if not temporal.preserve_source_fps:
        raise Phase4FatalError(
            "temporal.preserve_source_fps must be true"
        )

    if not temporal.preserve_duration:
        raise Phase4FatalError(
            "temporal.preserve_duration must be true"
        )

    if temporal.physically_drop_frames:
        raise Phase4FatalError(
            "temporal.physically_drop_frames must be false; "
            "Phase 3 decisions control quality, not physical "
            "frame deletion"
        )

    fps_tolerance = _nonnegative_number(
        validation_section,
        "fps_tolerance",
    )
    duration_tolerance = _nonnegative_number(
        validation_section,
        "duration_tolerance_seconds",
    )

    validation = OutputValidationConfig(
        fps_tolerance=fps_tolerance,
        duration_tolerance_seconds=duration_tolerance,
    )

    probe_timeout = _positive_number(
        runtime_section,
        "probe_timeout_seconds",
    )

    runtime = RuntimeConfig(
        ffmpeg_path=_resolve_executable_value(
            base,
            _string(runtime_section, "ffmpeg_path"),
        ),
        ffprobe_path=_resolve_executable_value(
            base,
            _string(runtime_section, "ffprobe_path"),
        ),
        overwrite=_boolean(
            runtime_section,
            "overwrite",
        ),
        keep_temporary_files=_boolean(
            runtime_section,
            "keep_temporary_files",
        ),
        probe_timeout_seconds=probe_timeout,
    )

    level = (
        _optional_string(logging_section, "level")
        or "INFO"
    ).upper()

    if level not in logging.getLevelNamesMapping():
        raise Phase4FatalError(
            f"Invalid logging level: {level}"
        )

    return Phase4Config(
        source_video=source_video,
        decision_file=decision_file,
        output_video=output_video,
        summary_file=summary_file,
        encoding=encoding,
        temporal=temporal,
        validation=validation,
        fail_on_malformed_decisions=_boolean(
            validation_section,
            "fail_on_malformed_decisions",
        ),
        runtime=runtime,
        logging_level=level,
    )


class Phase4Pipeline:
    """Orchestrate planning, encoding, validation, and metrics."""

    def __init__(self, config: Phase4Config) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        """Run the complete revised Phase 4 pipeline."""

        config = self.config

        LOGGER.info(
            "Starting temporal-preserving Phase 4 compression"
        )

        self._validate_inputs_and_outputs()

        selector = FrameSelector.load(
            config.decision_file,
            fail_on_malformed=(
                config.fail_on_malformed_decisions
            ),
        )

        encoder = VideoEncoder(
            config.encoding,
            config.runtime,
            config.validation,
        )

        source = encoder.probe(config.source_video)

        LOGGER.info(
            "Source %dx%d %.6f FPS, %d frames, %.6f seconds",
            source.width,
            source.height,
            source.average_fps,
            source.frame_count,
            source.duration_seconds,
        )

        plan = selector.build_temporal_plan(
            source_frame_count=source.frame_count,
            source_fps=source.average_fps,
            source_duration_seconds=source.duration_seconds,
        )

        result = encoder.encode(
            source,
            config.output_video,
            plan,
        )

        summary = build_compression_summary(
            source_video=config.source_video,
            decision_file=config.decision_file,
            output_video=config.output_video,
            decision_statistics=selector.statistics,
            temporal_plan=plan,
            encoding_result=result,
            encoding_config=config.encoding,
            malformed_records_are_fatal=(
                config.fail_on_malformed_decisions
            ),
        )

        summary["outputs"]["summary"] = str(
            config.summary_file
        )

        _write_json_atomic(
            config.summary_file,
            summary,
        )

        storage = summary["storage"]

        LOGGER.info(
            "Completed: output_fps=%.6f output_frames=%d "
            "duration=%.6fs storage_reduction=%s%%",
            result.output.average_fps,
            result.output.frame_count,
            result.output.duration_seconds,
            storage["storage_reduction_percent"],
        )
        LOGGER.info(
            "Output video: %s",
            config.output_video,
        )
        LOGGER.info(
            "Compression summary: %s",
            config.summary_file,
        )

        return summary

    def _validate_inputs_and_outputs(self) -> None:
        config = self.config

        for label, path in (
            ("Source video", config.source_video),
            (
                "Phase 3 decision file",
                config.decision_file,
            ),
        ):
            if not path.is_file():
                raise Phase4FatalError(
                    f"{label} not found: {path}"
                )

            if path.stat().st_size <= 0:
                raise Phase4FatalError(
                    f"{label} is empty: {path}"
                )

        if not config.runtime.overwrite:
            existing = [
                path
                for path in (
                    config.output_video,
                    config.summary_file,
                )
                if path.exists()
            ]

            if existing:
                raise Phase4FatalError(
                    "Output exists and overwrite is disabled: "
                    + ", ".join(
                        str(path)
                        for path in existing
                    )
                )

        try:
            config.output_video.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            config.summary_file.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as exc:
            raise Phase4FatalError(
                "Unable to create Phase 4 output "
                f"directory: {exc}"
            ) from exc


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        raise Phase4FatalError(
            f"Unable to write summary {path}: {exc}"
        ) from exc


def _section(
    root: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = root.get(key)

    if not isinstance(value, Mapping):
        raise Phase4FatalError(
            f"Configuration section '{key}' must be a mapping"
        )

    return value


def _string(
    section: Mapping[str, Any],
    key: str,
) -> str:
    value = section.get(key)

    if not isinstance(value, str) or not value.strip():
        raise Phase4FatalError(
            f"Configuration value '{key}' must be a string"
        )

    return value.strip()


def _optional_string(
    section: Mapping[str, Any],
    key: str,
) -> str | None:
    value = section.get(key)

    if value is None:
        return None

    if not isinstance(value, str) or not value.strip():
        raise Phase4FatalError(
            f"Configuration value '{key}' must be a "
            "nonempty string or null"
        )

    return value.strip()


def _boolean(
    section: Mapping[str, Any],
    key: str,
) -> bool:
    value = section.get(key)

    if not isinstance(value, bool):
        raise Phase4FatalError(
            f"Configuration value '{key}' must be a boolean"
        )

    return value


def _integer(
    section: Mapping[str, Any],
    key: str,
) -> int:
    value = section.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase4FatalError(
            f"Configuration value '{key}' must be an integer"
        )

    return value


def _number(
    section: Mapping[str, Any],
    key: str,
) -> float:
    value = section.get(key)

    if isinstance(value, bool):
        raise Phase4FatalError(
            f"Configuration value '{key}' must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError):
        raise Phase4FatalError(
            f"Configuration value '{key}' must be numeric"
        ) from None

    if not math.isfinite(result):
        raise Phase4FatalError(
            f"Configuration value '{key}' must be finite"
        )

    return result


def _nonnegative_number(
    section: Mapping[str, Any],
    key: str,
) -> float:
    result = _number(section, key)

    if result < 0.0:
        raise Phase4FatalError(
            f"Configuration value '{key}' cannot be negative"
        )

    return result


def _positive_number(
    section: Mapping[str, Any],
    key: str,
) -> float:
    result = _number(section, key)

    if result <= 0.0:
        raise Phase4FatalError(
            f"Configuration value '{key}' must be positive"
        )

    return result


def _resolve_path(
    base: Path,
    value: str,
) -> Path:
    path = Path(value).expanduser()

    return (
        path
        if path.is_absolute()
        else base / path
    ).resolve()


def _resolve_executable_value(
    base: Path,
    value: str,
) -> str:
    path = Path(value).expanduser()

    if path.is_absolute():
        return str(path.resolve())

    if path.parent != Path("."):
        return str((base / path).resolve())

    return value


def _configure_logging(level: str) -> None:
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
    parser = argparse.ArgumentParser(
        description=(
            "Encode all source frames using Phase 3 "
            "temporal quality."
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
            "Path to Phase 4 config.yaml "
            "(default: beside this script)"
        ),
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> int:
    arguments = _parse_arguments(argv)

    try:
        config = load_config(arguments.config)
        _configure_logging(config.logging_level)
        Phase4Pipeline(config).run()
        return 0
    except (
        Phase4FatalError,
        FrameSelectorError,
        VideoEncoderError,
        ValueError,
    ) as exc:
        if not logging.getLogger().handlers:
            _configure_logging("ERROR")

        LOGGER.error(
            "Phase 4 failed: %s",
            exc,
        )
        return 1
    except Exception:
        if not logging.getLogger().handlers:
            _configure_logging("ERROR")

        LOGGER.exception(
            "Phase 4 failed due to an unexpected error"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())