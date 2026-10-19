"""Phase 6 final integration and artifact delivery pipeline."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from artifact_manager import (
    ArtifactError,
    ArtifactManager,
)
from final_report import (
    FinalReportError,
    FinalReportWriter,
    build_final_report,
)
from integration import (
    IntegrationError,
    IntegrationPaths,
    IntegrationValidator,
    ValidationTolerances,
)


LOGGER = logging.getLogger("Phase6Pipeline")


class Phase6Error(RuntimeError):
    """Invalid configuration or orchestration failure."""


@dataclass(frozen=True, slots=True)
class OutputPaths:
    directory: Path
    video: Path
    json_report: Path
    text_report: Path
    manifest: Path


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    ffprobe_path: str
    probe_timeout_seconds: float
    overwrite: bool


@dataclass(frozen=True, slots=True)
class Phase6Config:
    inputs: IntegrationPaths
    outputs: OutputPaths
    tolerances: ValidationTolerances
    runtime: RuntimeConfig
    logging_level: str


def load_config(
    config_path: Path,
) -> Phase6Config:
    path = config_path.expanduser().resolve()

    if not path.is_file():
        raise Phase6Error(
            f"Configuration file not found: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8-sig",
        ) as stream:
            raw = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise Phase6Error(
            f"Invalid YAML in {path}: {exc}"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise Phase6Error(
            f"Unable to read configuration {path}: {exc}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise Phase6Error(
            "Configuration root must be a mapping"
        )

    paths = _section(raw, "paths")
    validation = _section(raw, "validation")
    runtime = _section(raw, "runtime")

    logging_section = raw.get("logging", {})

    if not isinstance(logging_section, Mapping):
        raise Phase6Error(
            "logging must be a mapping"
        )

    base = path.parent

    inputs = IntegrationPaths(
        source_video=_resolve(
            base,
            _string(paths, "source_video"),
        ),
        phase3_decisions=_resolve(
            base,
            _string(paths, "phase3_decisions"),
        ),
        phase4_compressed_video=_resolve(
            base,
            _string(paths, "phase4_compressed_video"),
        ),
        phase4_summary=_resolve(
            base,
            _string(paths, "phase4_summary"),
        ),
        phase5_report_json=_resolve(
            base,
            _string(paths, "phase5_report_json"),
        ),
        phase5_report_text=_resolve(
            base,
            _string(paths, "phase5_report_text"),
        ),
    )

    output_directory = _resolve(
        base,
        _string(paths, "output_directory"),
    )

    outputs = OutputPaths(
        directory=output_directory,
        video=_resolve(
            base,
            _string(paths, "final_video"),
        ),
        json_report=_resolve(
            base,
            _string(paths, "final_json_report"),
        ),
        text_report=_resolve(
            base,
            _string(paths, "final_text_report"),
        ),
        manifest=_resolve(
            base,
            _string(paths, "manifest"),
        ),
    )

    output_values = (
        outputs.video,
        outputs.json_report,
        outputs.text_report,
        outputs.manifest,
    )

    if len(set(output_values)) != len(output_values):
        raise Phase6Error(
            "Final artifact paths must be unique"
        )

    for output in output_values:
        if not output.is_relative_to(output_directory):
            raise Phase6Error(
                "Final artifact must be inside "
                f"{output_directory}: {output}"
            )

    if outputs.video.suffix.lower() != ".mp4":
        raise Phase6Error(
            "Final video must use the .mp4 extension"
        )

    if (
        outputs.json_report.suffix.lower() != ".json"
        or outputs.manifest.suffix.lower() != ".json"
    ):
        raise Phase6Error(
            "JSON report and manifest must use .json extensions"
        )

    if outputs.text_report.suffix.lower() != ".txt":
        raise Phase6Error(
            "Text report must use the .txt extension"
        )

    tolerances = ValidationTolerances(
        fps=_nonnegative(
            validation,
            "fps_tolerance",
        ),
        duration_seconds=_nonnegative(
            validation,
            "duration_tolerance_seconds",
        ),
        compression_ratio=_nonnegative(
            validation,
            "compression_ratio_tolerance",
        ),
        storage_reduction_percent=_nonnegative(
            validation,
            "storage_reduction_tolerance_percent",
        ),
    )

    executable = _string(
        runtime,
        "ffprobe_path",
    )
    executable_path = Path(executable).expanduser()

    if (
        not executable_path.is_absolute()
        and executable_path.parent != Path(".")
    ):
        executable = str(
            (base / executable_path).resolve()
        )

    runtime_config = RuntimeConfig(
        ffprobe_path=executable,
        probe_timeout_seconds=_positive(
            runtime,
            "probe_timeout_seconds",
        ),
        overwrite=_boolean(
            runtime,
            "overwrite",
        ),
    )

    level_value = logging_section.get(
        "level",
        "INFO",
    )

    if not isinstance(level_value, str):
        raise Phase6Error(
            "logging.level must be a string"
        )

    level = level_value.strip().upper()

    if level not in logging.getLevelNamesMapping():
        raise Phase6Error(
            f"Invalid logging level: {level}"
        )

    return Phase6Config(
        inputs=inputs,
        outputs=outputs,
        tolerances=tolerances,
        runtime=runtime_config,
        logging_level=level,
    )


class Phase6Pipeline:
    """Validate and publish final deliverables."""

    def __init__(
        self,
        config: Phase6Config,
    ) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        config = self.config

        LOGGER.info(
            "Starting Phase 6 final integration"
        )

        result = IntegrationValidator(
            paths=config.inputs,
            tolerances=config.tolerances,
            ffprobe_path=config.runtime.ffprobe_path,
            probe_timeout_seconds=(
                config.runtime.probe_timeout_seconds
            ),
        ).validate()

        if not result.valid:
            for error in result.errors:
                LOGGER.error(
                    "Consistency failure: %s",
                    error,
                )

            raise Phase6Error(
                "Critical integration validation failed with "
                f"{len(result.errors)} error(s); no final "
                "artifacts were published"
            )

        manager = ArtifactManager(
            config.outputs.directory,
            overwrite=config.runtime.overwrite,
        )
        manager.prepare()

        video = manager.copy_video(
            config.inputs.phase4_compressed_video,
            config.outputs.video,
        )

        artifact_paths = {
            "compressed_video": video.to_dict(),
            "json_report": str(
                config.outputs.json_report
            ),
            "text_report": str(
                config.outputs.text_report
            ),
            "manifest": str(
                config.outputs.manifest
            ),
            "source_references": {
                "phase3_decisions": str(
                    config.inputs.phase3_decisions
                ),
                "phase4_summary": str(
                    config.inputs.phase4_summary
                ),
                "phase5_report_json": str(
                    config.inputs.phase5_report_json
                ),
                "phase5_report_text": str(
                    config.inputs.phase5_report_text
                ),
            },
        }

        report = build_final_report(
            result,
            artifact_paths,
        )

        FinalReportWriter(
            config.outputs.json_report,
            config.outputs.text_report,
            overwrite=config.runtime.overwrite,
        ).write(report)

        records = [
            video,
            manager.describe_output(
                config.outputs.json_report,
                "final_json_report",
            ),
            manager.describe_output(
                config.outputs.text_report,
                "final_text_report",
            ),
            manager.describe_reference(
                config.inputs.phase3_decisions,
                "phase3_decisions",
                "phase_3",
            ),
            manager.describe_reference(
                config.inputs.phase4_summary,
                "phase4_summary",
                "phase_4",
            ),
            manager.describe_reference(
                config.inputs.phase5_report_json,
                "phase5_json_report",
                "phase_5",
            ),
            manager.describe_reference(
                config.inputs.phase5_report_text,
                "phase5_text_report",
                "phase_5",
            ),
        ]

        manager.write_manifest(
            config.outputs.manifest,
            records,
        )

        LOGGER.info(
            "Final video: %s",
            config.outputs.video,
        )
        LOGGER.info(
            "Final report: %s",
            config.outputs.json_report,
        )
        LOGGER.info(
            "Phase 6 completed successfully"
        )

        return report


def _section(
    root: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = root.get(key)

    if not isinstance(value, Mapping):
        raise Phase6Error(
            f"Configuration section '{key}' must be a mapping"
        )

    return value


def _string(
    section: Mapping[str, Any],
    key: str,
) -> str:
    value = section.get(key)

    if not isinstance(value, str) or not value.strip():
        raise Phase6Error(
            f"Configuration value '{key}' must be a nonempty string"
        )

    return value.strip()


def _boolean(
    section: Mapping[str, Any],
    key: str,
) -> bool:
    value = section.get(key)

    if not isinstance(value, bool):
        raise Phase6Error(
            f"Configuration value '{key}' must be a boolean"
        )

    return value


def _number(
    section: Mapping[str, Any],
    key: str,
) -> float:
    value = section.get(key)

    if isinstance(value, bool):
        raise Phase6Error(
            f"Configuration value '{key}' must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError):
        raise Phase6Error(
            f"Configuration value '{key}' must be numeric"
        ) from None

    if not math.isfinite(result):
        raise Phase6Error(
            f"Configuration value '{key}' must be finite"
        )

    return result


def _nonnegative(
    section: Mapping[str, Any],
    key: str,
) -> float:
    result = _number(section, key)

    if result < 0:
        raise Phase6Error(
            f"Configuration value '{key}' cannot be negative"
        )

    return result


def _positive(
    section: Mapping[str, Any],
    key: str,
) -> float:
    result = _number(section, key)

    if result <= 0:
        raise Phase6Error(
            f"Configuration value '{key}' must be positive"
        )

    return result


def _resolve(
    base: Path,
    value: str,
) -> Path:
    path = Path(value).expanduser()

    return (
        path
        if path.is_absolute()
        else base / path
    ).resolve()


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


def _arguments(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and deliver the final CCTV "
            "compression pipeline artifacts."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "config.yaml"
        ),
        help="Path to Phase 6 config.yaml",
    )

    return parser.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> int:
    arguments = _arguments(argv)

    try:
        config = load_config(arguments.config)
        _configure_logging(config.logging_level)
        Phase6Pipeline(config).run()
        return 0
    except (
        Phase6Error,
        IntegrationError,
        ArtifactError,
        FinalReportError,
    ) as exc:
        if not logging.getLogger().handlers:
            _configure_logging("ERROR")

        LOGGER.error(
            "Phase 6 failed: %s",
            exc,
        )
        return 1
    except Exception:
        if not logging.getLogger().handlers:
            _configure_logging("ERROR")

        LOGGER.exception(
            "Phase 6 failed due to an unexpected error"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())