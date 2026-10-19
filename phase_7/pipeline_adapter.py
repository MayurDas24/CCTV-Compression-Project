"""
Per-camera adapter for running CCTV Compression Phases 2 through 6.

The adapter:

- copies each phase's existing YAML configuration;
- changes only input and output path fields;
- writes generated configurations beneath the camera output directory;
- invokes every phase with the active Python interpreter;
- runs phases strictly in order;
- stops immediately after a failed phase or missing artifact;
- keeps every generated artifact isolated per camera;
- never reports compression success unless all phases complete and all
  required artifacts exist.

The --prepare-only option generates and validates configurations without
running perception, compression, or quality analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]


class PipelineAdapterError(RuntimeError):
    """Raised when the per-camera pipeline cannot complete safely."""


@dataclass(frozen=True, slots=True)
class PhasePlan:
    """One phase invocation and its required output artifacts."""

    name: str
    script_path: Path
    config_path: Path
    required_artifacts: tuple[Path, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "script_path": str(self.script_path),
            "config_path": str(self.config_path),
            "required_artifacts": [
                str(path)
                for path in self.required_artifacts
            ],
        }


class PipelineAdapter:
    """Prepare and execute an isolated Phase 2-6 camera pipeline."""

    def __init__(
        self,
        *,
        project_root: Path,
        source_video: Path,
        camera_id: str,
        camera_output_directory: Path,
        phase_timeout_seconds: float = 0.0,
        shutdown_grace_seconds: float = 15.0,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        self.project_root = (
            project_root.expanduser().resolve()
        )
        self.source_video = self._resolve_source(
            source_video
        )
        self.camera_id = self._validate_camera_id(
            camera_id
        )
        self.camera_output_directory = (
            camera_output_directory
            .expanduser()
            .resolve()
        )

        self.phase_timeout_seconds = float(
            phase_timeout_seconds
        )
        self.shutdown_grace_seconds = float(
            shutdown_grace_seconds
        )
        self.shutdown_event = (
            shutdown_event
            if shutdown_event is not None
            else threading.Event()
        )

        if self.phase_timeout_seconds < 0:
            raise PipelineAdapterError(
                "phase_timeout_seconds cannot be negative"
            )

        if self.shutdown_grace_seconds <= 0:
            raise PipelineAdapterError(
                "shutdown_grace_seconds must be positive"
            )

        self.logs_directory = (
            self.camera_output_directory / "logs"
        )
        self.final_directory = (
            self.camera_output_directory / "final"
        )
        self.result_path = (
            self.camera_output_directory
            / "pipeline_adapter_result.json"
        )

        self.phase_directories = {
            phase_name: (
                self.camera_output_directory / phase_name
            )
            for phase_name in (
                "phase_2",
                "phase_3",
                "phase_4",
                "phase_5",
                "phase_6",
            )
        }

    def prepare(self) -> tuple[PhasePlan, ...]:
        """
        Generate isolated per-camera configurations.

        This method does not execute any phase.
        """

        self._validate_project_inputs()
        self._create_directories()

        phase_2_output = (
            self.phase_directories["phase_2"]
            / "outputs"
        )
        phase_3_output = (
            self.phase_directories["phase_3"]
            / "outputs"
        )
        phase_4_output = (
            self.phase_directories["phase_4"]
            / "outputs"
        )
        phase_5_output = (
            self.phase_directories["phase_5"]
            / "outputs"
        )

        perception_metadata = (
            phase_2_output / "perception.jsonl"
        )
        perception_summary = (
            phase_2_output / "summary.json"
        )

        frame_decisions = (
            phase_3_output / "frame_decisions.jsonl"
        )
        phase_3_summary = (
            phase_3_output / "summary.json"
        )

        compressed_video = (
            phase_4_output / "compressed_video.mp4"
        )
        compression_summary = (
            phase_4_output
            / "compression_summary.json"
        )

        phase_5_json = (
            phase_5_output / "phase5_report.json"
        )
        phase_5_text = (
            phase_5_output / "phase5_report.txt"
        )

        final_video = (
            self.final_directory
            / "compressed_video.mp4"
        )
        final_json = (
            self.final_directory
            / "phase_6_final_report.json"
        )
        final_text = (
            self.final_directory
            / "phase_6_final_report.txt"
        )
        final_manifest = (
            self.final_directory
            / "phase_6_artifact_manifest.json"
        )

        phase_2_config = self._load_template(
            "phase_2"
        )
        self._set_path(
            phase_2_config,
            "input",
            "source",
            self.source_video,
            phase_name="phase_2",
        )
        self._set_path(
            phase_2_config,
            "output",
            "directory",
            phase_2_output,
            phase_name="phase_2",
        )
        self._set_path(
            phase_2_config,
            "output",
            "metadata_file",
            perception_metadata,
            phase_name="phase_2",
        )
        self._set_path(
            phase_2_config,
            "output",
            "summary_file",
            perception_summary,
            phase_name="phase_2",
        )

        phase_3_config = self._load_template(
            "phase_3"
        )
        self._set_path(
            phase_3_config,
            "input",
            "metadata_file",
            perception_metadata,
            phase_name="phase_3",
        )
        self._set_path(
            phase_3_config,
            "output",
            "directory",
            phase_3_output,
            phase_name="phase_3",
        )
        self._set_path(
            phase_3_config,
            "output",
            "decisions_file",
            frame_decisions,
            phase_name="phase_3",
        )
        self._set_path(
            phase_3_config,
            "output",
            "summary_file",
            phase_3_summary,
            phase_name="phase_3",
        )

        phase_4_config = self._load_template(
            "phase_4"
        )
        self._set_path(
            phase_4_config,
            "paths",
            "source_video",
            self.source_video,
            phase_name="phase_4",
        )
        self._set_path(
            phase_4_config,
            "paths",
            "phase3_decisions",
            frame_decisions,
            phase_name="phase_4",
        )
        self._set_path(
            phase_4_config,
            "paths",
            "output_video",
            compressed_video,
            phase_name="phase_4",
        )
        self._set_path(
            phase_4_config,
            "paths",
            "compression_summary",
            compression_summary,
            phase_name="phase_4",
        )

        phase_5_config = self._load_template(
            "phase_5"
        )
        self._set_path(
            phase_5_config,
            "inputs",
            "source_video",
            self.source_video,
            phase_name="phase_5",
        )
        self._set_path(
            phase_5_config,
            "inputs",
            "phase3_decisions",
            frame_decisions,
            phase_name="phase_5",
        )
        self._set_path(
            phase_5_config,
            "inputs",
            "phase4_compressed_video",
            compressed_video,
            phase_name="phase_5",
        )
        self._set_path(
            phase_5_config,
            "inputs",
            "phase4_compression_summary",
            compression_summary,
            phase_name="phase_5",
        )
        self._set_path(
            phase_5_config,
            "output",
            "directory",
            phase_5_output,
            phase_name="phase_5",
        )
        self._set_path(
            phase_5_config,
            "output",
            "json_report",
            phase_5_json,
            phase_name="phase_5",
        )
        self._set_path(
            phase_5_config,
            "output",
            "text_report",
            phase_5_text,
            phase_name="phase_5",
        )

        phase_6_config = self._load_template(
            "phase_6"
        )
        self._set_path(
            phase_6_config,
            "paths",
            "source_video",
            self.source_video,
            phase_name="phase_6",
        )
        self._set_path(
            phase_6_config,
            "paths",
            "phase3_decisions",
            frame_decisions,
            phase_name="phase_6",
        )
        self._set_path(
            phase_6_config,
            "paths",
            "phase4_compressed_video",
            compressed_video,
            phase_name="phase_6",
        )
        self._set_path(
            phase_6_config,
            "paths",
            "phase4_summary",
            compression_summary,
            phase_name="phase_6",
        )
        self._set_path(
            phase_6_config,
            "paths",
            "phase5_report_json",
            phase_5_json,
            phase_name="phase_6",
        )
        self._set_path(
            phase_6_config,
            "paths",
            "phase5_report_text",
            phase_5_text,
            phase_name="phase_6",
        )
        self._set_path(
            phase_6_config,
            "paths",
            "output_directory",
            self.final_directory,
            phase_name="phase_6",
        )
        self._set_path(
            phase_6_config,
            "paths",
            "final_video",
            final_video,
            phase_name="phase_6",
        )
        self._set_path(
            phase_6_config,
            "paths",
            "final_json_report",
            final_json,
            phase_name="phase_6",
        )
        self._set_path(
            phase_6_config,
            "paths",
            "final_text_report",
            final_text,
            phase_name="phase_6",
        )
        self._set_path(
            phase_6_config,
            "paths",
            "manifest",
            final_manifest,
            phase_name="phase_6",
        )

        generated_configs = {
            "phase_2": phase_2_config,
            "phase_3": phase_3_config,
            "phase_4": phase_4_config,
            "phase_5": phase_5_config,
            "phase_6": phase_6_config,
        }

        for phase_name, payload in generated_configs.items():
            config_path = (
                self.phase_directories[phase_name]
                / "config.yaml"
            )
            _write_yaml_atomic(
                config_path,
                payload,
            )

        return (
            PhasePlan(
                name="phase_2",
                script_path=(
                    self.project_root
                    / "phase_2"
                    / "perception.py"
                ),
                config_path=(
                    self.phase_directories["phase_2"]
                    / "config.yaml"
                ),
                required_artifacts=(
                    perception_metadata,
                    perception_summary,
                ),
            ),
            PhasePlan(
                name="phase_3",
                script_path=(
                    self.project_root
                    / "phase_3"
                    / "phase3_pipeline.py"
                ),
                config_path=(
                    self.phase_directories["phase_3"]
                    / "config.yaml"
                ),
                required_artifacts=(
                    frame_decisions,
                    phase_3_summary,
                ),
            ),
            PhasePlan(
                name="phase_4",
                script_path=(
                    self.project_root
                    / "phase_4"
                    / "phase4_pipeline.py"
                ),
                config_path=(
                    self.phase_directories["phase_4"]
                    / "config.yaml"
                ),
                required_artifacts=(
                    compressed_video,
                    compression_summary,
                ),
            ),
            PhasePlan(
                name="phase_5",
                script_path=(
                    self.project_root
                    / "phase_5"
                    / "phase5_pipeline.py"
                ),
                config_path=(
                    self.phase_directories["phase_5"]
                    / "config.yaml"
                ),
                required_artifacts=(
                    phase_5_json,
                    phase_5_text,
                ),
            ),
            PhasePlan(
                name="phase_6",
                script_path=(
                    self.project_root
                    / "phase_6"
                    / "phase6_pipeline.py"
                ),
                config_path=(
                    self.phase_directories["phase_6"]
                    / "config.yaml"
                ),
                required_artifacts=(
                    final_video,
                    final_json,
                    final_text,
                    final_manifest,
                ),
            ),
        )

    def run(self) -> dict[str, Any]:
        """Run Phases 2-6 and return the final adapter report."""

        started_at = _utc_now()
        started_clock = time.monotonic()
        plans: tuple[PhasePlan, ...] = ()
        phase_results: list[dict[str, Any]] = []
        failure: str | None = None

        try:
            plans = self.prepare()
            self._reject_existing_result_artifacts(
                plans
            )

            for plan in plans:
                if self.shutdown_event.is_set():
                    raise PipelineAdapterError(
                        "Pipeline cancelled before "
                        f"{plan.name} started"
                    )

                result = self._execute_phase(plan)
                phase_results.append(result)

                if result["status"] != "completed":
                    errors = result.get("errors", [])
                    detail = (
                        "; ".join(str(item) for item in errors)
                        if errors
                        else "unknown phase failure"
                    )
                    raise PipelineAdapterError(
                        f"{plan.name} failed: {detail}"
                    )

        except PipelineAdapterError as exc:
            failure = str(exc)

        success = (
            failure is None
            and len(phase_results) == len(plans)
            and all(
                result.get("status") == "completed"
                for result in phase_results
            )
        )

        report = {
            "schema_version": "1.0",
            "phase": "phase_7_camera_pipeline_adapter",
            "camera_id": self.camera_id,
            "status": (
                "completed" if success else "failed"
            ),
            "pipeline_success_verified": success,
            "compression_success_verified": success,
            "source_video": str(self.source_video),
            "camera_output_directory": str(
                self.camera_output_directory
            ),
            "python_executable": sys.executable,
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": round(
                time.monotonic() - started_clock,
                6,
            ),
            "generated_configs": [
                str(plan.config_path)
                for plan in plans
            ],
            "phase_results": phase_results,
            "final_artifacts": (
                _describe_artifacts(
                    plans[-1].required_artifacts
                )
                if plans
                else []
            ),
            "error": failure,
        }

        try:
            self.camera_output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
            _write_json_atomic(
                self.result_path,
                report,
            )
        except OSError as exc:
            raise PipelineAdapterError(
                "Unable to write pipeline adapter report "
                f"{self.result_path}: {exc}"
            ) from exc

        if not success:
            raise PipelineAdapterError(
                failure or "Camera pipeline did not complete"
            )

        return report

    def _execute_phase(
        self,
        plan: PhasePlan,
    ) -> dict[str, Any]:
        started_at = _utc_now()
        started_clock = time.monotonic()

        stdout_path = (
            self.logs_directory
            / f"{plan.name}_stdout.log"
        )
        stderr_path = (
            self.logs_directory
            / f"{plan.name}_stderr.log"
        )

        command = (
            sys.executable,
            "-u",
            str(plan.script_path),
            "--config",
            str(plan.config_path),
        )

        errors: list[str] = []
        exit_code: int | None = None
        timed_out = False
        cancelled = False
        process_started = False

        print(
            f"[{self.camera_id}] Starting {plan.name}",
            flush=True,
        )

        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )

        try:
            with stdout_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stdout_stream, stderr_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stderr_stream:
                try:
                    process = subprocess.Popen(
                        command,
                        cwd=plan.script_path.parent,
                        stdout=stdout_stream,
                        stderr=stderr_stream,
                        text=True,
                        encoding="utf-8",
                        errors="replace",
                        shell=False,
                        creationflags=creation_flags,
                    )
                    process_started = True
                except OSError as exc:
                    errors.append(
                        "Unable to start phase process: "
                        f"{exc}"
                    )
                else:
                    deadline = (
                        started_clock
                        + self.phase_timeout_seconds
                        if self.phase_timeout_seconds > 0
                        else None
                    )

                    while process.poll() is None:
                        if self.shutdown_event.is_set():
                            cancelled = True
                            errors.append(
                                "Phase was cancelled during "
                                "shutdown"
                            )
                            _terminate_process_tree(
                                process,
                                grace_seconds=(
                                    self.shutdown_grace_seconds
                                ),
                            )
                            break

                        if (
                            deadline is not None
                            and time.monotonic() >= deadline
                        ):
                            timed_out = True
                            errors.append(
                                "Phase exceeded timeout of "
                                f"{self.phase_timeout_seconds:.3f} "
                                "seconds"
                            )
                            _terminate_process_tree(
                                process,
                                grace_seconds=(
                                    self.shutdown_grace_seconds
                                ),
                            )
                            break

                        time.sleep(0.2)

                    try:
                        exit_code = process.wait(
                            timeout=(
                                self.shutdown_grace_seconds
                            )
                        )
                    except subprocess.TimeoutExpired:
                        _terminate_process_tree(
                            process,
                            grace_seconds=(
                                self.shutdown_grace_seconds
                            ),
                        )
                        exit_code = process.poll()

        except OSError as exc:
            errors.append(
                f"Unable to write phase logs: {exc}"
            )

        if process_started and exit_code != 0:
            errors.append(
                f"Phase process returned exit code {exit_code}"
            )

        artifacts = _describe_artifacts(
            plan.required_artifacts
        )

        missing_artifacts = [
            artifact["path"]
            for artifact in artifacts
            if (
                not artifact["exists"]
                or not artifact["is_file"]
                or not artifact["nonempty"]
            )
        ]

        if missing_artifacts:
            errors.append(
                "Required artifacts are missing or empty: "
                + ", ".join(missing_artifacts)
            )

        completed = (
            process_started
            and exit_code == 0
            and not timed_out
            and not cancelled
            and not errors
        )

        result = {
            "phase": plan.name,
            "status": (
                "completed" if completed else "failed"
            ),
            "started_at_utc": started_at,
            "completed_at_utc": _utc_now(),
            "elapsed_seconds": round(
                time.monotonic() - started_clock,
                6,
            ),
            "process_started": process_started,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "command": list(command),
            "working_directory": str(
                plan.script_path.parent
            ),
            "config_path": str(plan.config_path),
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
            "required_artifacts": artifacts,
            "errors": errors,
        }

        print(
            f"[{self.camera_id}] {plan.name}: "
            f"{result['status']}",
            flush=True,
        )

        return result

    def _reject_existing_result_artifacts(
        self,
        plans: Sequence[PhasePlan],
    ) -> None:
        existing = [
            path
            for plan in plans
            for path in plan.required_artifacts
            if path.exists()
        ]

        if self.result_path.exists():
            existing.append(self.result_path)

        if existing:
            raise PipelineAdapterError(
                "Refusing to use existing result artifacts in "
                "the camera output directory. Start a new "
                "Phase 7 run directory instead: "
                + ", ".join(str(path) for path in existing)
            )

    def _validate_project_inputs(self) -> None:
        if not self.project_root.is_dir():
            raise PipelineAdapterError(
                "Project root does not exist or is not a "
                f"directory: {self.project_root}"
            )

        if (
            not self.source_video.is_file()
            or self.source_video.stat().st_size <= 0
        ):
            raise PipelineAdapterError(
                "Source video does not exist or is empty: "
                f"{self.source_video}"
            )

        required_files = (
            self.project_root
            / "phase_2"
            / "perception.py",
            self.project_root
            / "phase_2"
            / "config.yaml",
            self.project_root
            / "phase_3"
            / "phase3_pipeline.py",
            self.project_root
            / "phase_3"
            / "config.yaml",
            self.project_root
            / "phase_4"
            / "phase4_pipeline.py",
            self.project_root
            / "phase_4"
            / "config.yaml",
            self.project_root
            / "phase_5"
            / "phase5_pipeline.py",
            self.project_root
            / "phase_5"
            / "config.yaml",
            self.project_root
            / "phase_6"
            / "phase6_pipeline.py",
            self.project_root
            / "phase_6"
            / "config.yaml",
        )

        missing = [
            str(path)
            for path in required_files
            if not path.is_file()
        ]

        if missing:
            raise PipelineAdapterError(
                "Required project files are missing: "
                + ", ".join(missing)
            )

    def _create_directories(self) -> None:
        directories = [
            self.camera_output_directory,
            self.logs_directory,
            self.final_directory,
        ]

        for phase_directory in (
            self.phase_directories.values()
        ):
            directories.append(phase_directory)
            directories.append(
                phase_directory / "outputs"
            )

        try:
            for directory in directories:
                directory.mkdir(
                    parents=True,
                    exist_ok=True,
                )
        except OSError as exc:
            raise PipelineAdapterError(
                "Unable to create isolated camera "
                f"directories: {exc}"
            ) from exc

    def _load_template(
        self,
        phase_name: str,
    ) -> dict[str, Any]:
        if yaml is None:
            raise PipelineAdapterError(
                "PyYAML is required. Install it with: "
                "python -m pip install PyYAML"
            )

        path = (
            self.project_root
            / phase_name
            / "config.yaml"
        )

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as stream:
                payload = yaml.safe_load(stream)
        except (OSError, yaml.YAMLError) as exc:
            raise PipelineAdapterError(
                f"Unable to load {phase_name} template "
                f"{path}: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise PipelineAdapterError(
                f"{phase_name} config must contain a YAML "
                "mapping"
            )

        return payload

    @staticmethod
    def _set_path(
        payload: dict[str, Any],
        section_name: str,
        field_name: str,
        path: Path,
        *,
        phase_name: str,
    ) -> None:
        section = payload.get(section_name)

        if not isinstance(section, dict):
            raise PipelineAdapterError(
                f"{phase_name} config section "
                f"{section_name!r} is missing or invalid"
            )

        if field_name not in section:
            raise PipelineAdapterError(
                f"{phase_name} config field "
                f"{section_name}.{field_name} is missing"
            )

        resolved_path = path.expanduser().resolve()
        section[field_name] = str(resolved_path)

    def _resolve_source(
        self,
        source: Path,
    ) -> Path:
        candidate = source.expanduser()

        if not candidate.is_absolute():
            candidate = self.project_root / candidate

        return candidate.resolve()

    @staticmethod
    def _validate_camera_id(
        camera_id: str,
    ) -> str:
        normalized = camera_id.strip()

        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}",
            normalized,
        ):
            raise PipelineAdapterError(
                "camera_id may contain only letters, numbers, "
                "periods, underscores, and hyphens"
            )

        return normalized


def _describe_artifacts(
    paths: Sequence[Path],
) -> list[dict[str, Any]]:
    descriptions: list[dict[str, Any]] = []

    for path in paths:
        exists = path.exists()
        is_file = path.is_file()
        size_bytes: int | None = None

        if is_file:
            try:
                size_bytes = path.stat().st_size
            except OSError:
                size_bytes = None

        descriptions.append(
            {
                "path": str(path),
                "exists": exists,
                "is_file": is_file,
                "nonempty": (
                    size_bytes is not None
                    and size_bytes > 0
                ),
                "size_bytes": size_bytes,
            }
        )

    return descriptions


def _write_yaml_atomic(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    if yaml is None:
        raise PipelineAdapterError(
            "PyYAML is required. Install it with: "
            "python -m pip install PyYAML"
        )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            yaml.safe_dump(
                dict(payload),
                stream,
                sort_keys=False,
                allow_unicode=True,
            )
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    except (OSError, yaml.YAMLError) as exc:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass

        raise PipelineAdapterError(
            f"Unable to write generated config {path}: {exc}"
        ) from exc


def _write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:
        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            json.dump(
                dict(payload),
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    except Exception:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        raise


def _terminate_process_tree(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float,
) -> None:
    if process.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(process.pid),
                    "/T",
                    "/F",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=grace_seconds,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if hasattr(
                        subprocess,
                        "CREATE_NO_WINDOW",
                    )
                    else 0
                ),
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            process.terminate()
            process.wait(timeout=grace_seconds)
        except (OSError, subprocess.TimeoutExpired):
            pass

    if process.poll() is None:
        try:
            process.kill()
            process.wait(timeout=3.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one isolated local video through CCTV "
            "Compression Phases 2 through 6."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        required=True,
        help="Absolute or relative CCTV_Compression project root.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Local source video. Relative paths use project root.",
    )
    parser.add_argument(
        "--camera-id",
        required=True,
        help="Camera identifier used in reports and logs.",
    )
    parser.add_argument(
        "--camera-output-dir",
        type=Path,
        required=True,
        help=(
            "Unique Phase 7 output directory for this camera."
        ),
    )
    parser.add_argument(
        "--phase-timeout-seconds",
        type=float,
        default=0.0,
        help=(
            "Maximum duration for each phase. Zero disables "
            "the per-phase timeout. The Phase 7 scheduler still "
            "enforces its overall processing timeout."
        ),
    )
    parser.add_argument(
        "--shutdown-grace-seconds",
        type=float,
        default=15.0,
        help="Seconds allowed for child-process shutdown.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help=(
            "Generate and validate isolated configurations "
            "without running any phase."
        ),
    )

    return parser.parse_args(argv)


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = _parse_arguments(argv)
    shutdown_event = threading.Event()
    previous_handlers: dict[int, Any] = {}

    def request_shutdown(
        signum: int,
        frame: Any,
    ) -> None:
        del signum, frame
        shutdown_event.set()

    for signal_name in (
    "SIGINT",
    "SIGTERM",
    "SIGBREAK",
):
        signal_number = getattr(
            signal,
            signal_name,
            None,
        )

        if signal_number is None:
            continue

        try:
            previous_handlers[signal_number] = (
                signal.getsignal(signal_number)
            )
            signal.signal(
                signal_number,
                request_shutdown,
            )
        except (OSError, ValueError):
            pass

    try:
        adapter = PipelineAdapter(
            project_root=arguments.project_root,
            source_video=arguments.source,
            camera_id=arguments.camera_id,
            camera_output_directory=(
                arguments.camera_output_dir
            ),
            phase_timeout_seconds=(
                arguments.phase_timeout_seconds
            ),
            shutdown_grace_seconds=(
                arguments.shutdown_grace_seconds
            ),
            shutdown_event=shutdown_event,
        )

        if arguments.prepare_only:
            plans = adapter.prepare()
            print(
                json.dumps(
                    {
                        "status": "prepared",
                        "camera_id": adapter.camera_id,
                        "processing_invoked": False,
                        "compression_ran": False,
                        "generated_configs": [
                            plan.to_dict()
                            for plan in plans
                        ],
                    },
                    indent=2,
                )
            )
            return 0

        report = adapter.run()
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "camera_id": report["camera_id"],
                    "pipeline_success_verified": (
                        report[
                            "pipeline_success_verified"
                        ]
                    ),
                    "compression_success_verified": (
                        report[
                            "compression_success_verified"
                        ]
                    ),
                    "report": str(adapter.result_path),
                },
                indent=2,
            )
        )
        return 0

    except PipelineAdapterError as exc:
        print(
            f"Phase 7 camera pipeline failed: {exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            "Phase 7 camera pipeline failed unexpectedly: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    finally:
        for signal_number, previous_handler in (
            previous_handlers.items()
        ):
            try:
                signal.signal(
                    signal_number,
                    previous_handler,
                )
            except (OSError, ValueError):
                pass


if __name__ == "__main__":
    sys.exit(main())