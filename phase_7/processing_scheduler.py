"""
Bounded, fair camera scheduling and isolated external-pipeline execution.

The default ``probe_only`` mode validates sources without claiming that
compression ran. ``external_pipeline`` mode invokes an explicitly configured
command separately for each camera.

A zero exit code alone is never accepted as proof of pipeline success.
External processing must also produce configured artifacts or a configured
JSON success report. Compression success is reported separately and requires
an explicitly configured JSON field.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import os
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

try:
    from .camera_manager import (
        CameraManager,
        CameraStatus,
        TERMINAL_CAMERA_STATUSES,
    )
    from .health_monitor import (
        HealthCheckResult,
        HealthMonitor,
    )
    from .resource_monitor import (
        AdmissionDecision,
        ResourceMonitor,
    )
    from .stream_manager import (
        StreamCancelledError,
        redact_source,
        utc_now,
    )
except ImportError:
    from camera_manager import (
        CameraManager,
        CameraStatus,
        TERMINAL_CAMERA_STATUSES,
    )
    from health_monitor import (
        HealthCheckResult,
        HealthMonitor,
    )
    from resource_monitor import (
        AdmissionDecision,
        ResourceMonitor,
    )
    from stream_manager import (
        StreamCancelledError,
        redact_source,
        utc_now,
    )


LOGGER = logging.getLogger(__name__)


class ProcessingSchedulerError(RuntimeError):
    """Base processing-scheduler error."""


class ProcessingConfigurationError(
    ProcessingSchedulerError
):
    """Raised when processing configuration is invalid."""


class ProcessingMode(str, Enum):
    """Supported Phase 7 processing modes."""

    PROBE_ONLY = "probe_only"
    EXTERNAL_PIPELINE = "external_pipeline"

    @classmethod
    def parse(cls, value: str) -> "ProcessingMode":
        if not isinstance(value, str):
            raise ProcessingConfigurationError(
                "processing.mode must be a string"
            )

        try:
            return cls(value.strip().lower())
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise ProcessingConfigurationError(
                f"Unsupported processing mode {value!r}; "
                f"supported values: {supported}"
            ) from exc


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    """Child-process output capture settings."""

    save_stdout: bool
    save_stderr: bool
    maximum_in_memory_characters: int

    def __post_init__(self) -> None:
        if self.maximum_in_memory_characters <= 0:
            raise ProcessingConfigurationError(
                "maximum_in_memory_characters must be positive"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "CaptureConfig":
        return cls(
            save_stdout=_required_boolean(
                value,
                "save_stdout",
                prefix="processing.command_adapter.capture",
            ),
            save_stderr=_required_boolean(
                value,
                "save_stderr",
                prefix="processing.command_adapter.capture",
            ),
            maximum_in_memory_characters=_required_integer(
                value,
                "maximum_in_memory_characters",
                prefix="processing.command_adapter.capture",
            ),
        )


@dataclass(frozen=True, slots=True)
class SuccessValidationConfig:
    """Required evidence for accepting an external pipeline run."""

    require_zero_exit_code: bool
    required_artifacts: tuple[str, ...]
    report_json: str | None
    status_fields: tuple[str, ...]
    accepted_status_values: frozenset[str]

    # Optional, intentionally absent from the default smoke-test config.
    # Configure this only after identifying the real JSON field produced by
    # the existing pipeline that explicitly proves compression success.
    compression_status_field: str | None = None
    accepted_compression_values: frozenset[str] = frozenset(
        {"success", "completed", "passed", "valid", "true"}
    )

    def __post_init__(self) -> None:
        for artifact in self.required_artifacts:
            _validate_relative_artifact_path(
                artifact,
                field_name="required_artifacts",
            )

        if self.report_json is not None:
            _validate_relative_artifact_path(
                self.report_json,
                field_name="report_json",
            )

            if not self.status_fields:
                raise ProcessingConfigurationError(
                    "status_fields cannot be empty when "
                    "report_json is configured"
                )

            if not self.accepted_status_values:
                raise ProcessingConfigurationError(
                    "accepted_status_values cannot be empty when "
                    "report_json is configured"
                )

        if self.compression_status_field is not None:
            if not self.compression_status_field.strip():
                raise ProcessingConfigurationError(
                    "compression_status_field cannot be empty"
                )

            if not self.accepted_compression_values:
                raise ProcessingConfigurationError(
                    "accepted_compression_values cannot be empty"
                )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "SuccessValidationConfig":
        prefix = (
            "processing.command_adapter.success_validation"
        )

        raw_artifacts = value.get(
            "required_artifacts",
            [],
        )

        if not isinstance(raw_artifacts, list):
            raise ProcessingConfigurationError(
                f"{prefix}.required_artifacts must be a list"
            )

        artifacts: list[str] = []

        for item in raw_artifacts:
            if not isinstance(item, str) or not item.strip():
                raise ProcessingConfigurationError(
                    f"{prefix}.required_artifacts must contain "
                    "non-empty strings"
                )
            artifacts.append(item.strip())

        report_json = value.get("report_json")

        if report_json is not None:
            if (
                not isinstance(report_json, str)
                or not report_json.strip()
            ):
                raise ProcessingConfigurationError(
                    f"{prefix}.report_json must be a non-empty "
                    "string or null"
                )
            report_json = report_json.strip()

        raw_status_fields = value.get("status_fields", [])

        if not isinstance(raw_status_fields, list):
            raise ProcessingConfigurationError(
                f"{prefix}.status_fields must be a list"
            )

        status_fields = tuple(
            _nonempty_string(
                item,
                field_name=f"{prefix}.status_fields",
            )
            for item in raw_status_fields
        )

        raw_accepted = value.get(
            "accepted_status_values",
            [],
        )

        if not isinstance(raw_accepted, list):
            raise ProcessingConfigurationError(
                f"{prefix}.accepted_status_values must be a list"
            )

        accepted = frozenset(
            _normalize_status(item)
            for item in raw_accepted
        )

        compression_field = value.get(
            "compression_status_field"
        )

        if compression_field is not None:
            compression_field = _nonempty_string(
                compression_field,
                field_name=(
                    f"{prefix}.compression_status_field"
                ),
            )

        raw_compression_values = value.get(
            "accepted_compression_values",
            [
                "success",
                "completed",
                "passed",
                "valid",
                "true",
            ],
        )

        if not isinstance(raw_compression_values, list):
            raise ProcessingConfigurationError(
                f"{prefix}.accepted_compression_values "
                "must be a list"
            )

        accepted_compression = frozenset(
            _normalize_status(item)
            for item in raw_compression_values
        )

        return cls(
            require_zero_exit_code=_required_boolean(
                value,
                "require_zero_exit_code",
                prefix=prefix,
            ),
            required_artifacts=tuple(artifacts),
            report_json=report_json,
            status_fields=status_fields,
            accepted_status_values=accepted,
            compression_status_field=compression_field,
            accepted_compression_values=(
                accepted_compression
            ),
        )


@dataclass(frozen=True, slots=True)
class CommandAdapterConfig:
    """Configuration for one isolated external command invocation."""

    working_directory: Path
    command: tuple[str, ...]
    environment: dict[str, str]
    success_validation: SuccessValidationConfig
    capture: CaptureConfig

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        config_directory: Path,
    ) -> "CommandAdapterConfig":
        prefix = "processing.command_adapter"

        raw_working_directory = value.get(
            "working_directory"
        )

        if (
            not isinstance(raw_working_directory, str)
            or not raw_working_directory.strip()
        ):
            raise ProcessingConfigurationError(
                f"{prefix}.working_directory must be a "
                "non-empty string"
            )

        working_path = Path(
            raw_working_directory
        ).expanduser()
        working_directory = (
            working_path
            if working_path.is_absolute()
            else config_directory / working_path
        ).resolve()

        raw_command = value.get("command")

        if not isinstance(raw_command, list):
            raise ProcessingConfigurationError(
                f"{prefix}.command must be a list"
            )

        command = tuple(
            _nonempty_string(
                item,
                field_name=f"{prefix}.command",
            )
            for item in raw_command
        )

        raw_environment = value.get("environment", {})

        if not isinstance(raw_environment, Mapping):
            raise ProcessingConfigurationError(
                f"{prefix}.environment must be a mapping"
            )

        environment: dict[str, str] = {}

        for key, item in raw_environment.items():
            if not isinstance(key, str) or not key:
                raise ProcessingConfigurationError(
                    f"{prefix}.environment keys must be "
                    "non-empty strings"
                )

            if not isinstance(item, str):
                raise ProcessingConfigurationError(
                    f"{prefix}.environment values must be strings"
                )

            environment[key] = item

        success_mapping = value.get(
            "success_validation"
        )

        if not isinstance(success_mapping, Mapping):
            raise ProcessingConfigurationError(
                f"{prefix}.success_validation must be a mapping"
            )

        capture_mapping = value.get("capture")

        if not isinstance(capture_mapping, Mapping):
            raise ProcessingConfigurationError(
                f"{prefix}.capture must be a mapping"
            )

        return cls(
            working_directory=working_directory,
            command=command,
            environment=environment,
            success_validation=(
                SuccessValidationConfig.from_mapping(
                    success_mapping
                )
            ),
            capture=CaptureConfig.from_mapping(
                capture_mapping
            ),
        )


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    """Validated processing mode and adapter settings."""

    mode: ProcessingMode
    command_adapter: CommandAdapterConfig

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        config_directory: Path,
    ) -> "ProcessingConfig":
        if not isinstance(value, Mapping):
            raise ProcessingConfigurationError(
                "processing must be a mapping"
            )

        mode = ProcessingMode.parse(
            value.get("mode")
        )

        adapter_mapping = value.get("command_adapter")

        if not isinstance(adapter_mapping, Mapping):
            raise ProcessingConfigurationError(
                "processing.command_adapter must be a mapping"
            )

        adapter = CommandAdapterConfig.from_mapping(
            adapter_mapping,
            config_directory=config_directory,
        )

        if mode is ProcessingMode.EXTERNAL_PIPELINE:
            if not adapter.command:
                raise ProcessingConfigurationError(
                    "external_pipeline mode requires a non-empty "
                    "processing.command_adapter.command"
                )

            validation = adapter.success_validation

            if (
                not validation.required_artifacts
                and validation.report_json is None
            ):
                raise ProcessingConfigurationError(
                    "external_pipeline mode requires explicit "
                    "success evidence: configure required_artifacts "
                    "and/or report_json"
                )

            if not adapter.working_directory.is_dir():
                raise ProcessingConfigurationError(
                    "External pipeline working directory does "
                    f"not exist: {adapter.working_directory}"
                )

        return cls(
            mode=mode,
            command_adapter=adapter,
        )


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """Bounded-concurrency and resource-admission settings."""

    max_concurrent_cameras: int
    queue_policy: str
    queue_poll_interval_seconds: float
    fail_fast: bool
    cpu_admission_threshold_percent: float
    memory_admission_threshold_percent: float
    resource_admission_timeout_seconds: float

    def __post_init__(self) -> None:
        if self.max_concurrent_cameras <= 0:
            raise ProcessingConfigurationError(
                "max_concurrent_cameras must be positive"
            )

        if self.queue_policy != "round_robin":
            raise ProcessingConfigurationError(
                "Only queue_policy=round_robin is currently "
                "supported"
            )

        if self.queue_poll_interval_seconds <= 0:
            raise ProcessingConfigurationError(
                "queue_poll_interval_seconds must be positive"
            )

        _validate_percent(
            "cpu_admission_threshold_percent",
            self.cpu_admission_threshold_percent,
        )
        _validate_percent(
            "memory_admission_threshold_percent",
            self.memory_admission_threshold_percent,
        )

        if self.resource_admission_timeout_seconds < 0:
            raise ProcessingConfigurationError(
                "resource_admission_timeout_seconds cannot "
                "be negative"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "SchedulerConfig":
        if not isinstance(value, Mapping):
            raise ProcessingConfigurationError(
                "scheduler must be a mapping"
            )

        return cls(
            max_concurrent_cameras=_required_integer(
                value,
                "max_concurrent_cameras",
                prefix="scheduler",
            ),
            queue_policy=_nonempty_string(
                value.get("queue_policy"),
                field_name="scheduler.queue_policy",
            ).lower(),
            queue_poll_interval_seconds=_required_number(
                value,
                "queue_poll_interval_seconds",
                prefix="scheduler",
            ),
            fail_fast=_required_boolean(
                value,
                "fail_fast",
                prefix="scheduler",
            ),
            cpu_admission_threshold_percent=(
                _required_number(
                    value,
                    "cpu_admission_threshold_percent",
                    prefix="scheduler",
                )
            ),
            memory_admission_threshold_percent=(
                _required_number(
                    value,
                    "memory_admission_threshold_percent",
                    prefix="scheduler",
                )
            ),
            resource_admission_timeout_seconds=(
                _required_number(
                    value,
                    "resource_admission_timeout_seconds",
                    prefix="scheduler",
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class ProcessingResult:
    """Structured result from one camera's external command."""

    camera_id: str
    mode: str
    started_at_utc: str
    completed_at_utc: str
    elapsed_seconds: float
    process_started: bool
    exit_code: int | None
    timed_out: bool
    cancelled: bool
    pipeline_success_verified: bool
    compression_success_verified: bool
    compression_verification_message: str
    safe_command: tuple[str, ...]
    working_directory: str
    required_artifacts: tuple[dict[str, Any], ...]
    report_json: str | None
    report_status_field: str | None
    report_status_value: Any
    stdout_log: str | None
    stderr_log: str | None
    stdout_tail: str
    stderr_tail: str
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "mode": self.mode,
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "process_started": self.process_started,
            "processing_invoked": self.process_started,
            "exit_code": self.exit_code,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "pipeline_success_verified": (
                self.pipeline_success_verified
            ),
            "compression_success_verified": (
                self.compression_success_verified
            ),
            "compression_verification_message": (
                self.compression_verification_message
            ),
            "command": list(self.safe_command),
            "working_directory": self.working_directory,
            "required_artifacts": [
                dict(item)
                for item in self.required_artifacts
            ],
            "report_json": self.report_json,
            "report_status_field": (
                self.report_status_field
            ),
            "report_status_value": (
                self.report_status_value
            ),
            "stdout_log": self.stdout_log,
            "stderr_log": self.stderr_log,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


@dataclass(frozen=True, slots=True)
class SchedulerRunResult:
    """Overall scheduler outcome."""

    started_at_utc: str
    completed_at_utc: str
    elapsed_seconds: float
    interrupted: bool
    submitted_camera_count: int
    completed_future_count: int
    status_counts: dict[str, int]
    required_failures: tuple[str, ...]

    @property
    def successful(self) -> bool:
        return (
            not self.interrupted
            and not self.required_failures
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "elapsed_seconds": self.elapsed_seconds,
            "interrupted": self.interrupted,
            "submitted_camera_count": (
                self.submitted_camera_count
            ),
            "completed_future_count": (
                self.completed_future_count
            ),
            "status_counts": dict(self.status_counts),
            "required_failures": list(
                self.required_failures
            ),
            "successful": self.successful,
        }


@dataclass(slots=True)
class _ActiveProcess:
    process: subprocess.Popen[str]
    last_output_monotonic: float
    last_reported_output_monotonic: float


class _OutputCollector:
    """Drain one child pipe while keeping a bounded text tail."""

    def __init__(
        self,
        *,
        path: Path | None,
        maximum_characters: int,
        replacements: tuple[tuple[str, str], ...],
        progress_callback: Any,
    ) -> None:
        self.path = path
        self.maximum_characters = maximum_characters
        self.replacements = replacements
        self.progress_callback = progress_callback

        self._tail = ""
        self._lock = threading.Lock()
        self.error: str | None = None

    def consume(
        self,
        pipe: Any,
    ) -> None:
        writer = None

        try:
            if self.path is not None:
                writer = self.path.open(
                    "w",
                    encoding="utf-8",
                    newline="\n",
                )

            for raw_line in iter(pipe.readline, ""):
                line = self._sanitize(raw_line)

                if writer is not None:
                    writer.write(line)
                    writer.flush()

                with self._lock:
                    self._tail = (
                        self._tail + line
                    )[-self.maximum_characters :]

                self.progress_callback()
        except Exception as exc:
            self.error = (
                f"Output capture failed: "
                f"{type(exc).__name__}: {exc}"
            )
        finally:
            try:
                pipe.close()
            except Exception:
                pass

            if writer is not None:
                try:
                    writer.flush()
                    writer.close()
                except OSError as exc:
                    self.error = (
                        self.error
                        or f"Unable to close output log: {exc}"
                    )

    def tail(self) -> str:
        with self._lock:
            return self._tail

    def _sanitize(self, value: str) -> str:
        result = value

        for actual, replacement in self.replacements:
            if actual and actual != replacement:
                result = result.replace(
                    actual,
                    replacement,
                )

        return result


class ProcessingScheduler:
    """
    Fairly schedule one isolated job per enabled camera.

    With one pending job per camera, configured-order round-robin scheduling is
    equivalent to fair FIFO dispatch: no camera can enqueue multiple jobs ahead
    of another camera.
    """

    def __init__(
        self,
        *,
        camera_manager: CameraManager,
        resource_monitor: ResourceMonitor,
        health_monitor: HealthMonitor,
        scheduler_config: SchedulerConfig,
        processing_config: ProcessingConfig,
        project_root: Path,
        run_output_directory: Path,
        processing_timeout_seconds: float,
        shutdown_grace_seconds: float,
        shutdown_event: threading.Event | None = None,
    ) -> None:
        if processing_timeout_seconds <= 0:
            raise ProcessingConfigurationError(
                "processing_timeout_seconds must be positive"
            )

        if shutdown_grace_seconds <= 0:
            raise ProcessingConfigurationError(
                "shutdown_grace_seconds must be positive"
            )

        self.camera_manager = camera_manager
        self.resource_monitor = resource_monitor
        self.health_monitor = health_monitor
        self.scheduler_config = scheduler_config
        self.processing_config = processing_config
        self.project_root = project_root.resolve()
        self.run_output_directory = (
            run_output_directory.resolve()
        )
        self.processing_timeout_seconds = float(
            processing_timeout_seconds
        )
        self.shutdown_grace_seconds = float(
            shutdown_grace_seconds
        )
        self.shutdown_event = (
            shutdown_event or threading.Event()
        )

        self._active_processes: dict[
            str,
            _ActiveProcess,
        ] = {}
        self._process_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._running = False

        # A process-health callback can observe meaningful child output.
        # No automatic reconnect callback is installed because restarting an
        # unknown Phase 3-6 command cannot be assumed idempotent or safe.
        self.health_monitor.set_callbacks(
            connection_health_callback=(
                self._connection_health_check
            ),
            reconnect_callback=None,
        )

    def run(
        self,
        camera_ids: tuple[str, ...] | None = None,
    ) -> SchedulerRunResult:
        with self._lifecycle_lock:
            if self._closed:
                raise ProcessingSchedulerError(
                    "Cannot run a closed scheduler"
                )

            if self._running:
                raise ProcessingSchedulerError(
                    "Scheduler is already running"
                )

            self._running = True

        started_at = utc_now()
        start_clock = time.monotonic()
        submitted = 0
        completed = 0

        try:
            selected = (
                camera_ids
                if camera_ids is not None
                else self.camera_manager.camera_ids(
                    include_disabled=False
                )
            )
            pending = deque(selected)

            for camera_id in pending:
                if (
                    self.camera_manager.status(camera_id)
                    is CameraStatus.REGISTERED
                ):
                    self.camera_manager.queue(camera_id)

            futures: dict[Future[None], str] = {}

            with ThreadPoolExecutor(
                max_workers=(
                    self.scheduler_config
                    .max_concurrent_cameras
                ),
                thread_name_prefix="phase7-camera",
            ) as executor:
                while pending or futures:
                    if self.shutdown_event.is_set():
                        break

                    while (
                        pending
                        and len(futures)
                        < self.scheduler_config
                        .max_concurrent_cameras
                        and not self.shutdown_event.is_set()
                    ):
                        camera_id = pending.popleft()

                        if self.camera_manager.is_terminal(
                            camera_id
                        ):
                            continue

                        future = executor.submit(
                            self._process_camera,
                            camera_id,
                        )
                        futures[future] = camera_id
                        submitted += 1

                    if not futures:
                        continue

                    done, _ = wait(
                        tuple(futures),
                        timeout=(
                            self.scheduler_config
                            .queue_poll_interval_seconds
                        ),
                        return_when=FIRST_COMPLETED,
                    )

                    for future in done:
                        camera_id = futures.pop(future)
                        completed += 1

                        try:
                            future.result()
                        except Exception as exc:
                            message = (
                                "Unhandled camera worker error: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            LOGGER.exception(
                                "Camera worker failed for %s",
                                camera_id,
                            )
                            self.camera_manager.mark_failed(
                                camera_id,
                                message,
                            )

                    if (
                        self.scheduler_config.fail_fast
                        and self.camera_manager
                        .required_failures()
                    ):
                        LOGGER.error(
                            "Fail-fast enabled; stopping after "
                            "required camera failure"
                        )
                        self.request_shutdown(
                            "Fail-fast required-camera failure"
                        )
                        break

                if self.shutdown_event.is_set():
                    for future in futures:
                        future.cancel()

                    self._terminate_all_processes()

            if self.shutdown_event.is_set():
                self.camera_manager.cancel_nonterminal(
                    "Cancelled during scheduler shutdown"
                )

            return SchedulerRunResult(
                started_at_utc=started_at,
                completed_at_utc=utc_now(),
                elapsed_seconds=round(
                    time.monotonic() - start_clock,
                    6,
                ),
                interrupted=self.shutdown_event.is_set(),
                submitted_camera_count=submitted,
                completed_future_count=completed,
                status_counts=(
                    self.camera_manager.status_counts()
                ),
                required_failures=(
                    self.camera_manager.required_failures()
                ),
            )
        finally:
            with self._lifecycle_lock:
                self._running = False

    def request_shutdown(
        self,
        reason: str = "Shutdown requested",
    ) -> None:
        LOGGER.warning("%s", reason)
        self.shutdown_event.set()
        self._terminate_all_processes()

    def close(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                return

            self._closed = True

        self._terminate_all_processes()

    def _process_camera(
        self,
        camera_id: str,
    ) -> None:
        if self.shutdown_event.is_set():
            self.camera_manager.mark_cancelled(camera_id)
            return

        try:
            self.camera_manager.begin_validation(
                camera_id
            )
            stream = self.camera_manager.stream(camera_id)
            inspection = stream.inspect()
            self.camera_manager.record_inspection(
                camera_id,
                inspection,
            )
        except StreamCancelledError:
            self.camera_manager.mark_cancelled(
                camera_id,
                "Source inspection cancelled during shutdown",
            )
            return
        except Exception as exc:
            self.camera_manager.mark_failed(
                camera_id,
                "Source inspection failed: "
                f"{type(exc).__name__}: {exc}",
            )
            return

        if not inspection.successful:
            return

        if (
            self.processing_config.mode
            is ProcessingMode.PROBE_ONLY
        ):
            self.camera_manager.complete_probe_only(
                camera_id
            )
            return

        self.camera_manager.begin_resource_wait(
            camera_id,
            message=(
                "Waiting for CPU and memory admission"
            ),
        )

        admission = self.resource_monitor.wait_for_capacity(
            cpu_threshold_percent=(
                self.scheduler_config
                .cpu_admission_threshold_percent
            ),
            memory_threshold_percent=(
                self.scheduler_config
                .memory_admission_threshold_percent
            ),
            timeout_seconds=(
                self.scheduler_config
                .resource_admission_timeout_seconds
            ),
            poll_interval_seconds=(
                self.scheduler_config
                .queue_poll_interval_seconds
            ),
            cancellation_event=self.shutdown_event,
        )

        if not admission.admitted:
            if self.shutdown_event.is_set():
                self.camera_manager.mark_cancelled(
                    camera_id,
                    admission.reason,
                )
            else:
                self.camera_manager.mark_failed(
                    camera_id,
                    admission.reason,
                )
            return

        self.camera_manager.end_resource_wait(
            camera_id,
            message=admission.reason,
        )
        self.camera_manager.begin_processing(
            camera_id,
            message=(
                "Starting isolated external pipeline command"
            ),
        )

        result = self._execute_external_pipeline(
            camera_id,
            admission,
        )

        current_status = self.camera_manager.status(
            camera_id
        )

        if current_status in {
            CameraStatus.FAILED,
            CameraStatus.CANCELLED,
        }:
            overridden = result.to_dict()
            overridden["pipeline_success_verified"] = False
            overridden["scheduler_acceptance"] = (
                "rejected_after_health_or_shutdown_failure"
            )
            self.camera_manager.record_processing_result(
                camera_id,
                overridden,
                successful=False,
                message=(
                    "External command result was not accepted "
                    f"because camera status is "
                    f"{current_status.value}"
                ),
            )
            return

        message = (
            "External pipeline completed with explicit "
            "success validation"
            if result.pipeline_success_verified
            else "External pipeline validation failed"
        )

        self.camera_manager.record_processing_result(
            camera_id,
            result.to_dict(),
            successful=(
                result.pipeline_success_verified
            ),
            message=message,
        )

    def _execute_external_pipeline(
        self,
        camera_id: str,
        admission: AdmissionDecision,
    ) -> ProcessingResult:
        del admission

        started_at = utc_now()
        start_clock = time.monotonic()
        adapter = self.processing_config.command_adapter
        validation = adapter.success_validation
        stream = self.camera_manager.stream(camera_id)
        output_directory = (
            self.camera_manager.output_directory(camera_id)
        )

        placeholders = {
            "camera_id": camera_id,
            "source": stream.source_argument,
            "camera_output_dir": str(output_directory),
            "run_output_dir": str(
                self.run_output_directory
            ),
            "project_root": str(self.project_root),
            "python_executable": sys.executable,
            "processing_timeout_seconds": str(
                self.processing_timeout_seconds
            ),
            "shutdown_grace_seconds": str(
                self.shutdown_grace_seconds
            ),
        }

        errors: list[str] = []
        warnings: list[str] = []

        try:
            command = tuple(
                _expand_template(item, placeholders)
                for item in adapter.command
            )
            environment_overrides = {
                key: _expand_template(value, placeholders)
                for key, value
                in adapter.environment.items()
            }
        except ProcessingConfigurationError as exc:
            errors.append(str(exc))
            return self._build_unstarted_result(
                camera_id=camera_id,
                started_at=started_at,
                start_clock=start_clock,
                errors=errors,
            )

        safe_source = redact_source(
            stream.source_argument
        )
        safe_command = tuple(
            item.replace(
                stream.source_argument,
                safe_source,
            )
            for item in command
        )

        environment = os.environ.copy()
        environment.update(environment_overrides)

        stdout_path = (
            output_directory / "adapter_stdout.log"
            if adapter.capture.save_stdout
            else None
        )
        stderr_path = (
            output_directory / "adapter_stderr.log"
            if adapter.capture.save_stderr
            else None
        )

        replacements = (
            (stream.source_argument, safe_source),
        )

        stdout_collector = _OutputCollector(
            path=stdout_path,
            maximum_characters=(
                adapter.capture
                .maximum_in_memory_characters
            ),
            replacements=replacements,
            progress_callback=lambda: (
                self._record_process_output(camera_id)
            ),
        )
        stderr_collector = _OutputCollector(
            path=stderr_path,
            maximum_characters=(
                adapter.capture
                .maximum_in_memory_characters
            ),
            replacements=replacements,
            progress_callback=lambda: (
                self._record_process_output(camera_id)
            ),
        )

        creation_flags = (
            getattr(
                subprocess,
                "CREATE_NEW_PROCESS_GROUP",
                0,
            )
            if os.name == "nt"
            else 0
        )

        try:
            process = subprocess.Popen(
                command,
                cwd=adapter.working_directory,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
                creationflags=creation_flags,
            )
        except FileNotFoundError as exc:
            errors.append(
                f"External command executable was not "
                f"found: {command[0]!r}"
            )
            return self._build_unstarted_result(
                camera_id=camera_id,
                started_at=started_at,
                start_clock=start_clock,
                errors=errors,
                safe_command=safe_command,
            )
        except OSError as exc:
            errors.append(
                f"Unable to start external command: {exc}"
            )
            return self._build_unstarted_result(
                camera_id=camera_id,
                started_at=started_at,
                start_clock=start_clock,
                errors=errors,
                safe_command=safe_command,
            )

        assert process.stdout is not None
        assert process.stderr is not None

        now = time.monotonic()

        with self._process_lock:
            self._active_processes[camera_id] = (
                _ActiveProcess(
                    process=process,
                    last_output_monotonic=now,
                    last_reported_output_monotonic=now,
                )
            )

        self.health_monitor.notify_progress(
            camera_id,
            message="External command started",
        )

        stdout_thread = threading.Thread(
            target=stdout_collector.consume,
            args=(process.stdout,),
            name=f"{camera_id}-stdout",
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=stderr_collector.consume,
            args=(process.stderr,),
            name=f"{camera_id}-stderr",
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()

        timed_out = False
        cancelled = False
        deadline = (
            start_clock
            + self.processing_timeout_seconds
        )

        while process.poll() is None:
            if self.shutdown_event.is_set():
                cancelled = True
                _terminate_process(
                    process,
                    grace_seconds=(
                        self.shutdown_grace_seconds
                    ),
                )
                break

            if time.monotonic() >= deadline:
                timed_out = True
                _terminate_process(
                    process,
                    grace_seconds=(
                        self.shutdown_grace_seconds
                    ),
                )
                break

            time.sleep(0.1)

        try:
            exit_code = process.wait(
                timeout=self.shutdown_grace_seconds
            )
        except subprocess.TimeoutExpired:
            _kill_process(process)
            exit_code = process.poll()

        stdout_thread.join(
            timeout=self.shutdown_grace_seconds
        )
        stderr_thread.join(
            timeout=self.shutdown_grace_seconds
        )

        with self._process_lock:
            self._active_processes.pop(
                camera_id,
                None,
            )

        if stdout_collector.error:
            warnings.append(stdout_collector.error)

        if stderr_collector.error:
            warnings.append(stderr_collector.error)

        if timed_out:
            errors.append(
                "External command exceeded processing timeout "
                f"of {self.processing_timeout_seconds:.3f} "
                "seconds"
            )

        if cancelled:
            errors.append(
                "External command was cancelled during shutdown"
            )

        if (
            validation.require_zero_exit_code
            and exit_code != 0
        ):
            errors.append(
                "External command returned exit code "
                f"{exit_code}; zero was required"
            )

        artifact_results, artifact_errors = (
            self._validate_required_artifacts(
                output_directory,
                validation.required_artifacts,
            )
        )
        errors.extend(artifact_errors)

        report_path: Path | None = None
        report_payload: dict[str, Any] | None = None
        report_status_field: str | None = None
        report_status_value: Any = None

        if validation.report_json is not None:
            try:
                report_path = _safe_child_path(
                    output_directory,
                    validation.report_json,
                )
                report_payload = _read_json_object(
                    report_path
                )
            except ProcessingSchedulerError as exc:
                errors.append(str(exc))
            else:
                (
                    report_status_field,
                    report_status_value,
                ) = _first_present_field(
                    report_payload,
                    validation.status_fields,
                )

                if report_status_field is None:
                    errors.append(
                        "None of the configured status fields "
                        "were present in the pipeline report: "
                        + ", ".join(
                            validation.status_fields
                        )
                    )
                elif (
                    _normalize_status(
                        report_status_value
                    )
                    not in validation
                    .accepted_status_values
                ):
                    errors.append(
                        "Pipeline report status was not "
                        "accepted: "
                        f"{report_status_field}="
                        f"{report_status_value!r}"
                    )

        compression_verified = False

        if (
            validation.compression_status_field
            is None
        ):
            compression_message = (
                "Compression success was not separately "
                "verified because compression_status_field "
                "is not configured"
            )
        elif report_payload is None:
            compression_message = (
                "Compression success could not be verified "
                "because no valid JSON report was available"
            )
        else:
            found, compression_value = _lookup_dot_path(
                report_payload,
                validation.compression_status_field,
            )

            if not found:
                compression_message = (
                    "Configured compression status field was "
                    "not present: "
                    f"{validation.compression_status_field}"
                )
            elif (
                _normalize_status(compression_value)
                in validation.accepted_compression_values
            ):
                compression_verified = True
                compression_message = (
                    "Compression success was explicitly "
                    "verified from "
                    f"{validation.compression_status_field}="
                    f"{compression_value!r}"
                )
            else:
                compression_message = (
                    "Compression status field did not contain "
                    "an accepted value: "
                    f"{validation.compression_status_field}="
                    f"{compression_value!r}"
                )

        pipeline_success = (
            not errors
            and not timed_out
            and not cancelled
            and (
                not validation.require_zero_exit_code
                or exit_code == 0
            )
        )

        return ProcessingResult(
            camera_id=camera_id,
            mode=ProcessingMode.EXTERNAL_PIPELINE.value,
            started_at_utc=started_at,
            completed_at_utc=utc_now(),
            elapsed_seconds=round(
                time.monotonic() - start_clock,
                6,
            ),
            process_started=True,
            exit_code=exit_code,
            timed_out=timed_out,
            cancelled=cancelled,
            pipeline_success_verified=pipeline_success,
            compression_success_verified=(
                compression_verified
            ),
            compression_verification_message=(
                compression_message
            ),
            safe_command=safe_command,
            working_directory=str(
                adapter.working_directory
            ),
            required_artifacts=tuple(
                artifact_results
            ),
            report_json=(
                str(report_path)
                if report_path is not None
                else None
            ),
            report_status_field=report_status_field,
            report_status_value=copy.deepcopy(
                report_status_value
            ),
            stdout_log=(
                str(stdout_path)
                if stdout_path is not None
                else None
            ),
            stderr_log=(
                str(stderr_path)
                if stderr_path is not None
                else None
            ),
            stdout_tail=stdout_collector.tail(),
            stderr_tail=stderr_collector.tail(),
            warnings=tuple(warnings),
            errors=tuple(errors),
        )

    def _build_unstarted_result(
        self,
        *,
        camera_id: str,
        started_at: str,
        start_clock: float,
        errors: list[str],
        safe_command: tuple[str, ...] = (),
    ) -> ProcessingResult:
        adapter = self.processing_config.command_adapter

        return ProcessingResult(
            camera_id=camera_id,
            mode=ProcessingMode.EXTERNAL_PIPELINE.value,
            started_at_utc=started_at,
            completed_at_utc=utc_now(),
            elapsed_seconds=round(
                time.monotonic() - start_clock,
                6,
            ),
            process_started=False,
            exit_code=None,
            timed_out=False,
            cancelled=False,
            pipeline_success_verified=False,
            compression_success_verified=False,
            compression_verification_message=(
                "Compression was not verified because the "
                "external command did not start"
            ),
            safe_command=safe_command,
            working_directory=str(
                adapter.working_directory
            ),
            required_artifacts=(),
            report_json=None,
            report_status_field=None,
            report_status_value=None,
            stdout_log=None,
            stderr_log=None,
            stdout_tail="",
            stderr_tail="",
            warnings=(),
            errors=tuple(errors),
        )

    def _validate_required_artifacts(
        self,
        output_directory: Path,
        artifacts: tuple[str, ...],
    ) -> tuple[
        list[dict[str, Any]],
        list[str],
    ]:
        results: list[dict[str, Any]] = []
        errors: list[str] = []

        for configured_path in artifacts:
            try:
                path = _safe_child_path(
                    output_directory,
                    configured_path,
                )
            except ProcessingSchedulerError as exc:
                errors.append(str(exc))
                continue

            exists = path.is_file()
            size: int | None = None

            if exists:
                try:
                    size = path.stat().st_size
                except OSError as exc:
                    errors.append(
                        f"Unable to stat required artifact "
                        f"{path}: {exc}"
                    )
                else:
                    if size <= 0:
                        errors.append(
                            f"Required artifact is empty: {path}"
                        )
            else:
                errors.append(
                    f"Required artifact is missing: {path}"
                )

            results.append(
                {
                    "configured_path": configured_path,
                    "resolved_path": str(path),
                    "exists": exists,
                    "size_bytes": size,
                    "nonempty": (
                        size is not None and size > 0
                    ),
                }
            )

        return results, errors

    def _record_process_output(
        self,
        camera_id: str,
    ) -> None:
        now = time.monotonic()

        with self._process_lock:
            active = self._active_processes.get(
                camera_id
            )

            if active is not None:
                active.last_output_monotonic = now

        self.health_monitor.notify_progress(
            camera_id,
            message="External command produced output",
        )

    def _connection_health_check(
        self,
        camera_id: str,
    ) -> HealthCheckResult:
        with self._process_lock:
            active = self._active_processes.get(
                camera_id
            )

            if active is None:
                return HealthCheckResult(
                    healthy=True,
                    message=(
                        "External process is starting or "
                        "finalizing"
                    ),
                    progress_observed=False,
                )

            exit_code = active.process.poll()

            if exit_code is not None:
                return HealthCheckResult(
                    healthy=False,
                    message=(
                        "External process exited with code "
                        f"{exit_code}"
                    ),
                    progress_observed=False,
                )

            progress_observed = (
                active.last_output_monotonic
                > active.last_reported_output_monotonic
            )

            if progress_observed:
                active.last_reported_output_monotonic = (
                    active.last_output_monotonic
                )

            return HealthCheckResult(
                healthy=True,
                message=(
                    "External process is running"
                    + (
                        " and produced new output"
                        if progress_observed
                        else " without new output"
                    )
                ),
                progress_observed=progress_observed,
            )

    def _terminate_all_processes(self) -> None:
        with self._process_lock:
            processes = [
                active.process
                for active in self._active_processes.values()
            ]

        for process in processes:
            _terminate_process(
                process,
                grace_seconds=self.shutdown_grace_seconds,
            )


def _expand_template(
    value: str,
    placeholders: Mapping[str, str],
) -> str:
    try:
        return value.format_map(
            _StrictPlaceholderMap(placeholders)
        )
    except (KeyError, ValueError) as exc:
        raise ProcessingConfigurationError(
            f"Invalid command placeholder in {value!r}: {exc}"
        ) from exc


class _StrictPlaceholderMap(dict[str, str]):
    def __missing__(self, key: str) -> str:
        raise KeyError(
            f"unknown placeholder {key!r}"
        )


def _safe_child_path(
    root: Path,
    configured_path: str,
) -> Path:
    relative = Path(configured_path)

    if relative.is_absolute():
        raise ProcessingSchedulerError(
            "Artifact paths must be relative to the camera "
            f"output directory: {configured_path!r}"
        )

    root = root.resolve()
    candidate = (root / relative).resolve()

    try:
        common = Path(
            os.path.commonpath(
                [str(root), str(candidate)]
            )
        )
    except ValueError as exc:
        raise ProcessingSchedulerError(
            "Artifact path resolved to an unexpected volume: "
            f"{configured_path!r}"
        ) from exc

    if common != root:
        raise ProcessingSchedulerError(
            "Artifact path escapes the camera output directory: "
            f"{configured_path!r}"
        )

    return candidate


def _read_json_object(
    path: Path,
) -> dict[str, Any]:
    if not path.is_file():
        raise ProcessingSchedulerError(
            f"Configured pipeline report is missing: {path}"
        )

    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcessingSchedulerError(
            f"Unable to read pipeline report {path}: {exc}"
        ) from exc

    if not isinstance(value, dict):
        raise ProcessingSchedulerError(
            f"Pipeline report must contain a JSON object: "
            f"{path}"
        )

    return value


def _first_present_field(
    payload: Mapping[str, Any],
    fields: tuple[str, ...],
) -> tuple[str | None, Any]:
    for field_name in fields:
        found, value = _lookup_dot_path(
            payload,
            field_name,
        )

        if found:
            return field_name, value

    return None, None


def _lookup_dot_path(
    payload: Mapping[str, Any],
    field_name: str,
) -> tuple[bool, Any]:
    current: Any = payload

    for part in field_name.split("."):
        if (
            not isinstance(current, Mapping)
            or part not in current
        ):
            return False, None

        current = current[part]

    return True, current


def _normalize_status(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"

    if value is None:
        return "null"

    return str(value).strip().lower()


def _validate_relative_artifact_path(
    value: str,
    *,
    field_name: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProcessingConfigurationError(
            f"{field_name} must contain non-empty paths"
        )

    path = Path(value)

    if path.is_absolute():
        raise ProcessingConfigurationError(
            f"{field_name} paths must be relative: {value!r}"
        )

    if any(part == ".." for part in path.parts):
        raise ProcessingConfigurationError(
            f"{field_name} paths cannot contain '..': "
            f"{value!r}"
        )


def _required_boolean(
    mapping: Mapping[str, Any],
    key: str,
    *,
    prefix: str,
) -> bool:
    value = mapping.get(key)

    if not isinstance(value, bool):
        raise ProcessingConfigurationError(
            f"{prefix}.{key} must be boolean"
        )

    return value


def _required_integer(
    mapping: Mapping[str, Any],
    key: str,
    *,
    prefix: str,
) -> int:
    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise ProcessingConfigurationError(
            f"{prefix}.{key} must be an integer"
        )

    return value


def _required_number(
    mapping: Mapping[str, Any],
    key: str,
    *,
    prefix: str,
) -> float:
    value = mapping.get(key)

    if isinstance(value, bool):
        raise ProcessingConfigurationError(
            f"{prefix}.{key} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ProcessingConfigurationError(
            f"{prefix}.{key} must be numeric"
        ) from None

    if not math.isfinite(result):
        raise ProcessingConfigurationError(
            f"{prefix}.{key} must be finite"
        )

    return result


def _nonempty_string(
    value: Any,
    *,
    field_name: str,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProcessingConfigurationError(
            f"{field_name} must contain non-empty strings"
        )

    return value.strip()


def _validate_percent(
    name: str,
    value: float,
) -> None:
    if not math.isfinite(value):
        raise ProcessingConfigurationError(
            f"{name} must be finite"
        )

    if value < 0 or value > 100:
        raise ProcessingConfigurationError(
            f"{name} must be between 0 and 100"
        )


def _terminate_process(
    process: subprocess.Popen[str],
    *,
    grace_seconds: float,
) -> None:
    if process.poll() is not None:
        return

    if (
        os.name == "nt"
        and hasattr(signal, "CTRL_BREAK_EVENT")
    ):
        try:
            process.send_signal(
                signal.CTRL_BREAK_EVENT
            )
            process.wait(timeout=grace_seconds)
            return
        except (
            OSError,
            PermissionError,
            subprocess.TimeoutExpired,
        ):
            _kill_process(process)
            return

    try:
        process.terminate()
        process.wait(timeout=grace_seconds)
    except (OSError, subprocess.TimeoutExpired):
        _kill_process(process)


def _kill_process(
    process: subprocess.Popen[str],
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
                timeout=5.0,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW
                    if hasattr(
                        subprocess,
                        "CREATE_NO_WINDOW",
                    )
                    else 0
                ),
            )
            process.wait(timeout=3.0)
            return
        except (
            OSError,
            subprocess.TimeoutExpired,
        ):
            pass

    try:
        process.kill()
        process.wait(timeout=3.0)
    except (OSError, subprocess.TimeoutExpired):
        pass


__all__ = [
    "CaptureConfig",
    "CommandAdapterConfig",
    "ProcessingConfig",
    "ProcessingConfigurationError",
    "ProcessingMode",
    "ProcessingResult",
    "ProcessingScheduler",
    "ProcessingSchedulerError",
    "SchedulerConfig",
    "SchedulerRunResult",
    "SuccessValidationConfig",
]
