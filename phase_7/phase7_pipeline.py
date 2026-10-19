"""
Main executable for CCTV Compression Phase 7.

Run from the phase_7 directory:

    python -u phase7_pipeline.py

Run the safe single-file smoke test:

    python -u phase7_pipeline.py --smoke-test

The default mode is probe-only. Probe-only mode verifies orchestration, source
readability, FFprobe integration, scheduling, resource monitoring, health
tracking, isolated output directories, and reporting. It does not claim that
compression ran.

External processing is enabled only through an explicit command adapter with
artifact or JSON-report success validation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sys
import threading
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

try:
    from .camera_manager import CameraManager
    from .health_monitor import (
        HealthMonitor,
        ReconnectPolicy,
    )
    from .processing_scheduler import (
        ProcessingConfig,
        ProcessingConfigurationError,
        ProcessingMode,
        ProcessingScheduler,
        SchedulerConfig,
        SchedulerRunResult,
    )
    from .resource_monitor import (
        ResourceMonitor,
        ResourceMonitorDependencyError,
    )
    from .stream_manager import (
        SourceType,
        StreamConfigurationError,
        StreamDefinition,
        StreamManager,
        utc_now,
    )
except ImportError:
    from camera_manager import CameraManager
    from health_monitor import (
        HealthMonitor,
        ReconnectPolicy,
    )
    from processing_scheduler import (
        ProcessingConfig,
        ProcessingConfigurationError,
        ProcessingMode,
        ProcessingScheduler,
        SchedulerConfig,
        SchedulerRunResult,
    )
    from resource_monitor import (
        ResourceMonitor,
        ResourceMonitorDependencyError,
    )
    from stream_manager import (
        SourceType,
        StreamConfigurationError,
        StreamDefinition,
        StreamManager,
        utc_now,
    )


LOGGER = logging.getLogger("phase_7")


class Phase7Error(RuntimeError):
    """Base Phase 7 error."""


class Phase7ConfigurationError(Phase7Error):
    """Raised when config.yaml is invalid."""


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    ffprobe_path: str
    ffprobe_timeout_seconds: float
    processing_timeout_seconds: float
    shutdown_grace_seconds: float
    log_level: str


@dataclass(frozen=True, slots=True)
class GPUMonitorConfig:
    enabled: bool
    nvidia_smi_path: str
    query_timeout_seconds: float


@dataclass(frozen=True, slots=True)
class ResourceMonitorConfig:
    enabled: bool
    sample_interval_seconds: float
    include_process_metrics: bool
    gpu: GPUMonitorConfig


@dataclass(frozen=True, slots=True)
class HealthMonitorConfig:
    enabled: bool
    check_interval_seconds: float
    default_stale_timeout_seconds: float
    maximum_consecutive_health_failures: int
    stale_detection_source_types: frozenset[SourceType]


@dataclass(frozen=True, slots=True)
class ReportingConfig:
    json_filename: str
    text_filename: str
    write_camera_reports: bool
    redact_uri_credentials: bool
    include_resource_samples: bool


@dataclass(frozen=True, slots=True)
class Phase7Config:
    schema_version: str
    phase: str
    config_path: Path
    config_directory: Path
    project_root: Path
    output_root: Path
    runtime: RuntimeConfig
    processing: ProcessingConfig
    scheduler: SchedulerConfig
    resource_monitor: ResourceMonitorConfig
    health_monitor: HealthMonitorConfig
    reconnect_policy: ReconnectPolicy
    reporting: ReportingConfig
    sources: tuple[StreamDefinition, ...]


@dataclass(frozen=True, slots=True)
class RunContext:
    run_id: str
    run_output_directory: Path
    started_at_utc: str
    smoke_test: bool
    selected_camera_ids: tuple[str, ...]


class Phase7Application:
    """Coordinate one complete Phase 7 run."""

    def __init__(
        self,
        config: Phase7Config,
        *,
        smoke_test: bool,
        selected_camera_ids: Sequence[str] | None,
        force_probe_only: bool,
        force_external_pipeline: bool = False,
    ) -> None:
        self.original_config = config
        self.config = _apply_runtime_overrides(
            config,
            smoke_test=smoke_test,
            selected_camera_ids=selected_camera_ids,
            force_probe_only=force_probe_only,
            force_external_pipeline=(
                force_external_pipeline
            ),
        )
        self.smoke_test = smoke_test
        self.shutdown_event = threading.Event()

        self.stream_manager: StreamManager | None = None
        self.camera_manager: CameraManager | None = None
        self.resource_monitor: ResourceMonitor | None = None
        self.health_monitor: HealthMonitor | None = None
        self.scheduler: ProcessingScheduler | None = None

        self._previous_signal_handlers: dict[
            signal.Signals,
            Any,
        ] = {}

    def run(self) -> int:
        context = _create_run_context(
            self.config,
            smoke_test=self.smoke_test,
        )
        _configure_logging(
            self.config.runtime.log_level,
            context.run_output_directory
            / "phase7.log",
        )

        LOGGER.info(
            "Phase 7 run started: %s",
            context.run_id,
        )
        LOGGER.info(
            "Mode: %s",
            self.config.processing.mode.value,
        )
        LOGGER.info(
            "Run output directory: %s",
            context.run_output_directory,
        )

        self._install_signal_handlers()

        scheduler_result: SchedulerRunResult | None = None
        fatal_errors: list[str] = []

        try:
            self._initialize_components(context)
            assert self.resource_monitor is not None
            assert self.health_monitor is not None
            assert self.scheduler is not None
            assert self.camera_manager is not None

            self.resource_monitor.start()
            self.health_monitor.start()

            scheduler_result = self.scheduler.run(
                self.camera_manager.camera_ids(
                    include_disabled=False
                )
            )
        except KeyboardInterrupt:
            fatal_errors.append(
                "Phase 7 was interrupted by the user"
            )
            self._request_shutdown(
                "Keyboard interrupt received"
            )
        except Exception as exc:
            fatal_errors.append(
                f"{type(exc).__name__}: {exc}"
            )
            LOGGER.exception(
                "Phase 7 failed with an unexpected error"
            )
            self._request_shutdown(
                "Fatal Phase 7 error"
            )
        finally:
            self._stop_monitors()
            self._restore_signal_handlers()

        try:
            report = self._build_report(
                context=context,
                scheduler_result=scheduler_result,
                fatal_errors=fatal_errors,
            )

            camera_report_warnings = (
                self._write_camera_reports(report)
            )

            if camera_report_warnings:
                report["warnings"].extend(
                    camera_report_warnings
                )

                if (
                    report["status"] == "success"
                    and report["warnings"]
                ):
                    report["status"] = (
                        "completed_with_warnings"
                    )

            json_path = (
                context.run_output_directory
                / self.config.reporting.json_filename
            )
            text_path = (
                context.run_output_directory
                / self.config.reporting.text_filename
            )

            _write_json_atomic(json_path, report)
            _write_text_atomic(
                text_path,
                _render_text_report(report),
            )

            LOGGER.info(
                "JSON report: %s",
                json_path,
            )
            LOGGER.info(
                "Text report: %s",
                text_path,
            )
            LOGGER.info(
                "Phase 7 status: %s",
                report["status"],
            )

            success = bool(report["successful"])
        except Exception as exc:
            LOGGER.exception(
                "Unable to generate Phase 7 reports"
            )
            success = False
        finally:
            self._close_components()

        return 0 if success else 1

    def _initialize_components(
        self,
        context: RunContext,
    ) -> None:
        config = self.config

        self.stream_manager = StreamManager(
            base_directory=config.config_directory,
            ffprobe_path=config.runtime.ffprobe_path,
            probe_timeout_seconds=(
                config.runtime.ffprobe_timeout_seconds
            ),
            stop_event=self.shutdown_event,
        )

        self.camera_manager = CameraManager(
            stream_manager=self.stream_manager,
            run_output_directory=(
                context.run_output_directory
            ),
            redact_source_credentials=(
                config.reporting.redact_uri_credentials
            ),
        )
        self.camera_manager.register_all(
            config.sources
        )

        self.resource_monitor = ResourceMonitor(
            enabled=config.resource_monitor.enabled,
            sample_interval_seconds=(
                config.resource_monitor
                .sample_interval_seconds
            ),
            include_process_metrics=(
                config.resource_monitor
                .include_process_metrics
            ),
            gpu_enabled=(
                config.resource_monitor.gpu.enabled
            ),
            nvidia_smi_path=(
                config.resource_monitor.gpu
                .nvidia_smi_path
            ),
            gpu_query_timeout_seconds=(
                config.resource_monitor.gpu
                .query_timeout_seconds
            ),
            shutdown_event=self.shutdown_event,
        )

        self.health_monitor = HealthMonitor(
            camera_manager=self.camera_manager,
            enabled=config.health_monitor.enabled,
            check_interval_seconds=(
                config.health_monitor
                .check_interval_seconds
            ),
            default_stale_timeout_seconds=(
                config.health_monitor
                .default_stale_timeout_seconds
            ),
            maximum_consecutive_health_failures=(
                config.health_monitor
                .maximum_consecutive_health_failures
            ),
            stale_detection_source_types=(
                config.health_monitor
                .stale_detection_source_types
            ),
            reconnect_policy=config.reconnect_policy,
            shutdown_event=self.shutdown_event,
        )

        self.scheduler = ProcessingScheduler(
            camera_manager=self.camera_manager,
            resource_monitor=self.resource_monitor,
            health_monitor=self.health_monitor,
            scheduler_config=config.scheduler,
            processing_config=config.processing,
            project_root=config.project_root,
            run_output_directory=(
                context.run_output_directory
            ),
            processing_timeout_seconds=(
                config.runtime
                .processing_timeout_seconds
            ),
            shutdown_grace_seconds=(
                config.runtime
                .shutdown_grace_seconds
            ),
            shutdown_event=self.shutdown_event,
        )

    def _build_report(
        self,
        *,
        context: RunContext,
        scheduler_result: SchedulerRunResult | None,
        fatal_errors: list[str],
    ) -> dict[str, Any]:
        camera_snapshots = (
            self.camera_manager.snapshots()
            if self.camera_manager is not None
            else []
        )
        resource_summary = (
            self.resource_monitor.summary()
            if self.resource_monitor is not None
            else {
                "enabled": (
                    self.config.resource_monitor.enabled
                ),
                "started": False,
                "sample_count": 0,
            }
        )
        health_summary = (
            self.health_monitor.summary()
            if self.health_monitor is not None
            else {
                "enabled": (
                    self.config.health_monitor.enabled
                ),
                "started": False,
                "events": [],
            }
        )

        warnings = _collect_camera_warnings(
            camera_snapshots
        )
        errors = list(fatal_errors)
        errors.extend(
            _collect_camera_errors(camera_snapshots)
        )

        if (
            self.config.resource_monitor.gpu.enabled
            and resource_summary.get("sample_count", 0) > 0
            and not resource_summary.get(
                "gpu_ever_available",
                False,
            )
        ):
            warnings.append(
                "GPU metrics were requested but unavailable. "
                "CPU and memory monitoring continued normally."
            )

        monitoring_errors = resource_summary.get(
            "monitoring_errors",
            [],
        )

        if isinstance(monitoring_errors, list):
            warnings.extend(
                f"Resource monitor: {item}"
                for item in monitoring_errors
            )

        required_failures = (
            list(scheduler_result.required_failures)
            if scheduler_result is not None
            else [
                snapshot["camera"]["camera_id"]
                for snapshot in camera_snapshots
                if snapshot["camera"]["required"]
                and snapshot["status"]
                in {"failed", "cancelled"}
            ]
        )

        interrupted = (
            scheduler_result.interrupted
            if scheduler_result is not None
            else self.shutdown_event.is_set()
        )

        successful = (
            scheduler_result is not None
            and scheduler_result.successful
            and not fatal_errors
            and not required_failures
        )

        status = _derive_run_status(
            successful=successful,
            interrupted=interrupted,
            mode=self.config.processing.mode,
            warnings=warnings,
        )

        processing_summary = (
            _build_processing_summary(
                camera_snapshots,
                self.config.processing.mode,
            )
        )

        limitations = [
            (
                "The default probe-only mode does not run "
                "compression and does not prove compression "
                "success."
            ),
            (
                "RTSP reconnect behavior has not been "
                "validated against a real camera."
            ),
            (
                "Real camera streams have not been validated "
                "by this implementation."
            ),
            (
                "Concurrent multi-camera encoding is disabled; "
                "multi-camera production throughput has not "
                "been validated."
            ),
            (
                "Edge-device or dedicated hardware performance "
                "has not been validated."
            ),
            (
                "Object-detection accuracy on compressed video "
                "has not been evaluated."
            ),
        ]

        report: dict[str, Any] = {
            "schema_version": self.config.schema_version,
            "phase": "phase_7",
            "run_id": context.run_id,
            "status": status,
            "successful": successful,
            "mode": self.config.processing.mode.value,
            "smoke_test": context.smoke_test,
            "started_at_utc": context.started_at_utc,
            "completed_at_utc": utc_now(),
            "project": {
                "project_root": str(
                    self.config.project_root
                ),
                "config_file": str(
                    self.config.config_path
                ),
                "run_output_directory": str(
                    context.run_output_directory
                ),
            },
            "scheduler": (
                scheduler_result.to_dict()
                if scheduler_result is not None
                else None
            ),
            "processing_summary": processing_summary,
            "camera_status_counts": (
                self.camera_manager.status_counts()
                if self.camera_manager is not None
                else {}
            ),
            "cameras": camera_snapshots,
            "resource_monitor": resource_summary,
            "health_monitor": health_summary,
            "warnings": _deduplicate(warnings),
            "errors": _deduplicate(errors),
            "limitations": limitations,
        }

        if (
            self.config.reporting.include_resource_samples
            and self.resource_monitor is not None
        ):
            report["resource_samples"] = [
                sample.to_dict()
                for sample
                in self.resource_monitor.samples()
            ]

        return report

    def _write_camera_reports(
        self,
        report: Mapping[str, Any],
    ) -> list[str]:
        if not self.config.reporting.write_camera_reports:
            return []

        warnings: list[str] = []
        cameras = report.get("cameras")

        if not isinstance(cameras, list):
            return [
                "Unable to write camera reports because "
                "the camera list is invalid"
            ]

        for camera in cameras:
            if not isinstance(camera, Mapping):
                continue

            output_value = camera.get(
                "output_directory"
            )

            if not isinstance(output_value, str):
                continue

            output_directory = Path(output_value)

            # Disabled cameras do not allocate an output directory.
            if not output_directory.is_dir():
                continue

            camera_payload = {
                "schema_version": (
                    self.config.schema_version
                ),
                "phase": "phase_7_camera",
                "run_id": report.get("run_id"),
                "run_status": report.get("status"),
                "mode": report.get("mode"),
                "camera_result": camera,
            }

            try:
                _write_json_atomic(
                    output_directory
                    / "camera_report.json",
                    camera_payload,
                )
            except Exception as exc:
                warnings.append(
                    "Unable to write camera report for "
                    f"{camera.get('camera', {}).get('camera_id', 'unknown')}: "
                    f"{exc}"
                )

        return warnings

    def _request_shutdown(
        self,
        reason: str,
    ) -> None:
        self.shutdown_event.set()

        if self.scheduler is not None:
            self.scheduler.request_shutdown(reason)
        else:
            LOGGER.warning("%s", reason)

    def _stop_monitors(self) -> None:
        if self.health_monitor is not None:
            try:
                self.health_monitor.close(
                    join_timeout_seconds=(
                        self.config.runtime
                        .shutdown_grace_seconds
                    )
                )
            except Exception:
                LOGGER.exception(
                    "Unable to close health monitor"
                )

        if self.resource_monitor is not None:
            try:
                self.resource_monitor.close(
                    join_timeout_seconds=(
                        self.config.runtime
                        .shutdown_grace_seconds
                    )
                )
            except Exception:
                LOGGER.exception(
                    "Unable to close resource monitor"
                )

    def _close_components(self) -> None:
        if self.scheduler is not None:
            try:
                self.scheduler.close()
            except Exception:
                LOGGER.exception(
                    "Unable to close processing scheduler"
                )

        if self.camera_manager is not None:
            try:
                self.camera_manager.close()
            except Exception:
                LOGGER.exception(
                    "Unable to close camera manager"
                )
        elif self.stream_manager is not None:
            try:
                self.stream_manager.close()
            except Exception:
                LOGGER.exception(
                    "Unable to close stream manager"
                )

    def _install_signal_handlers(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return

        handled_signals = [signal.SIGINT]

        if hasattr(signal, "SIGTERM"):
            handled_signals.append(signal.SIGTERM)

        for item in handled_signals:
            try:
                previous = signal.getsignal(item)
                self._previous_signal_handlers[item] = previous
                signal.signal(item, self._signal_handler)
            except (OSError, ValueError):
                continue

    def _restore_signal_handlers(self) -> None:
        for item, previous in (
            self._previous_signal_handlers.items()
        ):
            try:
                signal.signal(item, previous)
            except (OSError, ValueError):
                pass

        self._previous_signal_handlers.clear()

    def _signal_handler(
        self,
        signal_number: int,
        frame: Any,
    ) -> None:
        del frame
        self._request_shutdown(
            f"Received signal {signal_number}"
        )


def load_config(path: Path) -> Phase7Config:
    """Load and fully validate Phase 7 YAML configuration."""

    if yaml is None:
        raise Phase7ConfigurationError(
            "PyYAML is required. Install it with: "
            "python -m pip install PyYAML"
        )

    config_path = path.expanduser().resolve()

    if not config_path.is_file():
        raise Phase7ConfigurationError(
            f"Configuration file not found: {config_path}"
        )

    try:
        with config_path.open(
            "r",
            encoding="utf-8",
        ) as stream:
            raw = yaml.safe_load(stream)
    except OSError as exc:
        raise Phase7ConfigurationError(
            f"Unable to read configuration: {exc}"
        ) from exc
    except yaml.YAMLError as exc:
        raise Phase7ConfigurationError(
            f"Invalid YAML configuration: {exc}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise Phase7ConfigurationError(
            "The configuration root must be a mapping"
        )

    schema_version = _required_string(
        raw,
        "schema_version",
        prefix="configuration",
    )
    phase = _required_string(
        raw,
        "phase",
        prefix="configuration",
    )

    if phase != "phase_7":
        raise Phase7ConfigurationError(
            "configuration.phase must be 'phase_7'"
        )

    config_directory = config_path.parent
    project_root = config_directory.parent.resolve()

    paths = _required_mapping(raw, "paths")
    output_root_value = _required_string(
        paths,
        "output_root",
        prefix="paths",
    )
    output_root = _resolve_safe_relative_path(
        config_directory,
        output_root_value,
        field_name="paths.output_root",
    )

    runtime_mapping = _required_mapping(
        raw,
        "runtime",
    )
    runtime = RuntimeConfig(
        ffprobe_path=_resolve_executable(
            config_directory,
            _required_string(
                runtime_mapping,
                "ffprobe_path",
                prefix="runtime",
            ),
        ),
        ffprobe_timeout_seconds=_positive_number(
            runtime_mapping,
            "ffprobe_timeout_seconds",
            prefix="runtime",
        ),
        processing_timeout_seconds=_positive_number(
            runtime_mapping,
            "processing_timeout_seconds",
            prefix="runtime",
        ),
        shutdown_grace_seconds=_positive_number(
            runtime_mapping,
            "shutdown_grace_seconds",
            prefix="runtime",
        ),
        log_level=_validate_log_level(
            _required_string(
                runtime_mapping,
                "log_level",
                prefix="runtime",
            )
        ),
    )

    processing = ProcessingConfig.from_mapping(
        _required_mapping(raw, "processing"),
        config_directory=config_directory,
    )
    scheduler = SchedulerConfig.from_mapping(
        _required_mapping(raw, "scheduler")
    )

    resource_mapping = _required_mapping(
        raw,
        "resource_monitor",
    )
    gpu_mapping = _required_mapping(
        resource_mapping,
        "gpu",
    )

    resource_monitor = ResourceMonitorConfig(
        enabled=_required_boolean(
            resource_mapping,
            "enabled",
            prefix="resource_monitor",
        ),
        sample_interval_seconds=_positive_number(
            resource_mapping,
            "sample_interval_seconds",
            prefix="resource_monitor",
        ),
        include_process_metrics=_required_boolean(
            resource_mapping,
            "include_process_metrics",
            prefix="resource_monitor",
        ),
        gpu=GPUMonitorConfig(
            enabled=_required_boolean(
                gpu_mapping,
                "enabled",
                prefix="resource_monitor.gpu",
            ),
            nvidia_smi_path=_resolve_executable(
                config_directory,
                _required_string(
                    gpu_mapping,
                    "nvidia_smi_path",
                    prefix="resource_monitor.gpu",
                ),
            ),
            query_timeout_seconds=_positive_number(
                gpu_mapping,
                "query_timeout_seconds",
                prefix="resource_monitor.gpu",
            ),
        ),
    )

    health_mapping = _required_mapping(
        raw,
        "health_monitor",
    )
    raw_stale_types = health_mapping.get(
        "stale_detection_source_types"
    )

    if not isinstance(raw_stale_types, list):
        raise Phase7ConfigurationError(
            "health_monitor.stale_detection_source_types "
            "must be a list"
        )

    health_monitor = HealthMonitorConfig(
        enabled=_required_boolean(
            health_mapping,
            "enabled",
            prefix="health_monitor",
        ),
        check_interval_seconds=_positive_number(
            health_mapping,
            "check_interval_seconds",
            prefix="health_monitor",
        ),
        default_stale_timeout_seconds=_positive_number(
            health_mapping,
            "default_stale_timeout_seconds",
            prefix="health_monitor",
        ),
        maximum_consecutive_health_failures=(
            _positive_integer(
                health_mapping,
                "maximum_consecutive_health_failures",
                prefix="health_monitor",
            )
        ),
        stale_detection_source_types=frozenset(
            SourceType.parse(item)
            for item in raw_stale_types
        ),
    )

    reconnect_policy = ReconnectPolicy.from_mapping(
        _required_mapping(raw, "reconnect")
    )

    reporting_mapping = _required_mapping(
        raw,
        "reporting",
    )
    reporting = ReportingConfig(
        json_filename=_validate_filename(
            _required_string(
                reporting_mapping,
                "json_filename",
                prefix="reporting",
            ),
            field_name="reporting.json_filename",
        ),
        text_filename=_validate_filename(
            _required_string(
                reporting_mapping,
                "text_filename",
                prefix="reporting",
            ),
            field_name="reporting.text_filename",
        ),
        write_camera_reports=_required_boolean(
            reporting_mapping,
            "write_camera_reports",
            prefix="reporting",
        ),
        redact_uri_credentials=_required_boolean(
            reporting_mapping,
            "redact_uri_credentials",
            prefix="reporting",
        ),
        include_resource_samples=_required_boolean(
            reporting_mapping,
            "include_resource_samples",
            prefix="reporting",
        ),
    )

    raw_sources = raw.get("sources")

    if not isinstance(raw_sources, list):
        raise Phase7ConfigurationError(
            "sources must be a list"
        )

    if not raw_sources:
        raise Phase7ConfigurationError(
            "At least one source must be configured"
        )

    sources = tuple(
        StreamDefinition.from_mapping(item)
        for item in raw_sources
    )

    camera_ids = [
        source.camera_id
        for source in sources
    ]

    if len(camera_ids) != len(set(camera_ids)):
        duplicates = sorted(
            {
                item
                for item in camera_ids
                if camera_ids.count(item) > 1
            }
        )
        raise Phase7ConfigurationError(
            "Duplicate camera IDs: "
            + ", ".join(duplicates)
        )

    if not any(source.enabled for source in sources):
        raise Phase7ConfigurationError(
            "At least one source must be enabled"
        )

    return Phase7Config(
        schema_version=schema_version,
        phase=phase,
        config_path=config_path,
        config_directory=config_directory,
        project_root=project_root,
        output_root=output_root,
        runtime=runtime,
        processing=processing,
        scheduler=scheduler,
        resource_monitor=resource_monitor,
        health_monitor=health_monitor,
        reconnect_policy=reconnect_policy,
        reporting=reporting,
        sources=sources,
    )


def _apply_runtime_overrides(
    config: Phase7Config,
    *,
    smoke_test: bool,
    selected_camera_ids: Sequence[str] | None,
    force_probe_only: bool,
    force_external_pipeline: bool = False,
) -> Phase7Config:
    sources = config.sources
    processing = config.processing
    scheduler = config.scheduler

    if force_probe_only and force_external_pipeline:
        raise Phase7ConfigurationError(
            "Probe-only and external-pipeline modes cannot "
            "both be forced"
        )

    if smoke_test and force_external_pipeline:
        raise Phase7ConfigurationError(
            "The smoke test is always probe-only and cannot "
            "be combined with --run-pipeline"
        )

    if smoke_test:
        smoke_sources = tuple(
            source
            for source in sources
            if "smoke-test" in source.tags
        )

        if not smoke_sources:
            smoke_sources = tuple(
                source
                for source in sources
                if source.camera_id == "camera_01"
            )

        if not smoke_sources:
            raise Phase7ConfigurationError(
                "Smoke test requires a source tagged "
                "'smoke-test' or camera_id 'camera_01'"
            )

        sources = (
            replace(
                smoke_sources[0],
                enabled=True,
                required=True,
            ),
        )
        force_probe_only = True
        scheduler = replace(
            scheduler,
            max_concurrent_cameras=1,
        )

    if selected_camera_ids:
        requested = tuple(
            dict.fromkeys(selected_camera_ids)
        )
        by_id = {
            source.camera_id: source
            for source in sources
        }
        unknown = [
            camera_id
            for camera_id in requested
            if camera_id not in by_id
        ]

        if unknown:
            raise Phase7ConfigurationError(
                "Unknown selected camera IDs: "
                + ", ".join(unknown)
            )

        sources = tuple(
            by_id[camera_id]
            for camera_id in requested
        )

        if not any(source.enabled for source in sources):
            raise Phase7ConfigurationError(
                "Selected cameras are all disabled"
            )

    if force_external_pipeline:
        enabled_sources = tuple(
            source
            for source in sources
            if source.enabled
        )
        non_local_sources = [
            source.camera_id
            for source in enabled_sources
            if (
                source.source_type
                is not SourceType.LOCAL_FILE
            )
        ]

        if non_local_sources:
            raise Phase7ConfigurationError(
                "The Phase 2-6 pipeline adapter currently "
                "supports only local_file sources. Unsupported "
                "cameras: "
                + ", ".join(non_local_sources)
            )

        adapter = processing.command_adapter
        validation = adapter.success_validation

        if not adapter.command:
            raise Phase7ConfigurationError(
                "--run-pipeline requires a configured "
                "processing.command_adapter.command"
            )

        if not adapter.working_directory.is_dir():
            raise Phase7ConfigurationError(
                "Pipeline adapter working directory does not "
                f"exist: {adapter.working_directory}"
            )

        if (
            not validation.required_artifacts
            and validation.report_json is None
        ):
            raise Phase7ConfigurationError(
                "--run-pipeline requires explicit success "
                "artifacts or a JSON success report"
            )

        processing = replace(
            processing,
            mode=ProcessingMode.EXTERNAL_PIPELINE,
        )
        scheduler = replace(
            scheduler,
            max_concurrent_cameras=1,
        )
    elif force_probe_only:
        processing = replace(
            processing,
            mode=ProcessingMode.PROBE_ONLY,
        )

    return replace(
        config,
        sources=sources,
        processing=processing,
        scheduler=scheduler,
    )


def _create_run_context(
    config: Phase7Config,
    *,
    smoke_test: bool,
) -> RunContext:
    try:
        config.output_root.mkdir(
            parents=True,
            exist_ok=True,
        )
    except OSError as exc:
        raise Phase7Error(
            f"Unable to create output root "
            f"{config.output_root}: {exc}"
        ) from exc

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S%fZ")

    for _ in range(10):
        run_id = (
            f"phase7_{timestamp}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        run_directory = (
            config.output_root / run_id
        )

        try:
            run_directory.mkdir(
                parents=False,
                exist_ok=False,
            )
        except FileExistsError:
            continue
        except OSError as exc:
            raise Phase7Error(
                f"Unable to create run directory "
                f"{run_directory}: {exc}"
            ) from exc

        return RunContext(
            run_id=run_id,
            run_output_directory=(
                run_directory.resolve()
            ),
            started_at_utc=utc_now(),
            smoke_test=smoke_test,
            selected_camera_ids=tuple(
                source.camera_id
                for source in config.sources
                if source.enabled
            ),
        )

    raise Phase7Error(
        "Unable to allocate a unique run directory"
    )


def _build_processing_summary(
    cameras: list[dict[str, Any]],
    mode: ProcessingMode,
) -> dict[str, Any]:
    pipeline_verified = 0
    compression_verified = 0
    probe_only_completed = 0
    processing_invoked = 0

    for camera in cameras:
        result = camera.get("processing_result")

        if not isinstance(result, Mapping):
            continue

        if result.get("processing_invoked") is True:
            processing_invoked += 1

        if (
            result.get("pipeline_success_verified")
            is True
        ):
            pipeline_verified += 1

        if (
            result.get("compression_success_verified")
            is True
        ):
            compression_verified += 1

        if (
            result.get("mode") == "probe_only"
            and camera.get("status") == "completed"
        ):
            probe_only_completed += 1

    return {
        "mode": mode.value,
        "external_processing_invoked_camera_count": (
            processing_invoked
        ),
        "pipeline_success_verified_camera_count": (
            pipeline_verified
        ),
        "compression_success_verified_camera_count": (
            compression_verified
        ),
        "probe_only_completed_camera_count": (
            probe_only_completed
        ),
        "compression_claim": (
            "Compression success is reported only when "
            "explicitly proven by the configured pipeline "
            "report field."
        ),
    }


def _derive_run_status(
    *,
    successful: bool,
    interrupted: bool,
    mode: ProcessingMode,
    warnings: list[str],
) -> str:
    if interrupted:
        return "cancelled"

    if not successful:
        return "failed"

    if mode is ProcessingMode.PROBE_ONLY:
        return "probe_only_completed"

    if warnings:
        return "completed_with_warnings"

    return "success"


def _collect_camera_warnings(
    cameras: list[dict[str, Any]],
) -> list[str]:
    messages: list[str] = []

    for camera in cameras:
        camera_data = camera.get("camera", {})
        camera_id = camera_data.get(
            "camera_id",
            "unknown",
        )

        for warning in camera.get("warnings", []):
            messages.append(
                f"{camera_id}: {warning}"
            )

    return messages


def _collect_camera_errors(
    cameras: list[dict[str, Any]],
) -> list[str]:
    messages: list[str] = []

    for camera in cameras:
        error = camera.get("error")

        if not error:
            continue

        camera_data = camera.get("camera", {})
        camera_id = camera_data.get(
            "camera_id",
            "unknown",
        )
        messages.append(f"{camera_id}: {error}")

    return messages


def _render_text_report(
    report: Mapping[str, Any],
) -> str:
    lines = [
        "CCTV COMPRESSION PROJECT - PHASE 7 RUN REPORT",
        "=" * 52,
        "",
        f"Run ID: {report.get('run_id')}",
        f"Status: {report.get('status')}",
        f"Successful: {report.get('successful')}",
        f"Mode: {report.get('mode')}",
        f"Smoke test: {report.get('smoke_test')}",
        f"Started: {report.get('started_at_utc')}",
        f"Completed: {report.get('completed_at_utc')}",
        "",
    ]

    mode = report.get("mode")

    if mode == "probe_only":
        lines.extend(
            [
                "Processing statement:",
                "  Source inspection and orchestration were run.",
                "  Compression was not run or claimed.",
                "",
            ]
        )

    status_counts = report.get(
        "camera_status_counts",
        {},
    )
    lines.append("Camera status counts:")

    if isinstance(status_counts, Mapping):
        for key, value in status_counts.items():
            lines.append(f"  {key}: {value}")

    lines.extend(["", "Camera results:"])

    cameras = report.get("cameras", [])

    if isinstance(cameras, list):
        for camera in cameras:
            if not isinstance(camera, Mapping):
                continue

            definition = camera.get("camera", {})
            camera_id = definition.get(
                "camera_id",
                "unknown",
            )
            source = definition.get("source")
            state = camera.get("status")
            lines.append(
                f"  [{camera_id}] status={state}"
            )
            lines.append(f"    source={source}")

            inspection = camera.get("inspection")

            if isinstance(inspection, Mapping):
                probe = inspection.get("probe")

                if isinstance(probe, Mapping):
                    lines.append(
                        "    media="
                        f"{probe.get('resolution')}, "
                        f"codec={probe.get('codec_name')}, "
                        f"fps={probe.get('average_fps')}, "
                        f"duration={probe.get('duration_seconds')}, "
                        f"frames={probe.get('frame_count')}"
                    )
                    lines.append(
                        "    frame_count_source="
                        f"{probe.get('frame_count_source')}"
                    )
                else:
                    lines.append(
                        "    inspection="
                        f"{inspection.get('message')}"
                    )

            result = camera.get("processing_result")

            if isinstance(result, Mapping):
                lines.append(
                    "    pipeline_success_verified="
                    f"{result.get('pipeline_success_verified', False)}"
                )
                lines.append(
                    "    compression_success_verified="
                    f"{result.get('compression_success_verified', False)}"
                )

            if camera.get("error"):
                lines.append(
                    f"    error={camera.get('error')}"
                )

    processing = report.get(
        "processing_summary",
        {},
    )
    lines.extend(["", "Processing summary:"])

    if isinstance(processing, Mapping):
        for key in (
            "external_processing_invoked_camera_count",
            "pipeline_success_verified_camera_count",
            "compression_success_verified_camera_count",
            "probe_only_completed_camera_count",
        ):
            lines.append(
                f"  {key}: {processing.get(key)}"
            )

    resource = report.get(
        "resource_monitor",
        {},
    )
    lines.extend(["", "Resource monitoring:"])

    if isinstance(resource, Mapping):
        lines.append(
            f"  enabled: {resource.get('enabled')}"
        )
        lines.append(
            f"  samples: {resource.get('sample_count')}"
        )
        lines.append(
            "  GPU ever available: "
            f"{resource.get('gpu_ever_available')}"
        )

        cpu = resource.get("system_cpu_percent")
        memory = resource.get(
            "system_memory_percent"
        )

        if cpu is not None:
            lines.append(
                f"  system CPU percent: {cpu}"
            )

        if memory is not None:
            lines.append(
                f"  system memory percent: {memory}"
            )

    for heading, key in (
        ("Warnings", "warnings"),
        ("Errors", "errors"),
        ("Limitations", "limitations"),
    ):
        lines.extend(["", f"{heading}:"])
        values = report.get(key, [])

        if isinstance(values, list) and values:
            lines.extend(
                f"  - {value}"
                for value in values
            )
        else:
            lines.append("  None")

    lines.append("")
    return "\n".join(lines)


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
    except Exception:
        try:
            if temporary.exists():
                temporary.unlink()
        except OSError:
            pass
        raise


def _write_text_atomic(
    path: Path,
    content: str,
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
            stream.write(content)
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


def _configure_logging(
    level: str,
    log_path: Path,
) -> None:
    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s [%(levelname)s] "
            "%(name)s: %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_path,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logging.basicConfig(
        level=getattr(logging, level),
        handlers=[console, file_handler],
        force=True,
    )


def _required_mapping(
    mapping: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = mapping.get(key)

    if not isinstance(value, Mapping):
        raise Phase7ConfigurationError(
            f"{key} must be a mapping"
        )

    return value


def _required_string(
    mapping: Mapping[str, Any],
    key: str,
    *,
    prefix: str,
) -> str:
    value = mapping.get(key)

    if not isinstance(value, str) or not value.strip():
        raise Phase7ConfigurationError(
            f"{prefix}.{key} must be a non-empty string"
        )

    return value.strip()


def _required_boolean(
    mapping: Mapping[str, Any],
    key: str,
    *,
    prefix: str,
) -> bool:
    value = mapping.get(key)

    if not isinstance(value, bool):
        raise Phase7ConfigurationError(
            f"{prefix}.{key} must be boolean"
        )

    return value


def _positive_integer(
    mapping: Mapping[str, Any],
    key: str,
    *,
    prefix: str,
) -> int:
    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise Phase7ConfigurationError(
            f"{prefix}.{key} must be an integer"
        )

    if value <= 0:
        raise Phase7ConfigurationError(
            f"{prefix}.{key} must be positive"
        )

    return value


def _positive_number(
    mapping: Mapping[str, Any],
    key: str,
    *,
    prefix: str,
) -> float:
    value = mapping.get(key)

    if isinstance(value, bool):
        raise Phase7ConfigurationError(
            f"{prefix}.{key} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError):
        raise Phase7ConfigurationError(
            f"{prefix}.{key} must be numeric"
        ) from None

    if not math.isfinite(result) or result <= 0:
        raise Phase7ConfigurationError(
            f"{prefix}.{key} must be positive and finite"
        )

    return result


def _resolve_executable(
    base_directory: Path,
    value: str,
) -> str:
    path = Path(value).expanduser()

    if path.is_absolute():
        return str(path.resolve())

    if path.parent != Path("."):
        return str(
            (base_directory / path).resolve()
        )

    return value


def _resolve_safe_relative_path(
    base_directory: Path,
    value: str,
    *,
    field_name: str,
) -> Path:
    raw_path = Path(value).expanduser()

    if raw_path.is_absolute():
        raise Phase7ConfigurationError(
            f"{field_name} must be project-relative"
        )

    base = base_directory.resolve()
    resolved = (base / raw_path).resolve()

    try:
        common = Path(
            os.path.commonpath(
                [str(base), str(resolved)]
            )
        )
    except ValueError as exc:
        raise Phase7ConfigurationError(
            f"{field_name} resolved to an unexpected volume"
        ) from exc

    if common != base:
        raise Phase7ConfigurationError(
            f"{field_name} cannot escape the phase_7 directory"
        )

    return resolved


def _validate_filename(
    value: str,
    *,
    field_name: str,
) -> str:
    path = Path(value)

    if (
        path.is_absolute()
        or path.name != value
        or value in {".", ".."}
    ):
        raise Phase7ConfigurationError(
            f"{field_name} must be a plain filename"
        )

    return value


def _validate_log_level(value: str) -> str:
    normalized = value.upper()

    if normalized not in {
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    }:
        raise Phase7ConfigurationError(
            "runtime.log_level must be DEBUG, INFO, "
            "WARNING, ERROR, or CRITICAL"
        )

    return normalized


def _deduplicate(
    values: Sequence[str],
) -> list[str]:
    return list(dict.fromkeys(values))


def _parse_arguments(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run Phase 7 multi-source orchestration."
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
            "Path to Phase 7 config.yaml "
            "(default: beside this script)"
        ),
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Run only the configured smoke-test camera and "
            "force safe probe-only mode."
        ),
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help=(
            "Force probe-only mode without changing config.yaml."
        ),
    )
    parser.add_argument(
        "--run-pipeline",
        action="store_true",
        help=(
            "Explicitly run the configured local-file Phase "
            "2-6 adapter. Processing is limited to one camera "
            "at a time."
        ),
    )
    parser.add_argument(
        "--camera-id",
        action="append",
        default=None,
        help=(
            "Run only this camera ID. Repeat for multiple "
            "cameras."
        ),
    )
    parser.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate config.yaml without starting a run.",
    )

    arguments = parser.parse_args(argv)

    if arguments.run_pipeline and (
        arguments.probe_only
        or arguments.smoke_test
    ):
        parser.error(
            "--run-pipeline cannot be combined with "
            "--probe-only or --smoke-test"
        )

    return arguments


def main(
    argv: Sequence[str] | None = None,
) -> int:
    arguments = _parse_arguments(argv)

    try:
        config = load_config(arguments.config)

        if arguments.validate_config:
            validated_config = _apply_runtime_overrides(
                config,
                smoke_test=arguments.smoke_test,
                selected_camera_ids=arguments.camera_id,
                force_probe_only=arguments.probe_only,
                force_external_pipeline=(
                    arguments.run_pipeline
                ),
            )
            print(
                "Phase 7 configuration is valid: "
                f"{config.config_path}"
            )
            print(
                "Validated processing mode: "
                f"{validated_config.processing.mode.value}"
            )
            print(
                "Maximum concurrent cameras: "
                f"{validated_config.scheduler.max_concurrent_cameras}"
            )
            return 0

        application = Phase7Application(
            config,
            smoke_test=arguments.smoke_test,
            selected_camera_ids=arguments.camera_id,
            force_probe_only=arguments.probe_only,
            force_external_pipeline=(
                arguments.run_pipeline
            ),
        )
        return application.run()
    except (
        Phase7ConfigurationError,
        ProcessingConfigurationError,
        StreamConfigurationError,
        ResourceMonitorDependencyError,
        Phase7Error,
    ) as exc:
        print(
            f"Phase 7 failed: {exc}",
            file=sys.stderr,
        )
        return 1
    except Exception as exc:
        print(
            "Phase 7 failed unexpectedly: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
