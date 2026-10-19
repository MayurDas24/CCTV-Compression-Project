"""
CPU, memory, process, and optional GPU monitoring for Phase 7.

CPU and memory metrics use psutil because it provides consistent behavior on
Windows and Linux. GPU metrics are optional and collected only when the
configured nvidia-smi executable is available.

A missing GPU tool or unsupported GPU is never a fatal Phase 7 error.
"""

from __future__ import annotations

import csv
import logging
import math
import os
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore[assignment]

try:
    from .stream_manager import utc_now
except ImportError:
    from stream_manager import utc_now


LOGGER = logging.getLogger(__name__)


class ResourceMonitorError(RuntimeError):
    """Base resource-monitoring error."""


class ResourceMonitorDependencyError(ResourceMonitorError):
    """Raised when an enabled required dependency is unavailable."""


@dataclass(frozen=True, slots=True)
class GPUSample:
    """Metrics for one NVIDIA GPU."""

    index: int
    name: str
    utilization_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    memory_percent: float | None
    temperature_celsius: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "utilization_percent": self.utilization_percent,
            "memory_used_bytes": self.memory_used_bytes,
            "memory_total_bytes": self.memory_total_bytes,
            "memory_percent": self.memory_percent,
            "temperature_celsius": self.temperature_celsius,
        }


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """One immutable resource sample."""

    timestamp_utc: str
    system_cpu_percent: float
    system_memory_percent: float
    system_memory_used_bytes: int
    system_memory_available_bytes: int
    process_cpu_percent: float | None
    process_memory_rss_bytes: int | None
    process_memory_percent: float | None
    gpu_metrics_available: bool
    gpu_metrics_error: str | None
    gpus: tuple[GPUSample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "system": {
                "cpu_percent": self.system_cpu_percent,
                "memory_percent": self.system_memory_percent,
                "memory_used_bytes": self.system_memory_used_bytes,
                "memory_available_bytes": (
                    self.system_memory_available_bytes
                ),
            },
            "phase7_process": {
                "cpu_percent": self.process_cpu_percent,
                "memory_rss_bytes": self.process_memory_rss_bytes,
                "memory_percent": self.process_memory_percent,
            },
            "gpu": {
                "available": self.gpu_metrics_available,
                "error": self.gpu_metrics_error,
                "devices": [
                    gpu.to_dict()
                    for gpu in self.gpus
                ],
            },
        }


@dataclass(frozen=True, slots=True)
class AdmissionDecision:
    """Result of waiting for sufficient CPU and memory capacity."""

    admitted: bool
    reason: str
    waited_seconds: float
    sample: ResourceSample | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "admitted": self.admitted,
            "reason": self.reason,
            "waited_seconds": self.waited_seconds,
            "sample": (
                self.sample.to_dict()
                if self.sample is not None
                else None
            ),
        }


class ResourceMonitor:
    """
    Periodically collect resource metrics and provide admission control.

    The supplied ``shutdown_event`` is observed but never set by this class.
    ResourceMonitor uses a private event to stop its own worker thread.
    """

    def __init__(
        self,
        *,
        enabled: bool,
        sample_interval_seconds: float,
        include_process_metrics: bool,
        gpu_enabled: bool,
        nvidia_smi_path: str,
        gpu_query_timeout_seconds: float,
        shutdown_event: threading.Event | None = None,
        maximum_samples: int = 10000,
    ) -> None:
        if sample_interval_seconds <= 0:
            raise ResourceMonitorError(
                "sample_interval_seconds must be positive"
            )

        if gpu_query_timeout_seconds <= 0:
            raise ResourceMonitorError(
                "gpu_query_timeout_seconds must be positive"
            )

        if maximum_samples <= 0:
            raise ResourceMonitorError(
                "maximum_samples must be positive"
            )

        self.enabled = bool(enabled)
        self.sample_interval_seconds = float(
            sample_interval_seconds
        )
        self.include_process_metrics = bool(
            include_process_metrics
        )
        self.gpu_enabled = bool(gpu_enabled)
        self.nvidia_smi_path = nvidia_smi_path
        self.gpu_query_timeout_seconds = float(
            gpu_query_timeout_seconds
        )
        self.shutdown_event = (
            shutdown_event or threading.Event()
        )

        self._samples: deque[ResourceSample] = deque(
            maxlen=maximum_samples
        )
        self._errors: deque[str] = deque(maxlen=100)
        self._history_lock = threading.RLock()
        self._sample_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()

        self._local_stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._process: Any = None

        self._gpu_permanently_unavailable_reason: (
            str | None
        ) = None

    def start(self) -> None:
        """Start background monitoring."""

        with self._lifecycle_lock:
            if self._closed:
                raise ResourceMonitorError(
                    "Cannot start a closed ResourceMonitor"
                )

            if self._started:
                return

            if not self.enabled:
                self._started = True
                LOGGER.info(
                    "Resource monitoring is disabled"
                )
                return

            if psutil is None:
                raise ResourceMonitorDependencyError(
                    "Resource monitoring requires psutil. "
                    "Install it with: python -m pip install psutil"
                )

            self._process = psutil.Process(os.getpid())

            # Prime non-blocking CPU counters. The first values may otherwise
            # represent an undefined interval.
            psutil.cpu_percent(interval=None)

            if self.include_process_metrics:
                self._process.cpu_percent(interval=None)

            self._started = True
            self._thread = threading.Thread(
                target=self._monitor_loop,
                name="phase7-resource-monitor",
                daemon=True,
            )
            self._thread.start()

    def sample_now(self) -> ResourceSample | None:
        """
        Collect and store a sample immediately.

        Returns None when monitoring is disabled.
        """

        if not self.enabled:
            return None

        if psutil is None:
            raise ResourceMonitorDependencyError(
                "Resource monitoring requires psutil"
            )

        if not self._started:
            raise ResourceMonitorError(
                "ResourceMonitor must be started before sampling"
            )

        if self._closed:
            raise ResourceMonitorError(
                "Cannot sample a closed ResourceMonitor"
            )

        with self._sample_lock:
            sample = self._collect_sample()

            with self._history_lock:
                self._samples.append(sample)

            return sample

    def latest_sample(self) -> ResourceSample | None:
        with self._history_lock:
            if not self._samples:
                return None

            return self._samples[-1]

    def samples(self) -> tuple[ResourceSample, ...]:
        with self._history_lock:
            return tuple(self._samples)

    def errors(self) -> tuple[str, ...]:
        with self._history_lock:
            return tuple(self._errors)

    def wait_for_capacity(
        self,
        *,
        cpu_threshold_percent: float,
        memory_threshold_percent: float,
        timeout_seconds: float,
        poll_interval_seconds: float,
        cancellation_event: threading.Event | None = None,
    ) -> AdmissionDecision:
        """
        Wait until CPU and memory are within configured thresholds.

        Resource thresholds are admission controls for new camera jobs. They do
        not terminate jobs that are already running.
        """

        _validate_percent(
            "cpu_threshold_percent",
            cpu_threshold_percent,
        )
        _validate_percent(
            "memory_threshold_percent",
            memory_threshold_percent,
        )

        if timeout_seconds < 0:
            raise ResourceMonitorError(
                "timeout_seconds cannot be negative"
            )

        if poll_interval_seconds <= 0:
            raise ResourceMonitorError(
                "poll_interval_seconds must be positive"
            )

        if not self.enabled:
            return AdmissionDecision(
                admitted=True,
                reason="Resource monitoring is disabled",
                waited_seconds=0.0,
                sample=None,
            )

        if not self._started:
            raise ResourceMonitorError(
                "ResourceMonitor must be started before "
                "resource admission checks"
            )

        start = time.monotonic()
        deadline = start + timeout_seconds
        last_sample: ResourceSample | None = None
        last_reason = "No resource sample was available"

        while True:
            if self._is_cancelled(cancellation_event):
                return AdmissionDecision(
                    admitted=False,
                    reason=(
                        "Resource admission was cancelled "
                        "during shutdown"
                    ),
                    waited_seconds=round(
                        time.monotonic() - start,
                        6,
                    ),
                    sample=last_sample,
                )

            try:
                last_sample = self.sample_now()
            except ResourceMonitorError as exc:
                last_reason = (
                    f"Unable to collect resource metrics: {exc}"
                )
                self._record_error(last_reason)
            else:
                if last_sample is not None:
                    cpu_ok = (
                        last_sample.system_cpu_percent
                        <= cpu_threshold_percent
                    )
                    memory_ok = (
                        last_sample.system_memory_percent
                        <= memory_threshold_percent
                    )

                    if cpu_ok and memory_ok:
                        return AdmissionDecision(
                            admitted=True,
                            reason=(
                                "CPU and memory are within "
                                "admission thresholds"
                            ),
                            waited_seconds=round(
                                time.monotonic() - start,
                                6,
                            ),
                            sample=last_sample,
                        )

                    exceeded: list[str] = []

                    if not cpu_ok:
                        exceeded.append(
                            "CPU "
                            f"{last_sample.system_cpu_percent:.2f}% "
                            f"> {cpu_threshold_percent:.2f}%"
                        )

                    if not memory_ok:
                        exceeded.append(
                            "memory "
                            f"{last_sample.system_memory_percent:.2f}% "
                            f"> {memory_threshold_percent:.2f}%"
                        )

                    last_reason = (
                        "Resource thresholds exceeded: "
                        + ", ".join(exceeded)
                    )

            now = time.monotonic()

            if now >= deadline:
                return AdmissionDecision(
                    admitted=False,
                    reason=(
                        f"Resource admission timed out: "
                        f"{last_reason}"
                    ),
                    waited_seconds=round(now - start, 6),
                    sample=last_sample,
                )

            remaining = deadline - now
            wait_seconds = min(
                poll_interval_seconds,
                remaining,
            )

            if self._wait_for_cancellation(
                wait_seconds,
                cancellation_event,
            ):
                return AdmissionDecision(
                    admitted=False,
                    reason=(
                        "Resource admission was cancelled "
                        "during shutdown"
                    ),
                    waited_seconds=round(
                        time.monotonic() - start,
                        6,
                    ),
                    sample=last_sample,
                )

    def summary(self) -> dict[str, Any]:
        """Return bounded aggregate metrics for the final report."""

        samples = self.samples()
        errors = self.errors()

        result: dict[str, Any] = {
            "enabled": self.enabled,
            "started": self._started,
            "sample_count": len(samples),
            "monitoring_errors": list(errors),
            "gpu_requested": self.gpu_enabled,
            "gpu_ever_available": any(
                sample.gpu_metrics_available
                for sample in samples
            ),
        }

        if not samples:
            result.update(
                {
                    "first_sample_utc": None,
                    "last_sample_utc": None,
                    "system_cpu_percent": None,
                    "system_memory_percent": None,
                    "phase7_process_peak_rss_bytes": None,
                    "gpu": [],
                }
            )
            return result

        cpu_values = [
            sample.system_cpu_percent
            for sample in samples
        ]
        memory_values = [
            sample.system_memory_percent
            for sample in samples
        ]
        rss_values = [
            sample.process_memory_rss_bytes
            for sample in samples
            if sample.process_memory_rss_bytes is not None
        ]

        result.update(
            {
                "first_sample_utc": samples[0].timestamp_utc,
                "last_sample_utc": samples[-1].timestamp_utc,
                "system_cpu_percent": _aggregate(
                    cpu_values
                ),
                "system_memory_percent": _aggregate(
                    memory_values
                ),
                "phase7_process_peak_rss_bytes": (
                    max(rss_values)
                    if rss_values
                    else None
                ),
                "gpu": self._gpu_summary(samples),
            }
        )
        return result

    def close(
        self,
        *,
        join_timeout_seconds: float = 10.0,
    ) -> None:
        """Stop monitoring and release the worker thread."""

        if join_timeout_seconds < 0:
            raise ResourceMonitorError(
                "join_timeout_seconds cannot be negative"
            )

        with self._lifecycle_lock:
            if self._closed:
                return

            self._closed = True
            self._local_stop_event.set()
            thread = self._thread

        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=join_timeout_seconds)

            if thread.is_alive():
                LOGGER.warning(
                    "Resource monitor did not stop within %.3f seconds",
                    join_timeout_seconds,
                )

    def __enter__(self) -> "ResourceMonitor":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()

    def _monitor_loop(self) -> None:
        while not self._should_stop():
            try:
                self.sample_now()
            except Exception as exc:
                message = (
                    "Resource sample failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._record_error(message)
                LOGGER.warning("%s", message)

            if self._local_stop_event.wait(
                self.sample_interval_seconds
            ):
                break

    def _collect_sample(self) -> ResourceSample:
        if psutil is None:
            raise ResourceMonitorDependencyError(
                "psutil is unavailable"
            )

        try:
            system_cpu = float(
                psutil.cpu_percent(interval=None)
            )
            virtual_memory = psutil.virtual_memory()
        except Exception as exc:
            raise ResourceMonitorError(
                f"Unable to read system metrics: {exc}"
            ) from exc

        process_cpu: float | None = None
        process_rss: int | None = None
        process_memory_percent: float | None = None

        if self.include_process_metrics:
            try:
                process_cpu = float(
                    self._process.cpu_percent(interval=None)
                )
                process_info = self._process.memory_info()
                process_rss = int(process_info.rss)
                process_memory_percent = float(
                    self._process.memory_percent()
                )
            except Exception as exc:
                self._record_error(
                    f"Unable to read Phase 7 process metrics: "
                    f"{exc}"
                )

        gpus: tuple[GPUSample, ...] = ()
        gpu_available = False
        gpu_error: str | None = None

        if self.gpu_enabled:
            gpus, gpu_error = self._collect_gpu_metrics()
            gpu_available = bool(gpus)

        return ResourceSample(
            timestamp_utc=utc_now(),
            system_cpu_percent=_finite_percent(
                system_cpu
            ),
            system_memory_percent=_finite_percent(
                float(virtual_memory.percent)
            ),
            system_memory_used_bytes=int(
                virtual_memory.used
            ),
            system_memory_available_bytes=int(
                virtual_memory.available
            ),
            process_cpu_percent=(
                _finite_nonnegative(process_cpu)
                if process_cpu is not None
                else None
            ),
            process_memory_rss_bytes=process_rss,
            process_memory_percent=(
                _finite_nonnegative(
                    process_memory_percent
                )
                if process_memory_percent is not None
                else None
            ),
            gpu_metrics_available=gpu_available,
            gpu_metrics_error=gpu_error,
            gpus=gpus,
        )

    def _collect_gpu_metrics(
        self,
    ) -> tuple[tuple[GPUSample, ...], str | None]:
        if self._gpu_permanently_unavailable_reason:
            return (
                (),
                self._gpu_permanently_unavailable_reason,
            )

        command = [
            self.nvidia_smi_path,
            "--query-gpu="
            "index,name,utilization.gpu,"
            "memory.used,memory.total,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]

        creation_flags = (
            subprocess.CREATE_NO_WINDOW
            if hasattr(subprocess, "CREATE_NO_WINDOW")
            else 0
        )

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
                creationflags=creation_flags,
            )
        except FileNotFoundError:
            reason = (
                "nvidia-smi is unavailable; GPU metrics "
                "were not collected"
            )
            self._gpu_permanently_unavailable_reason = reason
            return (), reason
        except OSError as exc:
            return (
                (),
                f"Unable to start nvidia-smi: {exc}",
            )

        deadline = (
            time.monotonic()
            + self.gpu_query_timeout_seconds
        )

        while True:
            if self._should_stop():
                _terminate_process(process)
                return (
                    (),
                    "GPU query cancelled during shutdown",
                )

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                _terminate_process(process)
                return (
                    (),
                    "nvidia-smi query timed out after "
                    f"{self.gpu_query_timeout_seconds:.3f} "
                    "seconds",
                )

            try:
                stdout, stderr = process.communicate(
                    timeout=min(0.25, remaining)
                )
                break
            except subprocess.TimeoutExpired:
                continue

        if process.returncode != 0:
            diagnostic = (
                stderr.strip()
                or "no diagnostic was provided"
            )

            if len(diagnostic) > 1000:
                diagnostic = diagnostic[-1000:]

            return (
                (),
                "nvidia-smi failed with exit code "
                f"{process.returncode}: {diagnostic}",
            )

        try:
            devices = _parse_nvidia_smi_csv(stdout)
        except ValueError as exc:
            return (
                (),
                f"Unable to parse nvidia-smi output: {exc}",
            )

        if not devices:
            return (
                (),
                "nvidia-smi reported no GPU devices",
            )

        return tuple(devices), None

    def _gpu_summary(
        self,
        samples: tuple[ResourceSample, ...],
    ) -> list[dict[str, Any]]:
        by_index: dict[int, list[GPUSample]] = {}

        for sample in samples:
            for gpu in sample.gpus:
                by_index.setdefault(gpu.index, []).append(gpu)

        summaries: list[dict[str, Any]] = []

        for index in sorted(by_index):
            values = by_index[index]
            utilization = [
                value.utilization_percent
                for value in values
                if value.utilization_percent is not None
            ]
            memory_percent = [
                value.memory_percent
                for value in values
                if value.memory_percent is not None
            ]
            temperatures = [
                value.temperature_celsius
                for value in values
                if value.temperature_celsius is not None
            ]

            summaries.append(
                {
                    "index": index,
                    "name": values[-1].name,
                    "sample_count": len(values),
                    "utilization_percent": (
                        _aggregate(utilization)
                        if utilization
                        else None
                    ),
                    "memory_percent": (
                        _aggregate(memory_percent)
                        if memory_percent
                        else None
                    ),
                    "temperature_celsius": (
                        _aggregate(temperatures)
                        if temperatures
                        else None
                    ),
                }
            )

        return summaries

    def _record_error(self, message: str) -> None:
        with self._history_lock:
            self._errors.append(message)

    def _should_stop(self) -> bool:
        return (
            self._local_stop_event.is_set()
            or self.shutdown_event.is_set()
        )

    def _is_cancelled(
        self,
        cancellation_event: threading.Event | None,
    ) -> bool:
        return (
            self._should_stop()
            or (
                cancellation_event is not None
                and cancellation_event.is_set()
            )
        )

    def _wait_for_cancellation(
        self,
        seconds: float,
        cancellation_event: threading.Event | None,
    ) -> bool:
        deadline = time.monotonic() + seconds

        while True:
            if self._is_cancelled(cancellation_event):
                return True

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                return False

            self._local_stop_event.wait(
                min(0.1, remaining)
            )


def _parse_nvidia_smi_csv(
    output: str,
) -> list[GPUSample]:
    devices: list[GPUSample] = []
    reader = csv.reader(output.splitlines())

    for line_number, row in enumerate(reader, start=1):
        if not row or not any(item.strip() for item in row):
            continue

        if len(row) != 6:
            raise ValueError(
                f"line {line_number} contained "
                f"{len(row)} columns instead of 6"
            )

        index = _parse_required_integer(
            row[0],
            field_name="GPU index",
        )
        name = row[1].strip()

        if not name:
            name = f"GPU {index}"

        utilization = _parse_optional_number(row[2])
        memory_used_mib = _parse_optional_number(row[3])
        memory_total_mib = _parse_optional_number(row[4])
        temperature = _parse_optional_number(row[5])

        memory_used_bytes = (
            int(memory_used_mib * 1024 * 1024)
            if memory_used_mib is not None
            else None
        )
        memory_total_bytes = (
            int(memory_total_mib * 1024 * 1024)
            if memory_total_mib is not None
            else None
        )

        memory_percent: float | None = None

        if (
            memory_used_bytes is not None
            and memory_total_bytes is not None
            and memory_total_bytes > 0
        ):
            memory_percent = (
                memory_used_bytes
                / memory_total_bytes
                * 100.0
            )

        devices.append(
            GPUSample(
                index=index,
                name=name,
                utilization_percent=(
                    _finite_nonnegative(utilization)
                    if utilization is not None
                    else None
                ),
                memory_used_bytes=memory_used_bytes,
                memory_total_bytes=memory_total_bytes,
                memory_percent=(
                    _finite_nonnegative(memory_percent)
                    if memory_percent is not None
                    else None
                ),
                temperature_celsius=(
                    _finite_nonnegative(temperature)
                    if temperature is not None
                    else None
                ),
            )
        )

    return devices


def _parse_required_integer(
    value: str,
    *,
    field_name: str,
) -> int:
    try:
        result = int(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"{field_name} is not an integer: {value!r}"
        ) from exc

    if result < 0:
        raise ValueError(
            f"{field_name} cannot be negative"
        )

    return result


def _parse_optional_number(
    value: str,
) -> float | None:
    normalized = value.strip()

    if not normalized:
        return None

    if normalized.upper() in {
        "N/A",
        "[N/A]",
        "NOT SUPPORTED",
        "[NOT SUPPORTED]",
    }:
        return None

    try:
        result = float(normalized)
    except ValueError:
        return None

    if not math.isfinite(result):
        return None

    return result


def _finite_percent(value: float) -> float:
    if not math.isfinite(value):
        raise ResourceMonitorError(
            "Received a non-finite percentage"
        )

    return min(100.0, max(0.0, value))


def _finite_nonnegative(
    value: float,
) -> float:
    if not math.isfinite(value):
        raise ResourceMonitorError(
            "Received a non-finite resource value"
        )

    return max(0.0, value)


def _validate_percent(
    name: str,
    value: float,
) -> None:
    if not math.isfinite(value):
        raise ResourceMonitorError(
            f"{name} must be finite"
        )

    if value < 0 or value > 100:
        raise ResourceMonitorError(
            f"{name} must be between 0 and 100"
        )


def _aggregate(
    values: list[float],
) -> dict[str, float]:
    if not values:
        raise ValueError(
            "Cannot aggregate an empty value list"
        )

    return {
        "minimum": round(min(values), 6),
        "maximum": round(max(values), 6),
        "average": round(
            sum(values) / len(values),
            6,
        ),
    }


def _terminate_process(
    process: subprocess.Popen[str],
) -> None:
    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=2.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


__all__ = [
    "AdmissionDecision",
    "GPUSample",
    "ResourceMonitor",
    "ResourceMonitorDependencyError",
    "ResourceMonitorError",
    "ResourceSample",
]