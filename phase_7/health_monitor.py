"""
Connection health, progress heartbeat, stale-source, and reconnect monitoring.

The monitor is source-agnostic. It tracks meaningful progress heartbeats and
can use scheduler-provided callbacks for connection checks and reconnects.

Local finite files are excluded from stale-stream detection by the default
configuration. RTSP behavior is implemented as an abstraction but must not be
described as production-validated until tested with real cameras and networks.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping

try:
    from .camera_manager import (
        CameraManager,
        CameraManagerError,
        CameraStatus,
        TERMINAL_CAMERA_STATUSES,
    )
    from .stream_manager import SourceType, utc_now
except ImportError:
    from camera_manager import (
        CameraManager,
        CameraManagerError,
        CameraStatus,
        TERMINAL_CAMERA_STATUSES,
    )
    from stream_manager import SourceType, utc_now


LOGGER = logging.getLogger(__name__)


class HealthMonitorError(RuntimeError):
    """Base health-monitoring error."""


class HealthEventType(str, Enum):
    """Types of structured health events."""

    CONNECTION_CHECK_FAILED = "connection_check_failed"
    STALE_DETECTED = "stale_detected"
    RECONNECT_STARTED = "reconnect_started"
    RECONNECT_SUCCEEDED = "reconnect_succeeded"
    RECONNECT_FAILED = "reconnect_failed"
    CAMERA_FAILED = "camera_failed"
    MONITOR_ERROR = "monitor_error"


@dataclass(frozen=True, slots=True)
class HealthCheckResult:
    """
    Result returned by an optional scheduler health-check callback.

    ``progress_observed`` should be true only when meaningful source or
    processing progress was observed. A process merely existing does not prove
    that an RTSP stream is delivering frames.
    """

    healthy: bool
    message: str
    progress_observed: bool = False


@dataclass(frozen=True, slots=True)
class ReconnectResult:
    """
    Result returned by a scheduler reconnect callback.

    A successful result means the callback re-established the processing
    contract. It does not independently prove production RTSP performance.
    """

    succeeded: bool
    message: str


@dataclass(frozen=True, slots=True)
class ReconnectPolicy:
    """Validated configurable reconnect policy."""

    enabled: bool
    eligible_source_types: frozenset[SourceType]
    maximum_attempts: int
    initial_delay_seconds: float
    backoff_multiplier: float
    maximum_delay_seconds: float

    def __post_init__(self) -> None:
        if self.maximum_attempts < 0:
            raise HealthMonitorError(
                "maximum_attempts cannot be negative"
            )

        if not math.isfinite(self.initial_delay_seconds):
            raise HealthMonitorError(
                "initial_delay_seconds must be finite"
            )

        if self.initial_delay_seconds < 0:
            raise HealthMonitorError(
                "initial_delay_seconds cannot be negative"
            )

        if not math.isfinite(self.backoff_multiplier):
            raise HealthMonitorError(
                "backoff_multiplier must be finite"
            )

        if self.backoff_multiplier < 1:
            raise HealthMonitorError(
                "backoff_multiplier must be at least 1"
            )

        if not math.isfinite(self.maximum_delay_seconds):
            raise HealthMonitorError(
                "maximum_delay_seconds must be finite"
            )

        if self.maximum_delay_seconds < 0:
            raise HealthMonitorError(
                "maximum_delay_seconds cannot be negative"
            )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "ReconnectPolicy":
        if not isinstance(value, Mapping):
            raise HealthMonitorError(
                "reconnect configuration must be a mapping"
            )

        raw_types = value.get(
            "eligible_source_types",
            [],
        )

        if not isinstance(raw_types, list):
            raise HealthMonitorError(
                "reconnect.eligible_source_types must be a list"
            )

        eligible_types = frozenset(
            SourceType.parse(item)
            for item in raw_types
        )

        return cls(
            enabled=_required_boolean(
                value,
                "enabled",
            ),
            eligible_source_types=eligible_types,
            maximum_attempts=_required_integer(
                value,
                "maximum_attempts",
            ),
            initial_delay_seconds=_required_number(
                value,
                "initial_delay_seconds",
            ),
            backoff_multiplier=_required_number(
                value,
                "backoff_multiplier",
            ),
            maximum_delay_seconds=_required_number(
                value,
                "maximum_delay_seconds",
            ),
        )

    def delay_for_attempt(
        self,
        attempt_number: int,
    ) -> float:
        if attempt_number <= 0:
            raise HealthMonitorError(
                "attempt_number must be positive"
            )

        delay = (
            self.initial_delay_seconds
            * (
                self.backoff_multiplier
                ** (attempt_number - 1)
            )
        )

        return min(delay, self.maximum_delay_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "eligible_source_types": sorted(
                item.value
                for item in self.eligible_source_types
            ),
            "maximum_attempts": self.maximum_attempts,
            "initial_delay_seconds": (
                self.initial_delay_seconds
            ),
            "backoff_multiplier": self.backoff_multiplier,
            "maximum_delay_seconds": (
                self.maximum_delay_seconds
            ),
        }


@dataclass(frozen=True, slots=True)
class HealthEvent:
    """One structured health-monitor event."""

    sequence: int
    timestamp_utc: str
    camera_id: str
    event_type: HealthEventType
    severity: str
    message: str
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "camera_id": self.camera_id,
            "event_type": self.event_type.value,
            "severity": self.severity,
            "message": self.message,
            "details": dict(self.details),
        }


ConnectionHealthCallback = Callable[
    [str],
    HealthCheckResult,
]
ReconnectCallback = Callable[
    [str, int],
    ReconnectResult,
]


class HealthMonitor:
    """
    Monitor camera heartbeats and coordinate bounded reconnect attempts.

    Callbacks must be bounded and thread-safe because they run from the health
    monitor thread. The monitor never assumes that reconnection succeeded; a
    successful ReconnectResult is required.
    """

    def __init__(
        self,
        *,
        camera_manager: CameraManager,
        enabled: bool,
        check_interval_seconds: float,
        default_stale_timeout_seconds: float,
        maximum_consecutive_health_failures: int,
        stale_detection_source_types: Iterable[SourceType],
        reconnect_policy: ReconnectPolicy,
        shutdown_event: threading.Event | None = None,
        stale_timeouts_by_camera: (
            Mapping[str, float] | None
        ) = None,
        maximum_events: int = 5000,
    ) -> None:
        if check_interval_seconds <= 0:
            raise HealthMonitorError(
                "check_interval_seconds must be positive"
            )

        if default_stale_timeout_seconds <= 0:
            raise HealthMonitorError(
                "default_stale_timeout_seconds must be positive"
            )

        if maximum_consecutive_health_failures <= 0:
            raise HealthMonitorError(
                "maximum_consecutive_health_failures "
                "must be positive"
            )

        if maximum_events <= 0:
            raise HealthMonitorError(
                "maximum_events must be positive"
            )

        self.camera_manager = camera_manager
        self.enabled = bool(enabled)
        self.check_interval_seconds = float(
            check_interval_seconds
        )
        self.default_stale_timeout_seconds = float(
            default_stale_timeout_seconds
        )
        self.maximum_consecutive_health_failures = int(
            maximum_consecutive_health_failures
        )
        self.stale_detection_source_types = frozenset(
            stale_detection_source_types
        )
        self.reconnect_policy = reconnect_policy
        self.shutdown_event = (
            shutdown_event or threading.Event()
        )

        self.stale_timeouts_by_camera = dict(
            stale_timeouts_by_camera or {}
        )

        for camera_id, timeout in (
            self.stale_timeouts_by_camera.items()
        ):
            if (
                not isinstance(camera_id, str)
                or not camera_id
            ):
                raise HealthMonitorError(
                    "Stale-timeout camera IDs must be "
                    "non-empty strings"
                )

            if not math.isfinite(timeout) or timeout <= 0:
                raise HealthMonitorError(
                    f"Stale timeout for {camera_id!r} "
                    "must be positive and finite"
                )

        self._events: deque[HealthEvent] = deque(
            maxlen=maximum_events
        )
        self._event_lock = threading.RLock()
        self._lifecycle_lock = threading.Lock()
        self._state_lock = threading.RLock()

        self._connection_health_callback: (
            ConnectionHealthCallback | None
        ) = None
        self._reconnect_callback: ReconnectCallback | None = None

        self._local_stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._started = False
        self._closed = False
        self._event_sequence = 0

        self._consecutive_failures: dict[str, int] = {}
        self._stale_in_progress: set[str] = set()

        self._checks_performed = 0
        self._stale_detections = 0
        self._reconnect_attempts = 0
        self._reconnect_successes = 0
        self._reconnect_failures = 0

    def set_callbacks(
        self,
        *,
        connection_health_callback: (
            ConnectionHealthCallback | None
        ) = None,
        reconnect_callback: ReconnectCallback | None = None,
    ) -> None:
        """
        Install scheduler callbacks before monitoring starts.

        Replacing callbacks after start could race with an active check and is
        therefore rejected.
        """

        with self._lifecycle_lock:
            if self._started:
                raise HealthMonitorError(
                    "Health callbacks must be configured "
                    "before the monitor starts"
                )

            if self._closed:
                raise HealthMonitorError(
                    "Cannot configure a closed HealthMonitor"
                )

            self._connection_health_callback = (
                connection_health_callback
            )
            self._reconnect_callback = reconnect_callback

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._closed:
                raise HealthMonitorError(
                    "Cannot start a closed HealthMonitor"
                )

            if self._started:
                return

            self._started = True

            if not self.enabled:
                LOGGER.info(
                    "Camera health monitoring is disabled"
                )
                return

            self._thread = threading.Thread(
                target=self._monitor_loop,
                name="phase7-health-monitor",
                daemon=True,
            )
            self._thread.start()

    def notify_progress(
        self,
        camera_id: str,
        *,
        message: str = "Camera made processing progress",
    ) -> None:
        """Record a meaningful progress heartbeat."""

        self.camera_manager.heartbeat(
            camera_id,
            message=message,
        )

        with self._state_lock:
            self._consecutive_failures[camera_id] = 0

        self.camera_manager.reset_health_failures(
            camera_id
        )

    def check_now(self) -> None:
        """Run one health-monitor iteration synchronously."""

        if not self.enabled:
            return

        if self._closed:
            raise HealthMonitorError(
                "Cannot check a closed HealthMonitor"
            )

        camera_ids = self.camera_manager.active_camera_ids()

        for camera_id in camera_ids:
            if self._should_stop():
                return

            try:
                self._check_camera(camera_id)
            except Exception as exc:
                message = (
                    "Unexpected health-monitor error: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._record_event(
                    camera_id=camera_id,
                    event_type=(
                        HealthEventType.MONITOR_ERROR
                    ),
                    severity="error",
                    message=message,
                    details={},
                )
                LOGGER.exception(
                    "Health check failed for camera %s",
                    camera_id,
                )

    def events(self) -> tuple[HealthEvent, ...]:
        with self._event_lock:
            return tuple(self._events)

    def summary(self) -> dict[str, Any]:
        with self._state_lock:
            failure_counts = dict(
                self._consecutive_failures
            )
            checks_performed = self._checks_performed
            stale_detections = self._stale_detections
            reconnect_attempts = self._reconnect_attempts
            reconnect_successes = self._reconnect_successes
            reconnect_failures = self._reconnect_failures

        return {
            "enabled": self.enabled,
            "started": self._started,
            "check_interval_seconds": (
                self.check_interval_seconds
            ),
            "default_stale_timeout_seconds": (
                self.default_stale_timeout_seconds
            ),
            "maximum_consecutive_health_failures": (
                self.maximum_consecutive_health_failures
            ),
            "stale_detection_source_types": sorted(
                item.value
                for item
                in self.stale_detection_source_types
            ),
            "reconnect_policy": (
                self.reconnect_policy.to_dict()
            ),
            "callbacks": {
                "connection_health_callback_configured": (
                    self._connection_health_callback
                    is not None
                ),
                "reconnect_callback_configured": (
                    self._reconnect_callback is not None
                ),
            },
            "checks_performed": checks_performed,
            "stale_detections": stale_detections,
            "reconnect_attempts": reconnect_attempts,
            "reconnect_successes": reconnect_successes,
            "reconnect_failures": reconnect_failures,
            "consecutive_failures": failure_counts,
            "events": [
                event.to_dict()
                for event in self.events()
            ],
            "limitations": [
                (
                    "RTSP health and reconnect behavior must be "
                    "validated with the target camera and network."
                )
            ],
        }

    def close(
        self,
        *,
        join_timeout_seconds: float = 10.0,
    ) -> None:
        if join_timeout_seconds < 0:
            raise HealthMonitorError(
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
                    "Health monitor did not stop within "
                    "%.3f seconds",
                    join_timeout_seconds,
                )

    def __enter__(self) -> "HealthMonitor":
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
            self.check_now()

            if self._local_stop_event.wait(
                self.check_interval_seconds
            ):
                break

    def _check_camera(
        self,
        camera_id: str,
    ) -> None:
        status = self.camera_manager.status(camera_id)

        if status in TERMINAL_CAMERA_STATUSES:
            return

        definition = self.camera_manager.definition(
            camera_id
        )

        with self._state_lock:
            self._checks_performed += 1

        if status is not CameraStatus.PROCESSING:
            return

        callback_result: HealthCheckResult | None = None

        if self._connection_health_callback is not None:
            try:
                callback_result = (
                    self._connection_health_callback(
                        camera_id
                    )
                )
            except Exception as exc:
                callback_result = HealthCheckResult(
                    healthy=False,
                    message=(
                        "Connection health callback failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                    progress_observed=False,
                )

            if not isinstance(
                callback_result,
                HealthCheckResult,
            ):
                callback_result = HealthCheckResult(
                    healthy=False,
                    message=(
                        "Connection health callback returned "
                        "an invalid result"
                    ),
                    progress_observed=False,
                )

            if callback_result.healthy:
                self._reset_failure_count(camera_id)

                if callback_result.progress_observed:
                    self.notify_progress(
                        camera_id,
                        message=callback_result.message,
                    )
            else:
                failures = self._increment_failure_count(
                    camera_id
                )
                self._record_event(
                    camera_id=camera_id,
                    event_type=(
                        HealthEventType
                        .CONNECTION_CHECK_FAILED
                    ),
                    severity="warning",
                    message=callback_result.message,
                    details={
                        "consecutive_failures": failures,
                        "failure_limit": (
                            self
                            .maximum_consecutive_health_failures
                        ),
                    },
                )

                if (
                    failures
                    >= self
                    .maximum_consecutive_health_failures
                ):
                    self._handle_stale(
                        camera_id,
                        definition.source_type,
                        reason=(
                            "Connection health failed "
                            f"{failures} consecutive times: "
                            f"{callback_result.message}"
                        ),
                    )
                    return

        if (
            definition.source_type
            not in self.stale_detection_source_types
        ):
            return

        heartbeat_age = (
            self.camera_manager.seconds_since_heartbeat(
                camera_id
            )
        )

        if heartbeat_age is None:
            return

        stale_timeout = (
            self.stale_timeouts_by_camera.get(
                camera_id,
                self.default_stale_timeout_seconds,
            )
        )

        if heartbeat_age > stale_timeout:
            self._handle_stale(
                camera_id,
                definition.source_type,
                reason=(
                    "No meaningful progress heartbeat for "
                    f"{heartbeat_age:.3f} seconds; stale "
                    f"timeout is {stale_timeout:.3f} seconds"
                ),
            )

    def _handle_stale(
        self,
        camera_id: str,
        source_type: SourceType,
        *,
        reason: str,
    ) -> None:
        with self._state_lock:
            if camera_id in self._stale_in_progress:
                return

            self._stale_in_progress.add(camera_id)
            self._stale_detections += 1

        try:
            if (
                self.camera_manager.status(camera_id)
                is not CameraStatus.PROCESSING
            ):
                return

            self.camera_manager.mark_stale(
                camera_id,
                message=reason,
            )
            self._record_event(
                camera_id=camera_id,
                event_type=HealthEventType.STALE_DETECTED,
                severity="warning",
                message=reason,
                details={
                    "source_type": source_type.value,
                },
            )

            reconnect_applicable = (
                self.reconnect_policy.enabled
                and source_type
                in self.reconnect_policy
                .eligible_source_types
            )

            if not reconnect_applicable:
                message = (
                    "Camera became stale and reconnect is not "
                    "enabled for source type "
                    f"{source_type.value!r}"
                )
                self.camera_manager.mark_failed(
                    camera_id,
                    message,
                )
                self._record_event(
                    camera_id=camera_id,
                    event_type=(
                        HealthEventType.CAMERA_FAILED
                    ),
                    severity="error",
                    message=message,
                    details={},
                )
                return

            if self._reconnect_callback is None:
                message = (
                    "Camera became stale, but no scheduler "
                    "reconnect callback is configured"
                )
                self.camera_manager.mark_failed(
                    camera_id,
                    message,
                )
                self._record_event(
                    camera_id=camera_id,
                    event_type=(
                        HealthEventType.CAMERA_FAILED
                    ),
                    severity="error",
                    message=message,
                    details={},
                )
                return

            self._run_reconnect_attempts(camera_id)
        except CameraManagerError as exc:
            self._record_event(
                camera_id=camera_id,
                event_type=HealthEventType.MONITOR_ERROR,
                severity="error",
                message=(
                    "Unable to update camera health state: "
                    f"{exc}"
                ),
                details={},
            )
        finally:
            with self._state_lock:
                self._stale_in_progress.discard(
                    camera_id
                )

    def _run_reconnect_attempts(
        self,
        camera_id: str,
    ) -> None:
        last_message = "No reconnect attempt was made"

        for attempt in range(
            1,
            self.reconnect_policy.maximum_attempts + 1,
        ):
            if self._should_stop():
                self.camera_manager.mark_cancelled(
                    camera_id,
                    "Reconnect cancelled during shutdown",
                )
                return

            delay = self.reconnect_policy.delay_for_attempt(
                attempt
            )

            if delay > 0 and self._wait(delay):
                self.camera_manager.mark_cancelled(
                    camera_id,
                    "Reconnect cancelled during shutdown",
                )
                return

            self.camera_manager.begin_reconnect(
                camera_id,
                message=(
                    f"Starting reconnect attempt {attempt} "
                    f"of "
                    f"{self.reconnect_policy.maximum_attempts}"
                ),
            )

            with self._state_lock:
                self._reconnect_attempts += 1

            self._record_event(
                camera_id=camera_id,
                event_type=(
                    HealthEventType.RECONNECT_STARTED
                ),
                severity="warning",
                message=(
                    f"Reconnect attempt {attempt} started"
                ),
                details={
                    "attempt": attempt,
                    "maximum_attempts": (
                        self.reconnect_policy
                        .maximum_attempts
                    ),
                    "delay_seconds": delay,
                },
            )

            try:
                result = self._reconnect_callback(
                    camera_id,
                    attempt,
                )
            except Exception as exc:
                result = ReconnectResult(
                    succeeded=False,
                    message=(
                        "Reconnect callback failed: "
                        f"{type(exc).__name__}: {exc}"
                    ),
                )

            if not isinstance(result, ReconnectResult):
                result = ReconnectResult(
                    succeeded=False,
                    message=(
                        "Reconnect callback returned an "
                        "invalid result"
                    ),
                )

            last_message = result.message

            if result.succeeded:
                self.camera_manager.reconnect_succeeded(
                    camera_id,
                    message=result.message,
                )
                self._reset_failure_count(camera_id)

                with self._state_lock:
                    self._reconnect_successes += 1

                self._record_event(
                    camera_id=camera_id,
                    event_type=(
                        HealthEventType
                        .RECONNECT_SUCCEEDED
                    ),
                    severity="info",
                    message=result.message,
                    details={"attempt": attempt},
                )
                return

            with self._state_lock:
                self._reconnect_failures += 1

            self._record_event(
                camera_id=camera_id,
                event_type=(
                    HealthEventType.RECONNECT_FAILED
                ),
                severity="warning",
                message=result.message,
                details={"attempt": attempt},
            )

        message = (
            "Reconnect attempts exhausted: "
            f"{last_message}"
        )
        self.camera_manager.mark_failed(
            camera_id,
            message,
        )
        self._record_event(
            camera_id=camera_id,
            event_type=HealthEventType.CAMERA_FAILED,
            severity="error",
            message=message,
            details={
                "attempts": (
                    self.reconnect_policy.maximum_attempts
                )
            },
        )

    def _increment_failure_count(
        self,
        camera_id: str,
    ) -> int:
        with self._state_lock:
            value = (
                self._consecutive_failures.get(
                    camera_id,
                    0,
                )
                + 1
            )
            self._consecutive_failures[camera_id] = value
            return value

    def _reset_failure_count(
        self,
        camera_id: str,
    ) -> None:
        with self._state_lock:
            self._consecutive_failures[camera_id] = 0

        self.camera_manager.reset_health_failures(
            camera_id
        )

    def _record_event(
        self,
        *,
        camera_id: str,
        event_type: HealthEventType,
        severity: str,
        message: str,
        details: Mapping[str, Any],
    ) -> None:
        with self._event_lock:
            self._event_sequence += 1
            self._events.append(
                HealthEvent(
                    sequence=self._event_sequence,
                    timestamp_utc=utc_now(),
                    camera_id=camera_id,
                    event_type=event_type,
                    severity=severity,
                    message=message,
                    details=dict(details),
                )
            )

    def _wait(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds

        while True:
            if self._should_stop():
                return True

            remaining = deadline - time.monotonic()

            if remaining <= 0:
                return False

            self._local_stop_event.wait(
                min(0.1, remaining)
            )

    def _should_stop(self) -> bool:
        return (
            self._local_stop_event.is_set()
            or self.shutdown_event.is_set()
        )


def _required_boolean(
    mapping: Mapping[str, Any],
    key: str,
) -> bool:
    value = mapping.get(key)

    if not isinstance(value, bool):
        raise HealthMonitorError(
            f"reconnect.{key} must be boolean"
        )

    return value


def _required_integer(
    mapping: Mapping[str, Any],
    key: str,
) -> int:
    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        raise HealthMonitorError(
            f"reconnect.{key} must be an integer"
        )

    return value


def _required_number(
    mapping: Mapping[str, Any],
    key: str,
) -> float:
    value = mapping.get(key)

    if isinstance(value, bool):
        raise HealthMonitorError(
            f"reconnect.{key} must be numeric"
        )

    try:
        result = float(value)
    except (TypeError, ValueError):
        raise HealthMonitorError(
            f"reconnect.{key} must be numeric"
        ) from None

    if not math.isfinite(result):
        raise HealthMonitorError(
            f"reconnect.{key} must be finite"
        )

    return result


__all__ = [
    "ConnectionHealthCallback",
    "HealthCheckResult",
    "HealthEvent",
    "HealthEventType",
    "HealthMonitor",
    "HealthMonitorError",
    "ReconnectCallback",
    "ReconnectPolicy",
    "ReconnectResult",
]