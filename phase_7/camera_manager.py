"""
Thread-safe camera registration and lifecycle tracking for Phase 7.

Each camera receives:

- a unique validated camera ID;
- an isolated output directory;
- an independent lifecycle state;
- inspection and processing results;
- warnings, errors, attempts, and heartbeat data;
- a chronological event history.

This module does not perform compression. It records only results supplied by
the scheduler and processing adapter.
"""

from __future__ import annotations

import copy
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from .stream_manager import (
        BaseVideoStream,
        StreamDefinition,
        StreamInspection,
        StreamManager,
        utc_now,
    )
except ImportError:
    from stream_manager import (
        BaseVideoStream,
        StreamDefinition,
        StreamInspection,
        StreamManager,
        utc_now,
    )


class CameraManagerError(RuntimeError):
    """Base camera-management error."""


class CameraNotFoundError(CameraManagerError):
    """Raised when an unknown camera ID is requested."""


class DuplicateCameraError(CameraManagerError):
    """Raised when a camera ID is registered more than once."""


class InvalidCameraTransitionError(CameraManagerError):
    """Raised when a camera lifecycle transition is invalid."""


class CameraStatus(str, Enum):
    """Lifecycle states for independently managed cameras."""

    REGISTERED = "registered"
    DISABLED = "disabled"
    QUEUED = "queued"
    VALIDATING = "validating"
    READY = "ready"
    WAITING_RESOURCES = "waiting_resources"
    PROCESSING = "processing"
    STALE = "stale"
    RECONNECTING = "reconnecting"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_CAMERA_STATUSES = frozenset(
    {
        CameraStatus.DISABLED,
        CameraStatus.COMPLETED,
        CameraStatus.SKIPPED,
        CameraStatus.FAILED,
        CameraStatus.CANCELLED,
    }
)


_ALLOWED_TRANSITIONS: dict[
    CameraStatus,
    frozenset[CameraStatus],
] = {
    CameraStatus.REGISTERED: frozenset(
        {
            CameraStatus.DISABLED,
            CameraStatus.QUEUED,
            CameraStatus.VALIDATING,
            CameraStatus.SKIPPED,
            CameraStatus.FAILED,
            CameraStatus.CANCELLED,
        }
    ),
    CameraStatus.QUEUED: frozenset(
        {
            CameraStatus.VALIDATING,
            CameraStatus.SKIPPED,
            CameraStatus.FAILED,
            CameraStatus.CANCELLED,
        }
    ),
    CameraStatus.VALIDATING: frozenset(
        {
            CameraStatus.READY,
            CameraStatus.SKIPPED,
            CameraStatus.FAILED,
            CameraStatus.CANCELLED,
        }
    ),
    CameraStatus.READY: frozenset(
        {
            CameraStatus.WAITING_RESOURCES,
            CameraStatus.PROCESSING,
            CameraStatus.STALE,
            CameraStatus.COMPLETED,
            CameraStatus.SKIPPED,
            CameraStatus.FAILED,
            CameraStatus.CANCELLED,
        }
    ),
    CameraStatus.WAITING_RESOURCES: frozenset(
        {
            CameraStatus.READY,
            CameraStatus.PROCESSING,
            CameraStatus.STALE,
            CameraStatus.FAILED,
            CameraStatus.CANCELLED,
        }
    ),
    CameraStatus.PROCESSING: frozenset(
        {
            CameraStatus.STALE,
            CameraStatus.RECONNECTING,
            CameraStatus.COMPLETED,
            CameraStatus.FAILED,
            CameraStatus.CANCELLED,
        }
    ),
    CameraStatus.STALE: frozenset(
        {
            CameraStatus.PROCESSING,
            CameraStatus.RECONNECTING,
            CameraStatus.FAILED,
            CameraStatus.CANCELLED,
        }
    ),
    CameraStatus.RECONNECTING: frozenset(
        {
            CameraStatus.PROCESSING,
            CameraStatus.STALE,
            CameraStatus.FAILED,
            CameraStatus.CANCELLED,
        }
    ),
    CameraStatus.DISABLED: frozenset(),
    CameraStatus.COMPLETED: frozenset(),
    CameraStatus.SKIPPED: frozenset(),
    CameraStatus.FAILED: frozenset(),
    CameraStatus.CANCELLED: frozenset(),
}


@dataclass(frozen=True, slots=True)
class CameraEvent:
    """One immutable camera lifecycle event."""

    sequence: int
    timestamp_utc: str
    previous_status: str | None
    status: str
    message: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "timestamp_utc": self.timestamp_utc,
            "previous_status": self.previous_status,
            "status": self.status,
            "message": self.message,
        }


@dataclass(slots=True)
class CameraRecord:
    """Mutable internal record protected by CameraManager's lock."""

    definition: StreamDefinition
    output_directory: Path
    stream: BaseVideoStream | None = field(
        default=None,
        repr=False,
    )
    status: CameraStatus = CameraStatus.REGISTERED
    created_at_utc: str = field(default_factory=utc_now)
    updated_at_utc: str = field(default_factory=utc_now)
    started_at_utc: str | None = None
    completed_at_utc: str | None = None
    last_heartbeat_at_utc: str | None = None
    last_heartbeat_monotonic: float | None = field(
        default=None,
        repr=False,
    )
    health_message: str | None = None
    processing_attempts: int = 0
    reconnect_attempts: int = 0
    consecutive_health_failures: int = 0
    inspection: StreamInspection | None = None
    processing_result: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    events: list[CameraEvent] = field(default_factory=list)


class CameraManager:
    """
    Register cameras and maintain isolated, thread-safe camera state.

    The run output directory must already identify one unique Phase 7 run.
    CameraManager creates one child directory per enabled camera under:

        <run_output_directory>/cameras/<camera_id>/
    """

    def __init__(
        self,
        *,
        stream_manager: StreamManager,
        run_output_directory: Path,
        redact_source_credentials: bool = True,
    ) -> None:
        self.stream_manager = stream_manager
        self.run_output_directory = (
            run_output_directory.expanduser().resolve()
        )
        self.camera_output_root = (
            self.run_output_directory / "cameras"
        ).resolve()
        self.redact_source_credentials = bool(
            redact_source_credentials
        )

        self._records: dict[str, CameraRecord] = {}
        self._lock = threading.RLock()
        self._event_sequence = 0
        self._closed = False

        self._create_directory(self.run_output_directory)
        self._create_directory(self.camera_output_root)

    def register(
        self,
        definition: StreamDefinition,
    ) -> None:
        """Register one camera and reserve its isolated output path."""

        with self._lock:
            if self._closed:
                raise CameraManagerError(
                    "Cannot register cameras after manager shutdown"
                )

            camera_id = definition.camera_id

            if camera_id in self._records:
                raise DuplicateCameraError(
                    f"Camera is already registered: {camera_id!r}"
                )

            output_directory = self._safe_camera_output_directory(
                camera_id
            )

            stream: BaseVideoStream | None = None

            if definition.enabled:
                self._create_new_camera_directory(
                    output_directory
                )
                stream = self.stream_manager.register(definition)
                initial_status = CameraStatus.REGISTERED
                initial_message = "Camera registered"
            else:
                initial_status = CameraStatus.DISABLED
                initial_message = (
                    "Camera is disabled by configuration"
                )

            record = CameraRecord(
                definition=definition,
                output_directory=output_directory,
                stream=stream,
                status=initial_status,
            )
            record.events.append(
                self._new_event(
                    previous_status=None,
                    status=initial_status,
                    message=initial_message,
                )
            )

            if initial_status in TERMINAL_CAMERA_STATUSES:
                record.completed_at_utc = record.created_at_utc

            self._records[camera_id] = record

    def register_all(
        self,
        definitions: Iterable[StreamDefinition],
    ) -> None:
        """Register definitions in their configured order."""

        for definition in definitions:
            self.register(definition)

    def camera_ids(
        self,
        *,
        include_disabled: bool = True,
    ) -> tuple[str, ...]:
        with self._lock:
            if include_disabled:
                return tuple(self._records)

            return tuple(
                camera_id
                for camera_id, record in self._records.items()
                if record.status is not CameraStatus.DISABLED
            )

    def active_camera_ids(self) -> tuple[str, ...]:
        """Return enabled cameras that have not reached a terminal state."""

        with self._lock:
            return tuple(
                camera_id
                for camera_id, record in self._records.items()
                if record.status
                not in TERMINAL_CAMERA_STATUSES
            )

    def definition(
        self,
        camera_id: str,
    ) -> StreamDefinition:
        with self._lock:
            return self._get_record(camera_id).definition

    def stream(
        self,
        camera_id: str,
    ) -> BaseVideoStream:
        with self._lock:
            record = self._get_record(camera_id)

            if record.stream is None:
                raise CameraManagerError(
                    f"Camera {camera_id!r} has no active stream"
                )

            return record.stream

    def output_directory(
        self,
        camera_id: str,
    ) -> Path:
        with self._lock:
            return self._get_record(
                camera_id
            ).output_directory

    def status(
        self,
        camera_id: str,
    ) -> CameraStatus:
        with self._lock:
            return self._get_record(camera_id).status

    def is_terminal(
        self,
        camera_id: str,
    ) -> bool:
        return self.status(camera_id) in TERMINAL_CAMERA_STATUSES

    def transition(
        self,
        camera_id: str,
        status: CameraStatus,
        *,
        message: str | None = None,
    ) -> None:
        """Apply a validated camera lifecycle transition."""

        with self._lock:
            record = self._get_record(camera_id)
            self._transition_unlocked(
                record,
                status,
                message=message,
            )

    def queue(
        self,
        camera_id: str,
    ) -> None:
        self.transition(
            camera_id,
            CameraStatus.QUEUED,
            message="Camera queued for fair scheduling",
        )

    def begin_validation(
        self,
        camera_id: str,
    ) -> None:
        self.transition(
            camera_id,
            CameraStatus.VALIDATING,
            message="Validating and probing camera source",
        )

    def record_inspection(
        self,
        camera_id: str,
        inspection: StreamInspection,
    ) -> None:
        """
        Store inspection data and derive the next honest camera state.

        A required unreadable source fails. An optional unreadable source is
        skipped with a warning.
        """

        with self._lock:
            record = self._get_record(camera_id)

            if inspection.camera_id != camera_id:
                raise CameraManagerError(
                    "Inspection camera_id does not match target "
                    f"camera: {inspection.camera_id!r} != "
                    f"{camera_id!r}"
                )

            record.inspection = inspection
            record.updated_at_utc = utc_now()

            if inspection.successful:
                self._transition_unlocked(
                    record,
                    CameraStatus.READY,
                    message=inspection.message,
                )
                self._heartbeat_unlocked(
                    record,
                    message="Source inspection succeeded",
                )
                return

            message = inspection.message

            if record.definition.required:
                record.error = message
                self._transition_unlocked(
                    record,
                    CameraStatus.FAILED,
                    message=message,
                )
            else:
                self._append_warning_unlocked(record, message)
                self._transition_unlocked(
                    record,
                    CameraStatus.SKIPPED,
                    message=(
                        "Optional camera skipped: "
                        f"{message}"
                    ),
                )

    def begin_resource_wait(
        self,
        camera_id: str,
        *,
        message: str,
    ) -> None:
        self.transition(
            camera_id,
            CameraStatus.WAITING_RESOURCES,
            message=message,
        )

    def end_resource_wait(
        self,
        camera_id: str,
        *,
        message: str,
    ) -> None:
        self.transition(
            camera_id,
            CameraStatus.READY,
            message=message,
        )

    def begin_processing(
        self,
        camera_id: str,
        *,
        message: str = "Camera processing started",
    ) -> int:
        """Increment the processing attempt and enter processing state."""

        with self._lock:
            record = self._get_record(camera_id)
            record.processing_attempts += 1
            self._transition_unlocked(
                record,
                CameraStatus.PROCESSING,
                message=message,
            )
            self._heartbeat_unlocked(
                record,
                message="Processing started",
            )
            return record.processing_attempts

    def record_processing_result(
        self,
        camera_id: str,
        result: Mapping[str, Any],
        *,
        successful: bool,
        message: str,
    ) -> None:
        """
        Store an adapter result.

        ``successful`` must come from explicit adapter validation. This method
        never infers compression success from the presence of a process alone.
        """

        with self._lock:
            record = self._get_record(camera_id)
            record.processing_result = copy.deepcopy(dict(result))
            record.updated_at_utc = utc_now()

            if successful:
                self._transition_unlocked(
                    record,
                    CameraStatus.COMPLETED,
                    message=message,
                )
            else:
                record.error = message
                self._transition_unlocked(
                    record,
                    CameraStatus.FAILED,
                    message=message,
                )

    def complete_probe_only(
        self,
        camera_id: str,
        *,
        message: str = (
            "Probe-only inspection completed; compression was not run"
        ),
    ) -> None:
        """
        Complete a probe-only camera without claiming compression success.
        """

        with self._lock:
            record = self._get_record(camera_id)

            if record.inspection is None:
                raise CameraManagerError(
                    f"Cannot complete probe-only camera "
                    f"{camera_id!r} without an inspection"
                )

            if not record.inspection.successful:
                raise CameraManagerError(
                    f"Cannot complete probe-only camera "
                    f"{camera_id!r} after a failed inspection"
                )

            record.processing_result = {
                "mode": "probe_only",
                "processing_invoked": False,
                "compression_ran": False,
                "compression_success": False,
                "source_probe_success": True,
                "message": message,
            }

            self._transition_unlocked(
                record,
                CameraStatus.COMPLETED,
                message=message,
            )

    def mark_stale(
        self,
        camera_id: str,
        *,
        message: str,
    ) -> None:
        with self._lock:
            record = self._get_record(camera_id)
            record.consecutive_health_failures += 1
            self._transition_unlocked(
                record,
                CameraStatus.STALE,
                message=message,
            )

    def begin_reconnect(
        self,
        camera_id: str,
        *,
        message: str,
    ) -> int:
        with self._lock:
            record = self._get_record(camera_id)
            record.reconnect_attempts += 1
            self._transition_unlocked(
                record,
                CameraStatus.RECONNECTING,
                message=message,
            )
            return record.reconnect_attempts

    def reconnect_succeeded(
        self,
        camera_id: str,
        *,
        message: str,
    ) -> None:
        with self._lock:
            record = self._get_record(camera_id)
            record.consecutive_health_failures = 0
            self._transition_unlocked(
                record,
                CameraStatus.PROCESSING,
                message=message,
            )
            self._heartbeat_unlocked(
                record,
                message="Reconnect succeeded",
            )

    def mark_failed(
        self,
        camera_id: str,
        message: str,
    ) -> None:
        with self._lock:
            record = self._get_record(camera_id)

            if record.status in TERMINAL_CAMERA_STATUSES:
                return

            record.error = message
            self._transition_unlocked(
                record,
                CameraStatus.FAILED,
                message=message,
            )

    def mark_skipped(
        self,
        camera_id: str,
        message: str,
    ) -> None:
        with self._lock:
            record = self._get_record(camera_id)

            if record.status in TERMINAL_CAMERA_STATUSES:
                return

            self._append_warning_unlocked(record, message)
            self._transition_unlocked(
                record,
                CameraStatus.SKIPPED,
                message=message,
            )

    def mark_cancelled(
        self,
        camera_id: str,
        message: str = "Cancelled during shutdown",
    ) -> None:
        with self._lock:
            record = self._get_record(camera_id)

            if record.status in TERMINAL_CAMERA_STATUSES:
                return

            self._transition_unlocked(
                record,
                CameraStatus.CANCELLED,
                message=message,
            )

    def cancel_nonterminal(
        self,
        message: str = "Cancelled during shutdown",
    ) -> None:
        with self._lock:
            for record in self._records.values():
                if record.status not in TERMINAL_CAMERA_STATUSES:
                    self._transition_unlocked(
                        record,
                        CameraStatus.CANCELLED,
                        message=message,
                    )

    def add_warning(
        self,
        camera_id: str,
        message: str,
    ) -> None:
        with self._lock:
            record = self._get_record(camera_id)
            self._append_warning_unlocked(record, message)
            record.updated_at_utc = utc_now()

    def heartbeat(
        self,
        camera_id: str,
        *,
        message: str | None = None,
    ) -> None:
        with self._lock:
            record = self._get_record(camera_id)

            if record.status in TERMINAL_CAMERA_STATUSES:
                return

            self._heartbeat_unlocked(
                record,
                message=message,
            )

    def seconds_since_heartbeat(
        self,
        camera_id: str,
    ) -> float | None:
        with self._lock:
            record = self._get_record(camera_id)
            last = record.last_heartbeat_monotonic

            if last is None:
                return None

            return max(0.0, time.monotonic() - last)

    def consecutive_health_failures(
        self,
        camera_id: str,
    ) -> int:
        with self._lock:
            return self._get_record(
                camera_id
            ).consecutive_health_failures

    def reset_health_failures(
        self,
        camera_id: str,
    ) -> None:
        with self._lock:
            record = self._get_record(camera_id)
            record.consecutive_health_failures = 0
            record.updated_at_utc = utc_now()

    def snapshot(
        self,
        camera_id: str,
    ) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked(
                self._get_record(camera_id)
            )

    def snapshots(self) -> list[dict[str, Any]]:
        with self._lock:
            return [
                self._snapshot_unlocked(record)
                for record in self._records.values()
            ]

    def status_counts(self) -> dict[str, int]:
        with self._lock:
            counts = {
                status.value: 0
                for status in CameraStatus
            }

            for record in self._records.values():
                counts[record.status.value] += 1

            return {
                key: value
                for key, value in counts.items()
                if value > 0
            }

    def required_failures(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                camera_id
                for camera_id, record in self._records.items()
                if record.definition.required
                and record.status
                in {
                    CameraStatus.FAILED,
                    CameraStatus.CANCELLED,
                }
            )

    def all_terminal(self) -> bool:
        with self._lock:
            return all(
                record.status in TERMINAL_CAMERA_STATUSES
                for record in self._records.values()
            )

    def close(self) -> None:
        """Cancel unfinished cameras and close the stream manager."""

        with self._lock:
            if self._closed:
                return

            self._closed = True

            for record in self._records.values():
                if record.status not in TERMINAL_CAMERA_STATUSES:
                    self._transition_unlocked(
                        record,
                        CameraStatus.CANCELLED,
                        message="Camera manager closed",
                    )

        self.stream_manager.close()

    def __enter__(self) -> "CameraManager":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()

    def _get_record(
        self,
        camera_id: str,
    ) -> CameraRecord:
        try:
            return self._records[camera_id]
        except KeyError as exc:
            raise CameraNotFoundError(
                f"Unknown camera_id: {camera_id!r}"
            ) from exc

    def _transition_unlocked(
        self,
        record: CameraRecord,
        status: CameraStatus,
        *,
        message: str | None,
    ) -> None:
        previous = record.status

        if previous is status:
            record.updated_at_utc = utc_now()

            if message:
                record.health_message = message

            return

        allowed = _ALLOWED_TRANSITIONS[previous]

        if status not in allowed:
            raise InvalidCameraTransitionError(
                f"Invalid camera transition for "
                f"{record.definition.camera_id!r}: "
                f"{previous.value} -> {status.value}"
            )

        timestamp = utc_now()
        record.status = status
        record.updated_at_utc = timestamp

        if (
            record.started_at_utc is None
            and status
            in {
                CameraStatus.VALIDATING,
                CameraStatus.PROCESSING,
            }
        ):
            record.started_at_utc = timestamp

        if status in TERMINAL_CAMERA_STATUSES:
            record.completed_at_utc = timestamp

        record.events.append(
            self._new_event(
                previous_status=previous,
                status=status,
                message=message,
            )
        )

    def _heartbeat_unlocked(
        self,
        record: CameraRecord,
        *,
        message: str | None,
    ) -> None:
        record.last_heartbeat_at_utc = utc_now()
        record.last_heartbeat_monotonic = time.monotonic()
        record.updated_at_utc = record.last_heartbeat_at_utc

        if message is not None:
            record.health_message = message

    def _append_warning_unlocked(
        self,
        record: CameraRecord,
        message: str,
    ) -> None:
        normalized = message.strip()

        if normalized and normalized not in record.warnings:
            record.warnings.append(normalized)

    def _new_event(
        self,
        *,
        previous_status: CameraStatus | None,
        status: CameraStatus,
        message: str | None,
    ) -> CameraEvent:
        self._event_sequence += 1

        return CameraEvent(
            sequence=self._event_sequence,
            timestamp_utc=utc_now(),
            previous_status=(
                previous_status.value
                if previous_status is not None
                else None
            ),
            status=status.value,
            message=message,
        )

    def _snapshot_unlocked(
        self,
        record: CameraRecord,
    ) -> dict[str, Any]:
        return {
            "camera": record.definition.to_public_dict(
                redact_credentials=self.redact_source_credentials
            ),
            "status": record.status.value,
            "output_directory": str(record.output_directory),
            "created_at_utc": record.created_at_utc,
            "updated_at_utc": record.updated_at_utc,
            "started_at_utc": record.started_at_utc,
            "completed_at_utc": record.completed_at_utc,
            "last_heartbeat_at_utc": (
                record.last_heartbeat_at_utc
            ),
            "health_message": record.health_message,
            "processing_attempts": record.processing_attempts,
            "reconnect_attempts": record.reconnect_attempts,
            "consecutive_health_failures": (
                record.consecutive_health_failures
            ),
            "inspection": (
                record.inspection.to_dict()
                if record.inspection is not None
                else None
            ),
            "processing_result": copy.deepcopy(
                record.processing_result
            ),
            "warnings": list(record.warnings),
            "error": record.error,
            "events": [
                event.to_dict()
                for event in record.events
            ],
        }

    def _safe_camera_output_directory(
        self,
        camera_id: str,
    ) -> Path:
        candidate = (
            self.camera_output_root / camera_id
        ).resolve()

        try:
            common = Path(
                os.path.commonpath(
                    [
                        str(self.camera_output_root),
                        str(candidate),
                    ]
                )
            )
        except ValueError as exc:
            raise CameraManagerError(
                f"Camera output path is on an unexpected volume: "
                f"{camera_id!r}"
            ) from exc

        if common != self.camera_output_root:
            raise CameraManagerError(
                f"Unsafe camera output path for "
                f"{camera_id!r}: {candidate}"
            )

        return candidate

    @staticmethod
    def _create_directory(path: Path) -> None:
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise CameraManagerError(
                f"Unable to create directory {path}: {exc}"
            ) from exc

        if not path.is_dir():
            raise CameraManagerError(
                f"Expected a directory but found another "
                f"filesystem object: {path}"
            )

    @staticmethod
    def _create_new_camera_directory(
        path: Path,
    ) -> None:
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise CameraManagerError(
                f"Camera output directory already exists: {path}"
            ) from exc
        except OSError as exc:
            raise CameraManagerError(
                f"Unable to create camera output directory "
                f"{path}: {exc}"
            ) from exc


__all__ = [
    "CameraEvent",
    "CameraManager",
    "CameraManagerError",
    "CameraNotFoundError",
    "CameraRecord",
    "CameraStatus",
    "DuplicateCameraError",
    "InvalidCameraTransitionError",
    "TERMINAL_CAMERA_STATUSES",
]