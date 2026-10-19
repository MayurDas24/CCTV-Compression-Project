"""
Video-source abstraction and media probing for Phase 7.

This module supports finite local video files and defines an RTSP-compatible
abstraction for future camera integration. RTSP support is intentionally not
treated as production-validated until it has been tested with real cameras.

FFprobe is used for media inspection. Frame counts are never estimated from
FPS and duration. For finite local files, missing container frame metadata
triggers an ffprobe decoded-frame-count fallback.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import SplitResult, urlsplit, urlunsplit


class StreamManagerError(RuntimeError):
    """Base error raised by the Phase 7 stream layer."""


class StreamConfigurationError(StreamManagerError):
    """Raised when a stream definition is invalid."""


class StreamProbeError(StreamManagerError):
    """Raised when FFprobe cannot reliably inspect a source."""


class StreamCancelledError(StreamManagerError):
    """Raised when probing is interrupted by shutdown."""


class SourceType(str, Enum):
    """Supported Phase 7 source types."""

    LOCAL_FILE = "local_file"
    RTSP = "rtsp"

    @classmethod
    def parse(cls, value: str) -> "SourceType":
        if not isinstance(value, str):
            raise StreamConfigurationError(
                "source_type must be a string"
            )

        normalized = value.strip().lower()

        try:
            return cls(normalized)
        except ValueError as exc:
            supported = ", ".join(item.value for item in cls)
            raise StreamConfigurationError(
                f"Unsupported source_type {value!r}; "
                f"supported values: {supported}"
            ) from exc


class StreamLifecycleState(str, Enum):
    """Internal lifecycle of a registered source."""

    REGISTERED = "registered"
    VALIDATING = "validating"
    READY = "ready"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    CLOSED = "closed"


_CAMERA_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


@dataclass(frozen=True, slots=True)
class StreamDefinition:
    """Validated, configuration-level description of one source."""

    camera_id: str
    source_type: SourceType
    source: str = field(repr=False)
    required: bool = True
    enabled: bool = True
    display_name: str | None = None
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        camera_id = self.camera_id.strip()

        if not _CAMERA_ID_PATTERN.fullmatch(camera_id):
            raise StreamConfigurationError(
                "camera_id must begin with a letter or digit, contain only "
                "letters, digits, underscores, hyphens, or periods, and be "
                "no longer than 64 characters"
            )

        if not isinstance(self.source_type, SourceType):
            raise StreamConfigurationError(
                "source_type must be a SourceType"
            )

        if not isinstance(self.source, str) or not self.source.strip():
            raise StreamConfigurationError(
                f"Source for camera {camera_id!r} must be non-empty"
            )

        if not isinstance(self.required, bool):
            raise StreamConfigurationError(
                f"required for camera {camera_id!r} must be boolean"
            )

        if not isinstance(self.enabled, bool):
            raise StreamConfigurationError(
                f"enabled for camera {camera_id!r} must be boolean"
            )

        if self.display_name is not None:
            if (
                not isinstance(self.display_name, str)
                or not self.display_name.strip()
            ):
                raise StreamConfigurationError(
                    f"display_name for camera {camera_id!r} must be "
                    "a non-empty string or null"
                )

        for tag in self.tags:
            if not isinstance(tag, str) or not tag.strip():
                raise StreamConfigurationError(
                    f"Every tag for camera {camera_id!r} must be "
                    "a non-empty string"
                )

        object.__setattr__(self, "camera_id", camera_id)
        object.__setattr__(self, "source", self.source.strip())

        if self.display_name is not None:
            object.__setattr__(
                self,
                "display_name",
                self.display_name.strip(),
            )

        object.__setattr__(
            self,
            "tags",
            tuple(tag.strip() for tag in self.tags),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> "StreamDefinition":
        """Create a stream definition from one YAML source mapping."""

        if not isinstance(value, Mapping):
            raise StreamConfigurationError(
                "Each source entry must be a mapping"
            )

        camera_id = value.get("camera_id")
        source_type = value.get("source_type")
        source = value.get("source")

        if not isinstance(camera_id, str):
            raise StreamConfigurationError(
                "Every source requires a string camera_id"
            )

        if not isinstance(source_type, str):
            raise StreamConfigurationError(
                f"Camera {camera_id!r} requires a string source_type"
            )

        if not isinstance(source, str):
            raise StreamConfigurationError(
                f"Camera {camera_id!r} requires a string source"
            )

        required = value.get("required", True)
        enabled = value.get("enabled", True)
        display_name = value.get("display_name")

        raw_tags = value.get("tags", [])

        if not isinstance(raw_tags, list):
            raise StreamConfigurationError(
                f"tags for camera {camera_id!r} must be a list"
            )

        if not all(isinstance(tag, str) for tag in raw_tags):
            raise StreamConfigurationError(
                f"tags for camera {camera_id!r} must contain strings"
            )

        return cls(
            camera_id=camera_id,
            source_type=SourceType.parse(source_type),
            source=source,
            required=required,
            enabled=enabled,
            display_name=display_name,
            tags=tuple(raw_tags),
        )

    def to_public_dict(
        self,
        *,
        redact_credentials: bool = True,
    ) -> dict[str, Any]:
        source = (
            redact_source(self.source)
            if redact_credentials
            else self.source
        )

        return {
            "camera_id": self.camera_id,
            "source_type": self.source_type.value,
            "source": source,
            "required": self.required,
            "enabled": self.enabled,
            "display_name": self.display_name,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class MediaProbe:
    """Reliable metadata returned by FFprobe."""

    width: int
    height: int
    codec_name: str | None
    pixel_format: str | None
    average_fps: float | None
    duration_seconds: float | None
    frame_count: int | None
    frame_count_source: str | None
    format_name: str | None
    size_bytes: int | None
    bit_rate_bps: int | None
    has_audio: bool
    probed_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "resolution": f"{self.width}x{self.height}",
            "codec_name": self.codec_name,
            "pixel_format": self.pixel_format,
            "average_fps": self.average_fps,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "frame_count_source": self.frame_count_source,
            "format_name": self.format_name,
            "size_bytes": self.size_bytes,
            "bit_rate_bps": self.bit_rate_bps,
            "has_audio": self.has_audio,
            "probed_at_utc": self.probed_at_utc,
        }


@dataclass(frozen=True, slots=True)
class StreamInspection:
    """Result of validating and probing one stream."""

    camera_id: str
    source_type: SourceType
    safe_source: str
    available: bool
    readable: bool
    message: str
    probe: MediaProbe | None
    started_at_utc: str
    completed_at_utc: str
    elapsed_seconds: float

    @property
    def successful(self) -> bool:
        return self.available and self.readable and self.probe is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "source_type": self.source_type.value,
            "source": self.safe_source,
            "available": self.available,
            "readable": self.readable,
            "successful": self.successful,
            "message": self.message,
            "probe": (
                self.probe.to_dict()
                if self.probe is not None
                else None
            ),
            "started_at_utc": self.started_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "elapsed_seconds": self.elapsed_seconds,
        }


class BaseVideoStream(ABC):
    """Common source abstraction used by the scheduler."""

    def __init__(
        self,
        definition: StreamDefinition,
        *,
        ffprobe_path: str,
        probe_timeout_seconds: float,
        stop_event: threading.Event,
    ) -> None:
        if probe_timeout_seconds <= 0:
            raise StreamConfigurationError(
                "probe_timeout_seconds must be positive"
            )

        self.definition = definition
        self.ffprobe_path = ffprobe_path
        self.probe_timeout_seconds = float(
            probe_timeout_seconds
        )
        self.stop_event = stop_event

        self._state = StreamLifecycleState.REGISTERED
        self._state_lock = threading.Lock()

    @property
    def camera_id(self) -> str:
        return self.definition.camera_id

    @property
    def state(self) -> StreamLifecycleState:
        with self._state_lock:
            return self._state

    def _set_state(
        self,
        value: StreamLifecycleState,
    ) -> None:
        with self._state_lock:
            self._state = value

    @property
    @abstractmethod
    def source_argument(self) -> str:
        """Actual source value passed to FFprobe or a pipeline."""

    @property
    def safe_source(self) -> str:
        """Source safe for logs and reports."""

        return redact_source(self.source_argument)

    @property
    @abstractmethod
    def supports_frame_count_fallback(self) -> bool:
        """Whether a complete decoded-frame count can terminate safely."""

    @abstractmethod
    def inspect(self) -> StreamInspection:
        """Validate and probe the source."""

    def close(self) -> None:
        """Release stream-owned resources."""

        self._set_state(StreamLifecycleState.CLOSED)

    def _probe(self) -> MediaProbe:
        return probe_media(
            source=self.source_argument,
            source_type=self.definition.source_type,
            ffprobe_path=self.ffprobe_path,
            timeout_seconds=self.probe_timeout_seconds,
            stop_event=self.stop_event,
            count_frames_fallback=(
                self.supports_frame_count_fallback
            ),
        )


class LocalFileStream(BaseVideoStream):
    """Finite local-file source."""

    def __init__(
        self,
        definition: StreamDefinition,
        *,
        base_directory: Path,
        ffprobe_path: str,
        probe_timeout_seconds: float,
        stop_event: threading.Event,
    ) -> None:
        if definition.source_type is not SourceType.LOCAL_FILE:
            raise StreamConfigurationError(
                "LocalFileStream requires source_type=local_file"
            )

        super().__init__(
            definition,
            ffprobe_path=ffprobe_path,
            probe_timeout_seconds=probe_timeout_seconds,
            stop_event=stop_event,
        )

        raw_path = Path(definition.source).expanduser()
        self.path = (
            raw_path
            if raw_path.is_absolute()
            else base_directory / raw_path
        ).resolve()

    @property
    def source_argument(self) -> str:
        return str(self.path)

    @property
    def supports_frame_count_fallback(self) -> bool:
        return True

    def inspect(self) -> StreamInspection:
        started_at = utc_now()
        start_clock = time.monotonic()
        self._set_state(StreamLifecycleState.VALIDATING)

        if self.stop_event.is_set():
            self._set_state(StreamLifecycleState.FAILED)
            raise StreamCancelledError(
                f"Inspection cancelled for camera {self.camera_id}"
            )

        if not self.path.exists():
            self._set_state(StreamLifecycleState.UNAVAILABLE)
            return _inspection(
                stream=self,
                available=False,
                readable=False,
                message=f"Local source does not exist: {self.path}",
                probe=None,
                started_at=started_at,
                start_clock=start_clock,
            )

        if not self.path.is_file():
            self._set_state(StreamLifecycleState.UNAVAILABLE)
            return _inspection(
                stream=self,
                available=False,
                readable=False,
                message=f"Local source is not a file: {self.path}",
                probe=None,
                started_at=started_at,
                start_clock=start_clock,
            )

        try:
            size = self.path.stat().st_size
        except OSError as exc:
            self._set_state(StreamLifecycleState.UNAVAILABLE)
            return _inspection(
                stream=self,
                available=False,
                readable=False,
                message=(
                    f"Unable to inspect local source "
                    f"{self.path}: {exc}"
                ),
                probe=None,
                started_at=started_at,
                start_clock=start_clock,
            )

        if size <= 0:
            self._set_state(StreamLifecycleState.UNAVAILABLE)
            return _inspection(
                stream=self,
                available=False,
                readable=False,
                message=f"Local source is empty: {self.path}",
                probe=None,
                started_at=started_at,
                start_clock=start_clock,
            )

        try:
            probe = self._probe()
        except StreamCancelledError:
            self._set_state(StreamLifecycleState.FAILED)
            raise
        except StreamProbeError as exc:
            self._set_state(StreamLifecycleState.FAILED)
            return _inspection(
                stream=self,
                available=True,
                readable=False,
                message=str(exc),
                probe=None,
                started_at=started_at,
                start_clock=start_clock,
            )

        self._set_state(StreamLifecycleState.READY)

        return _inspection(
            stream=self,
            available=True,
            readable=True,
            message="Local source is readable",
            probe=probe,
            started_at=started_at,
            start_clock=start_clock,
        )


class RTSPStream(BaseVideoStream):
    """
    RTSP source abstraction.

    This implementation provides validation and FFprobe inspection only.
    Production reconnection and sustained capture behavior must be validated
    with the target cameras and network before deployment.
    """

    def __init__(
        self,
        definition: StreamDefinition,
        *,
        ffprobe_path: str,
        probe_timeout_seconds: float,
        stop_event: threading.Event,
    ) -> None:
        if definition.source_type is not SourceType.RTSP:
            raise StreamConfigurationError(
                "RTSPStream requires source_type=rtsp"
            )

        _validate_rtsp_uri(definition.source)

        super().__init__(
            definition,
            ffprobe_path=ffprobe_path,
            probe_timeout_seconds=probe_timeout_seconds,
            stop_event=stop_event,
        )

    @property
    def source_argument(self) -> str:
        return self.definition.source

    @property
    def supports_frame_count_fallback(self) -> bool:
        # Counting every frame of a live stream may never terminate.
        return False

    def inspect(self) -> StreamInspection:
        started_at = utc_now()
        start_clock = time.monotonic()
        self._set_state(StreamLifecycleState.VALIDATING)

        if self.stop_event.is_set():
            self._set_state(StreamLifecycleState.FAILED)
            raise StreamCancelledError(
                f"Inspection cancelled for camera {self.camera_id}"
            )

        try:
            probe = self._probe()
        except StreamCancelledError:
            self._set_state(StreamLifecycleState.FAILED)
            raise
        except StreamProbeError as exc:
            self._set_state(StreamLifecycleState.UNAVAILABLE)
            return _inspection(
                stream=self,
                available=False,
                readable=False,
                message=str(exc),
                probe=None,
                started_at=started_at,
                start_clock=start_clock,
            )

        self._set_state(StreamLifecycleState.READY)

        return _inspection(
            stream=self,
            available=True,
            readable=True,
            message=(
                "RTSP source responded to FFprobe; sustained operation "
                "has not been validated"
            ),
            probe=probe,
            started_at=started_at,
            start_clock=start_clock,
        )


class StreamManager:
    """Thread-safe registry and factory for Phase 7 streams."""

    def __init__(
        self,
        *,
        base_directory: Path,
        ffprobe_path: str,
        probe_timeout_seconds: float,
        stop_event: threading.Event | None = None,
    ) -> None:
        self.base_directory = base_directory.resolve()
        self.ffprobe_path = ffprobe_path
        self.probe_timeout_seconds = float(
            probe_timeout_seconds
        )
        self.stop_event = stop_event or threading.Event()

        self._streams: dict[str, BaseVideoStream] = {}
        self._lock = threading.RLock()
        self._closed = False

    def register(
        self,
        definition: StreamDefinition,
    ) -> BaseVideoStream:
        """Register one unique camera and return its stream object."""

        with self._lock:
            if self._closed:
                raise StreamManagerError(
                    "Cannot register a stream after manager shutdown"
                )

            existing = self._streams.get(definition.camera_id)

            if existing is not None:
                if existing.definition == definition:
                    return existing

                raise StreamConfigurationError(
                    f"Duplicate camera_id with conflicting definitions: "
                    f"{definition.camera_id!r}"
                )

            if definition.source_type is SourceType.LOCAL_FILE:
                stream: BaseVideoStream = LocalFileStream(
                    definition,
                    base_directory=self.base_directory,
                    ffprobe_path=self.ffprobe_path,
                    probe_timeout_seconds=(
                        self.probe_timeout_seconds
                    ),
                    stop_event=self.stop_event,
                )
            elif definition.source_type is SourceType.RTSP:
                stream = RTSPStream(
                    definition,
                    ffprobe_path=self.ffprobe_path,
                    probe_timeout_seconds=(
                        self.probe_timeout_seconds
                    ),
                    stop_event=self.stop_event,
                )
            else:
                raise StreamConfigurationError(
                    f"No stream implementation exists for "
                    f"{definition.source_type.value!r}"
                )

            self._streams[definition.camera_id] = stream
            return stream

    def get(self, camera_id: str) -> BaseVideoStream:
        with self._lock:
            try:
                return self._streams[camera_id]
            except KeyError as exc:
                raise StreamManagerError(
                    f"Camera is not registered: {camera_id!r}"
                ) from exc

    def registered_camera_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._streams)

    def inspect(
        self,
        camera_id: str,
    ) -> StreamInspection:
        return self.get(camera_id).inspect()

    def close(self) -> None:
        """Signal shutdown and release all registered streams."""

        with self._lock:
            if self._closed:
                return

            self._closed = True
            self.stop_event.set()
            streams = tuple(self._streams.values())

        for stream in streams:
            stream.close()

    def __enter__(self) -> "StreamManager":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        traceback: object,
    ) -> None:
        self.close()


def probe_media(
    *,
    source: str,
    source_type: SourceType,
    ffprobe_path: str,
    timeout_seconds: float,
    stop_event: threading.Event,
    count_frames_fallback: bool,
) -> MediaProbe:
    """
    Probe media and optionally count decoded frames.

    The fallback uses ffprobe ``-count_frames`` and ``nb_read_frames``. It does
    not derive or estimate the count from FPS and duration.
    """

    safe_source = redact_source(source)
    command = _base_probe_command(
        ffprobe_path=ffprobe_path,
        source_type=source_type,
        timeout_seconds=timeout_seconds,
    )
    command.extend(
        [
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            source,
        ]
    )

    payload = _run_ffprobe_json(
        command=command,
        source=source,
        safe_source=safe_source,
        timeout_seconds=timeout_seconds,
        stop_event=stop_event,
    )

    streams = payload.get("streams")
    format_data = payload.get("format")

    if not isinstance(streams, list):
        raise StreamProbeError(
            f"FFprobe returned no stream list for {safe_source}"
        )

    video_stream = next(
        (
            item
            for item in streams
            if isinstance(item, Mapping)
            and item.get("codec_type") == "video"
        ),
        None,
    )

    if video_stream is None:
        raise StreamProbeError(
            f"No video stream was found in {safe_source}"
        )

    width = _positive_integer(video_stream.get("width"))
    height = _positive_integer(video_stream.get("height"))

    if width is None or height is None:
        raise StreamProbeError(
            f"FFprobe returned an invalid video resolution for "
            f"{safe_source}"
        )

    frame_count = _positive_integer(
        video_stream.get("nb_frames")
    )
    frame_count_source = (
        "container_metadata"
        if frame_count is not None
        else None
    )

    if frame_count is None and count_frames_fallback:
        frame_count = _count_decoded_frames(
            source=source,
            source_type=source_type,
            ffprobe_path=ffprobe_path,
            timeout_seconds=timeout_seconds,
            stop_event=stop_event,
        )

        if frame_count is not None:
            frame_count_source = "ffprobe_count_frames"

    average_fps = _parse_frame_rate(
        video_stream.get("avg_frame_rate")
    )

    if average_fps is None:
        average_fps = _parse_frame_rate(
            video_stream.get("r_frame_rate")
        )

    format_mapping = (
        format_data
        if isinstance(format_data, Mapping)
        else {}
    )

    duration_seconds = _positive_float(
        video_stream.get("duration")
    )

    if duration_seconds is None:
        duration_seconds = _positive_float(
            format_mapping.get("duration")
        )

    size_bytes = _nonnegative_integer(
        format_mapping.get("size")
    )

    if source_type is SourceType.LOCAL_FILE:
        try:
            size_bytes = Path(source).stat().st_size
        except OSError:
            # FFprobe's value remains useful if a second stat fails.
            pass

    bit_rate_bps = _positive_integer(
        format_mapping.get("bit_rate")
    )

    if bit_rate_bps is None:
        bit_rate_bps = _positive_integer(
            video_stream.get("bit_rate")
        )

    codec_name = _optional_text(
        video_stream.get("codec_name")
    )
    pixel_format = _optional_text(
        video_stream.get("pix_fmt")
    )
    format_name = _optional_text(
        format_mapping.get("format_name")
    )

    has_audio = any(
        isinstance(item, Mapping)
        and item.get("codec_type") == "audio"
        for item in streams
    )

    return MediaProbe(
        width=width,
        height=height,
        codec_name=codec_name,
        pixel_format=pixel_format,
        average_fps=average_fps,
        duration_seconds=duration_seconds,
        frame_count=frame_count,
        frame_count_source=frame_count_source,
        format_name=format_name,
        size_bytes=size_bytes,
        bit_rate_bps=bit_rate_bps,
        has_audio=has_audio,
        probed_at_utc=utc_now(),
    )


def redact_source(source: str) -> str:
    """Remove credentials from URI-like sources before logging."""

    try:
        parsed = urlsplit(source)
    except ValueError:
        return _fallback_redact(source)

    if not parsed.scheme or "@" not in parsed.netloc:
        return source

    hostname = parsed.hostname

    if hostname is None:
        return _fallback_redact(source)

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    port = f":{parsed.port}" if parsed.port is not None else ""
    safe_netloc = f"<redacted>@{hostname}{port}"

    safe_parts = SplitResult(
        scheme=parsed.scheme,
        netloc=safe_netloc,
        path=parsed.path,
        query=parsed.query,
        fragment=parsed.fragment,
    )
    return urlunsplit(safe_parts)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inspection(
    *,
    stream: BaseVideoStream,
    available: bool,
    readable: bool,
    message: str,
    probe: MediaProbe | None,
    started_at: str,
    start_clock: float,
) -> StreamInspection:
    return StreamInspection(
        camera_id=stream.camera_id,
        source_type=stream.definition.source_type,
        safe_source=stream.safe_source,
        available=available,
        readable=readable,
        message=message,
        probe=probe,
        started_at_utc=started_at,
        completed_at_utc=utc_now(),
        elapsed_seconds=round(
            max(0.0, time.monotonic() - start_clock),
            6,
        ),
    )


def _base_probe_command(
    *,
    ffprobe_path: str,
    source_type: SourceType,
    timeout_seconds: float,
) -> list[str]:
    command = [
        ffprobe_path,
        "-v",
        "error",
    ]

    if source_type is SourceType.RTSP:
        timeout_microseconds = max(
            1,
            int(timeout_seconds * 1_000_000),
        )
        command.extend(
            [
                "-rtsp_transport",
                "tcp",
                "-rw_timeout",
                str(timeout_microseconds),
            ]
        )

    return command


def _count_decoded_frames(
    *,
    source: str,
    source_type: SourceType,
    ffprobe_path: str,
    timeout_seconds: float,
    stop_event: threading.Event,
) -> int | None:
    command = _base_probe_command(
        ffprobe_path=ffprobe_path,
        source_type=source_type,
        timeout_seconds=timeout_seconds,
    )
    command.extend(
        [
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            source,
        ]
    )

    payload = _run_ffprobe_json(
        command=command,
        source=source,
        safe_source=redact_source(source),
        timeout_seconds=timeout_seconds,
        stop_event=stop_event,
    )

    streams = payload.get("streams")

    if not isinstance(streams, list) or not streams:
        return None

    first = streams[0]

    if not isinstance(first, Mapping):
        return None

    return _positive_integer(first.get("nb_read_frames"))


def _run_ffprobe_json(
    *,
    command: list[str],
    source: str,
    safe_source: str,
    timeout_seconds: float,
    stop_event: threading.Event,
) -> dict[str, Any]:
    if stop_event.is_set():
        raise StreamCancelledError(
            f"FFprobe cancelled before starting for {safe_source}"
        )

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
    except FileNotFoundError as exc:
        raise StreamProbeError(
            f"FFprobe executable was not found: "
            f"{command[0]!r}"
        ) from exc
    except OSError as exc:
        raise StreamProbeError(
            f"Unable to start FFprobe for {safe_source}: {exc}"
        ) from exc

    deadline = time.monotonic() + timeout_seconds

    try:
        while True:
            remaining = deadline - time.monotonic()

            if remaining <= 0:
                _terminate_process(process)
                raise StreamProbeError(
                    f"FFprobe timed out after "
                    f"{timeout_seconds:.3f} seconds for {safe_source}"
                )

            if stop_event.is_set():
                _terminate_process(process)
                raise StreamCancelledError(
                    f"FFprobe cancelled for {safe_source}"
                )

            try:
                stdout, stderr = process.communicate(
                    timeout=min(0.25, remaining)
                )
                break
            except subprocess.TimeoutExpired:
                continue
    finally:
        if process.poll() is None:
            _terminate_process(process)

    if process.returncode != 0:
        message = _clean_process_message(
            stderr,
            actual_source=source,
            safe_source=safe_source,
        )
        raise StreamProbeError(
            f"FFprobe failed for {safe_source} with exit code "
            f"{process.returncode}: {message}"
        )

    if not stdout.strip():
        raise StreamProbeError(
            f"FFprobe returned no JSON for {safe_source}"
        )

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise StreamProbeError(
            f"FFprobe returned invalid JSON for {safe_source}: "
            f"{exc}"
        ) from exc

    if not isinstance(payload, dict):
        raise StreamProbeError(
            f"FFprobe returned an unexpected JSON type for "
            f"{safe_source}"
        )

    return payload


def _terminate_process(
    process: subprocess.Popen[str],
) -> None:
    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=3.0)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=3.0)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _clean_process_message(
    value: str,
    *,
    actual_source: str,
    safe_source: str,
) -> str:
    message = value.strip() or "no FFprobe diagnostic was provided"

    if actual_source != safe_source:
        message = message.replace(actual_source, safe_source)

    # Keep reports readable and bounded.
    if len(message) > 2000:
        message = message[-2000:]

    return message


def _parse_frame_rate(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.upper() == "N/A":
        return None

    if "/" in text:
        numerator_text, denominator_text = text.split("/", 1)

        try:
            numerator = float(numerator_text)
            denominator = float(denominator_text)
        except ValueError:
            return None

        if denominator == 0:
            return None

        result = numerator / denominator
    else:
        try:
            result = float(text)
        except ValueError:
            return None

    if not math.isfinite(result) or result <= 0:
        return None

    return result


def _positive_integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        result = int(str(value).strip())
    except (TypeError, ValueError):
        return None

    return result if result > 0 else None


def _nonnegative_integer(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        result = int(str(value).strip())
    except (TypeError, ValueError):
        return None

    return result if result >= 0 else None


def _positive_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None

    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None

    if not math.isfinite(result) or result <= 0:
        return None

    return result


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()

    if not text or text.upper() == "N/A":
        return None

    return text


def _validate_rtsp_uri(value: str) -> None:
    try:
        parsed = urlsplit(value)
    except ValueError as exc:
        raise StreamConfigurationError(
            "RTSP source is not a valid URI"
        ) from exc

    if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
        raise StreamConfigurationError(
            "RTSP source must use the rtsp:// or rtsps:// scheme"
        )

    if not parsed.hostname:
        raise StreamConfigurationError(
            "RTSP source must include a hostname"
        )


def _fallback_redact(value: str) -> str:
    return re.sub(
        r"(?i)(rtsps?://)[^/@\s]+@",
        r"\1<redacted>@",
        value,
    )


__all__ = [
    "BaseVideoStream",
    "LocalFileStream",
    "MediaProbe",
    "RTSPStream",
    "SourceType",
    "StreamCancelledError",
    "StreamConfigurationError",
    "StreamDefinition",
    "StreamInspection",
    "StreamLifecycleState",
    "StreamManager",
    "StreamManagerError",
    "StreamProbeError",
    "probe_media",
    "redact_source",
    "utc_now",
]