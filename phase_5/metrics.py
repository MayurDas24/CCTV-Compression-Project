"""Pure end-to-end metric calculations for Phase 5."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from validation import (
    DecisionStatistics,
    MediaInformation,
    Phase4Summary,
)


BYTES_PER_MEGABYTE = 1_000_000
BYTES_PER_MEBIBYTE = 1024 * 1024


@dataclass(frozen=True, slots=True)
class CalculatedMetrics:
    """Measured metrics grouped for the final report."""

    storage: dict[str, Any]
    frame_decisions: dict[str, Any]
    source_video: dict[str, Any]
    compressed_video: dict[str, Any]
    differences: dict[str, Any]
    performance: dict[str, Any]


def calculate_metrics(
    source: MediaInformation,
    compressed: MediaInformation,
    decisions: DecisionStatistics,
    phase4: Phase4Summary,
) -> CalculatedMetrics:
    """Calculate reliable metrics without estimates."""

    bytes_saved = (
        source.size_bytes
        - compressed.size_bytes
    )
    compression_ratio = _divide(
        source.size_bytes,
        compressed.size_bytes,
    )
    storage_reduction = _reduction_percent(
        source.size_bytes,
        compressed.size_bytes,
    )

    bitrate_reduction = (
        _reduction_percent(
            source.bitrate_bps,
            compressed.bitrate_bps,
        )
        if source.bitrate_bps is not None
        and compressed.bitrate_bps is not None
        else None
    )

    fps_difference = _difference(
        compressed.average_fps,
        source.average_fps,
    )
    duration_difference = _difference(
        compressed.duration_seconds,
        source.duration_seconds,
    )

    frame_count_difference = (
        compressed.frame_count
        - source.frame_count
        if source.frame_count is not None
        and compressed.frame_count is not None
        else None
    )

    encoding_duration = (
        phase4.encoding_duration_seconds
    )

    encoding_throughput = (
        source.frame_count / encoding_duration
        if source.frame_count is not None
        and encoding_duration is not None
        and encoding_duration > 0.0
        else None
    )

    return CalculatedMetrics(
        storage={
            "original_bytes": source.size_bytes,
            "compressed_bytes": compressed.size_bytes,
            "bytes_saved": bytes_saved,
            "original_megabytes": _round(
                source.size_bytes
                / BYTES_PER_MEGABYTE
            ),
            "compressed_megabytes": _round(
                compressed.size_bytes
                / BYTES_PER_MEGABYTE
            ),
            "original_mebibytes": _round(
                source.size_bytes
                / BYTES_PER_MEBIBYTE
            ),
            "compressed_mebibytes": _round(
                compressed.size_bytes
                / BYTES_PER_MEBIBYTE
            ),
            "compression_ratio": _round(
                compression_ratio
            ),
            "storage_reduction_percent": _round(
                storage_reduction
            ),
        },
        frame_decisions={
            "total_decision_records": (
                decisions.total_decision_records
            ),
            "retained_frames": (
                decisions.retained_frames
            ),
            "dropped_frames": (
                decisions.dropped_frames
            ),
            "retention_ratio": _round(
                decisions.retention_ratio
            ),
            "drop_ratio": _round(
                decisions.drop_ratio
            ),
            "malformed_records": (
                decisions.malformed_records
            ),
            "first_frame_index": (
                decisions.first_frame_index
            ),
            "last_frame_index": (
                decisions.last_frame_index
            ),
            "minimum_timestamp_seconds": _round(
                decisions.minimum_timestamp_seconds
            ),
            "maximum_timestamp_seconds": _round(
                decisions.maximum_timestamp_seconds
            ),
            "priority_distribution": (
                decisions.priority_distribution
            ),
            "reason_distribution": (
                decisions.reason_distribution
            ),
        },
        source_video=_media_dict(source),
        compressed_video=_media_dict(compressed),
        differences={
            "resolution_changed": (
                source.width != compressed.width
                or source.height != compressed.height
            ),
            "fps_difference": _round(
                fps_difference
            ),
            "absolute_fps_difference": _round(
                _absolute(fps_difference)
            ),
            "duration_difference_seconds": _round(
                duration_difference
            ),
            "absolute_duration_difference_seconds": (
                _round(
                    _absolute(duration_difference)
                )
            ),
            "bitrate_reduction_percent": _round(
                bitrate_reduction
            ),
            "codec_changed": (
                source.codec != compressed.codec
            ),
            "pixel_format_changed": (
                source.pixel_format
                != compressed.pixel_format
            ),
            "audio_presence_changed": (
                source.has_audio
                != compressed.has_audio
            ),
            "audio_codec_changed": (
                source.audio_codec
                != compressed.audio_codec
            ),
            "frame_count_difference": (
                frame_count_difference
            ),
        },
        performance={
            "phase4_encoding_duration_seconds": (
                _round(encoding_duration)
            ),
            "phase4_encoding_throughput_source_frames_per_second": (
                _round(encoding_throughput)
            ),
        },
    )


def _media_dict(
    media: MediaInformation,
) -> dict[str, Any]:
    return {
        "path": str(media.path),
        "size_bytes": media.size_bytes,
        "width": media.width,
        "height": media.height,
        "average_fps": _round(
            media.average_fps
        ),
        "nominal_fps": _round(
            media.nominal_fps
        ),
        "duration_seconds": _round(
            media.duration_seconds
        ),
        "frame_count": media.frame_count,
        "frame_count_method": (
            media.frame_count_method
        ),
        "codec": media.codec,
        "pixel_format": media.pixel_format,
        "bitrate_bps": media.bitrate_bps,
        "has_audio": media.has_audio,
        "audio_codec": media.audio_codec,
    }


def _divide(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def _reduction_percent(
    original: int | float,
    reduced: int | float,
) -> float | None:
    if original == 0:
        return None

    return (
        (original - reduced)
        / original
    ) * 100.0


def _difference(
    current: float | None,
    baseline: float | None,
) -> float | None:
    if current is None or baseline is None:
        return None

    return current - baseline


def _absolute(
    value: float | None,
) -> float | None:
    if value is None:
        return None

    return abs(value)


def _round(
    value: float | None,
    digits: int = 6,
) -> float | None:
    if value is None:
        return None

    return round(value, digits)