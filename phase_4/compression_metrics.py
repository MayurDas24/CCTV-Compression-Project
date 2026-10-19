"""Measured compression and temporal-preservation metrics for Phase 4."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from frame_selector import (
    DecisionStatistics,
    Priority,
    TemporalPlan,
)
from video_encoder import EncodingConfig, EncodingResult


BYTES_PER_MEGABYTE = 1_000_000
BYTES_PER_MEBIBYTE = 1024 * 1024


def build_compression_summary(
    *,
    source_video: Path,
    decision_file: Path,
    output_video: Path,
    decision_statistics: DecisionStatistics,
    temporal_plan: TemporalPlan,
    encoding_result: EncodingResult,
    encoding_config: EncodingConfig,
    malformed_records_are_fatal: bool,
) -> dict[str, Any]:
    """Build a Phase 5-compatible summary using measured values."""

    source = encoding_result.source
    output = encoding_result.output

    original_bytes = source_video.stat().st_size
    compressed_bytes = output_video.stat().st_size
    bytes_saved = original_bytes - compressed_bytes

    compression_ratio = _divide(
        original_bytes,
        compressed_bytes,
    )
    storage_reduction = _reduction(
        original_bytes,
        compressed_bytes,
    )

    bitrate_reduction = (
        _reduction(
            source.bitrate_bps,
            output.bitrate_bps,
        )
        if source.bitrate_bps is not None
        and output.bitrate_bps is not None
        else None
    )

    source_frames = source.frame_count
    output_frames = output.frame_count
    physically_removed = (
    source_frames - output_frames
    if source_frames is not None
    and output_frames is not None
    else None
)

    segment_distribution = Counter(
        segment.priority.value
        for segment in encoding_result.segments
    )

    crf_mapping = {
        priority.value: encoding_config.priority_crf[priority]
        for priority in Priority
    }

    return {
        "schema_version": "1.0",
        "phase": "phase_4_intelligent_compression",
        "status": (
            "completed_with_warnings"
            if decision_statistics.malformed_records
            else "completed"
        ),
        "inputs": {
            "video": str(source_video),
            "decision_file": str(decision_file),
        },
        "outputs": {
            "video": str(output_video),
        },
        "video": {
            "source": _video_dict(source),
            "output": _video_dict(output),
            "timing_strategy": (
                "all_frames_temporal_quality_segments"
            ),
            "source_fps_preserved": (
                abs(
                    source.average_fps
                    - output.average_fps
                )
                <= 0.000001
            ),
            "source_duration_preserved": (
                abs(
                    source.duration_seconds
                    - output.duration_seconds
                )
                <= 0.000001
            ),
        },
        "decisions": {
            "frames_analyzed": (
                decision_statistics.valid_records
            ),
            "frames_retained": (
                decision_statistics.retain_decisions
            ),
            "frames_dropped": (
                decision_statistics.drop_decisions
            ),
            "retention_ratio": _round(
                _divide(
                    decision_statistics.retain_decisions,
                    decision_statistics.valid_records,
                )
            ),
            "drop_ratio": _round(
                _divide(
                    decision_statistics.drop_decisions,
                    decision_statistics.valid_records,
                )
            ),
            "malformed_records_skipped": (
                decision_statistics.malformed_records
            ),
            "malformed_records_are_fatal": (
                malformed_records_are_fatal
            ),
            "gap_count": decision_statistics.gap_count,
            "missing_frame_indices_between_decisions": (
                decision_statistics
                .missing_frame_indices_between_decisions
            ),
            "priority_distribution": (
                decision_statistics.priority_distribution
            ),
            "interpretation": (
                "sampled_temporal_importance_signals"
            ),
        },
        "temporal_plan": {
            "quality_segment_count": (
                temporal_plan.quality_segment_count
            ),
            "covered_source_frames": (
                temporal_plan.covered_frame_count
            ),
            "source_frame_count": (
                temporal_plan.source_frame_count
            ),
            "all_source_frames_covered": (
                temporal_plan.covered_frame_count
                == temporal_plan.source_frame_count
            ),
            "segments_by_priority": dict(
                sorted(segment_distribution.items())
            ),
        },
        "frame_reduction": {
            "source_frame_count": source_frames,
            "output_frame_count": output_frames,
            "actual_physically_removed_frames": (
                physically_removed
            ),
            "actual_retained_frame_ratio": _round(
                _divide(output_frames, source_frames)
            ),
            "actual_removed_frame_ratio": _round(
                _divide(physically_removed, source_frames)
            ),
            "physically_drop_frames_enabled": False,
        },
        "compression": {
            "codec": encoding_config.codec,
            "preset": encoding_config.preset,
            "pixel_format": encoding_config.pixel_format,
            "container": encoding_config.container,
            "priority_crf": crf_mapping,
            "audio_enabled": encoding_config.audio_enabled,
            "audio_mode": (
                encoding_config.audio_mode
                if encoding_config.audio_enabled
                else "disabled"
            ),
        },
        "storage": {
            "original_bytes": original_bytes,
            "compressed_bytes": compressed_bytes,
            "bytes_saved": bytes_saved,
            "original_megabytes": _round(
                original_bytes / BYTES_PER_MEGABYTE
            ),
            "compressed_megabytes": _round(
                compressed_bytes / BYTES_PER_MEGABYTE
            ),
            "original_mebibytes": _round(
                original_bytes / BYTES_PER_MEBIBYTE
            ),
            "compressed_mebibytes": _round(
                compressed_bytes / BYTES_PER_MEBIBYTE
            ),
            "compression_ratio": _round(compression_ratio),
            "storage_reduction_percent": _round(
                storage_reduction
            ),
            "bitrate_reduction_percent": _round(
                bitrate_reduction
            ),
        },
        "performance": {
            "encoding_duration_seconds": _round(
                encoding_result.encoding_duration_seconds
            ),
            "encoding_throughput_source_frames_per_second": (
                _round(
                    _divide(
                        source_frames,
                        encoding_result.encoding_duration_seconds,
                    )
                )
            ),
        },
        "validation": {
            "resolution_matches": (
                (source.width, source.height)
                == (output.width, output.height)
            ),
            "fps_difference": _round(
                output.average_fps - source.average_fps
            ),
            "absolute_fps_difference": _round(
                abs(
                    output.average_fps
                    - source.average_fps
                )
            ),
            "duration_difference_seconds": _round(
                output.duration_seconds
                - source.duration_seconds
            ),
            "absolute_duration_difference_seconds": _round(
                abs(
                    output.duration_seconds
                    - source.duration_seconds
                )
            ),
            "frame_count_difference": (
                output_frames - source_frames
            ),
            "all_source_frames_preserved": (
                output_frames == source_frames
            ),
        },
        "runtime": {
            "temporary_directory": (
                encoding_result.temporary_directory
            ),
        },
    }


def _video_dict(properties: Any) -> dict[str, Any]:
    return {
        "width": properties.width,
        "height": properties.height,
        "average_fps": _round(
            properties.average_fps
        ),
        "nominal_fps": _round(
            properties.nominal_fps
        ),
        "frame_count": properties.frame_count,
        "duration_seconds": _round(
            properties.duration_seconds
        ),
        "bitrate_bps": properties.bitrate_bps,
        "codec": properties.codec,
        "pixel_format": properties.pixel_format,
        "has_audio": properties.has_audio,
        "audio_codec": properties.audio_codec,
    }


def _divide(
    numerator: int | float,
    denominator: int | float,
) -> float | None:
    if denominator == 0:
        return None

    return numerator / denominator


def _reduction(
    original: int | float,
    compressed: int | float,
) -> float | None:
    if original == 0:
        return None

    return ((original - compressed) / original) * 100.0


def _round(
    value: float | None,
    digits: int = 6,
) -> float | None:
    return (
        round(value, digits)
        if value is not None
        else None
    )