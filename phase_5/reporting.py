"""Atomic machine-readable and human-readable Phase 5 reporting."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping


class ReportingError(RuntimeError):
    """Raised when reports cannot be written safely."""


class ReportWriter:
    """Write JSON and text reports atomically."""

    def __init__(
        self,
        json_path: Path,
        text_path: Path,
    ) -> None:
        self.json_path = (
            json_path.expanduser().resolve()
        )

        self.text_path = (
            text_path.expanduser().resolve()
        )

    def write(
        self,
        report: Mapping[str, Any],
    ) -> None:
        """Publish both reports after temporary files complete."""

        if self.json_path == self.text_path:
            raise ReportingError(
                "JSON and text report paths "
                "must be different"
            )

        json_temp = self.json_path.with_suffix(
            self.json_path.suffix + ".tmp"
        )

        text_temp = self.text_path.with_suffix(
            self.text_path.suffix + ".tmp"
        )

        try:
            self.json_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.text_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            with json_temp.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                json.dump(
                    report,
                    stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                )

                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())

            text = render_text_report(report)

            with text_temp.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                stream.write(text)

                if not text.endswith("\n"):
                    stream.write("\n")

                stream.flush()
                os.fsync(stream.fileno())

            os.replace(
                json_temp,
                self.json_path,
            )

            os.replace(
                text_temp,
                self.text_path,
            )

        except (
            OSError,
            TypeError,
            ValueError,
        ) as exc:
            raise ReportingError(
                "Unable to write Phase 5 reports: "
                f"{exc}"
            ) from exc


def render_text_report(
    report: Mapping[str, Any],
) -> str:
    """Render an engineering-manager summary."""

    storage = _mapping(
        report.get("storage_metrics")
    )

    decisions = _mapping(
        report.get("frame_decisions")
    )

    source = _mapping(
        report.get("source_video")
    )

    compressed = _mapping(
        report.get("compressed_video")
    )

    differences = _mapping(
        report.get("integrity_metrics")
    )

    performance = _mapping(
        report.get("performance_metrics")
    )

    quality = _mapping(
        report.get("quality_analysis")
    )

    validation = _mapping(
        report.get("validation")
    )

    lines = [
        "PHASE 5 - COMPRESSION VALIDATION REPORT",
        "=" * 43,
        f"Status: {report.get('status', 'unknown')}",
        "",
        "STORAGE",
        "Original bytes: "
        f"{_value(storage.get('original_bytes'))}",
        "Compressed bytes: "
        f"{_value(storage.get('compressed_bytes'))}",
        "Bytes saved: "
        f"{_value(storage.get('bytes_saved'))}",
        "Compression ratio: "
        f"{_value(storage.get('compression_ratio'), 'x')}",
        "Storage reduction: "
        f"{_value(storage.get('storage_reduction_percent'), '%')}",
        "",
        "FRAME DECISIONS",
        "Decision records: "
        f"{_value(decisions.get('total_decision_records'))}",
        "Retained decisions: "
        f"{_value(decisions.get('retained_frames'))}",
        "Dropped decisions: "
        f"{_value(decisions.get('dropped_frames'))}",
        "Retention ratio: "
        f"{_value(decisions.get('retention_ratio'))}",
        "Malformed records: "
        f"{_value(decisions.get('malformed_records'))}",
        "",
        "VIDEO COMPARISON",
        f"Source: {_media_line(source)}",
        f"Compressed: {_media_line(compressed)}",
        "Duration difference: "
        f"{_value(differences.get('duration_difference_seconds'), ' s')}",
        "FPS difference: "
        f"{_value(differences.get('fps_difference'))}",
        "Bitrate reduction: "
        f"{_value(differences.get('bitrate_reduction_percent'), '%')}",
        "",
        "PERFORMANCE",
        "Phase 4 encoding duration: "
        f"{_value(performance.get('phase4_encoding_duration_seconds'), ' s')}",
        "Phase 5 analysis duration: "
        f"{_value(performance.get('phase5_analysis_duration_seconds'), ' s')}",
        "",
        "QUALITY METRICS",
    ]

    raw_quality_metrics = quality.get("metrics")

    if (
        isinstance(raw_quality_metrics, list)
        and raw_quality_metrics
    ):
        for item in raw_quality_metrics:
            metric = _mapping(item)

            name = str(
                metric.get("name", "unknown")
            ).upper()

            status = str(
                metric.get("status", "unknown")
            )

            if status == "completed":
                if metric.get("infinite"):
                    result = "infinite"
                else:
                    result = _value(
                        metric.get("value")
                    )

                lines.append(
                    f"{name}: {result} ({status})"
                )
            else:
                lines.append(
                    f"{name}: {status} - "
                    f"{metric.get('reason', 'no reason given')}"
                )
    else:
        lines.append(
            "No quality metrics were requested."
        )

    warnings = validation.get("warnings")

    lines.extend([
        "",
        "VALIDATION WARNINGS",
    ])

    if isinstance(warnings, list) and warnings:
        lines.extend(
            f"- {warning}"
            for warning in warnings
        )
    else:
        lines.append("None")

    limitations = report.get("limitations")

    lines.extend([
        "",
        "LIMITATIONS",
    ])

    if (
        isinstance(limitations, list)
        and limitations
    ):
        lines.extend(
            f"- {limitation}"
            for limitation in limitations
        )
    else:
        lines.append(
            "None identified by this run."
        )

    return "\n".join(lines) + "\n"


def _media_line(
    media: Mapping[str, Any],
) -> str:
    """Render one compact media-information line."""

    return (
        f"{_value(media.get('width'))}"
        f"x{_value(media.get('height'))}, "
        f"fps={_value(media.get('average_fps'))}, "
        "duration="
        f"{_value(media.get('duration_seconds'), ' s')}, "
        f"codec={_value(media.get('codec'))}, "
        "bitrate="
        f"{_value(media.get('bitrate_bps'), ' bps')}"
    )


def _mapping(
    value: Any,
) -> Mapping[str, Any]:
    """Return a mapping or an empty mapping."""

    if isinstance(value, Mapping):
        return value

    return {}


def _value(
    value: Any,
    suffix: str = "",
) -> str:
    """Format an optional report value."""

    if value is None:
        return "unavailable"

    return f"{value}{suffix}"