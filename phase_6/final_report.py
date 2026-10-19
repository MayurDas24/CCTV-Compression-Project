"""Consolidated JSON and text report generation for Phase 6."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from integration import IntegrationResult


class FinalReportError(RuntimeError):
    """Raised when final reports cannot be published."""


def build_final_report(
    result: IntegrationResult,
    final_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a report without recalculating upstream metrics."""

    data = result.data
    phase4 = data.phase4
    phase5 = data.phase5

    phase4_storage = _mapping(
        phase4.get("storage")
    )
    phase4_compression = _mapping(
        phase4.get("compression")
    )
    phase5_quality = _mapping(
        phase5.get("quality_analysis")
    )
    phase5_integrity = _mapping(
        phase5.get("integrity_metrics")
    )
    phase4_performance = _mapping(
        phase4.get("performance")
    )
    phase5_performance = _mapping(
        phase5.get("performance_metrics")
    )

    limitations = phase5.get("limitations")
    if not isinstance(limitations, list):
        limitations = []

    limitations = list(limitations) + [
        (
            "Phase 6 consolidates and verifies upstream results; "
            "it does not rerun compression, objective quality "
            "analysis, or object detection."
        )
    ]

    return {
        "schema_version": "1.0",
        "phase": "phase_6_final_integration",
        "status": (
            "completed_with_warnings"
            if result.warnings
            else "completed"
        ),
        "pipeline_status": {
            "valid": result.valid,
            "phase3": "completed",
            "phase4": phase4.get("status"),
            "phase5": phase5.get("status"),
            "successful_checks": result.successful_checks,
            "warning_count": len(result.warnings),
            "error_count": len(result.errors),
        },
        "source_video": data.source_media.to_dict(),
        "phase3": data.phase3.to_dict(),
        "phase4": {
            "summary_path": str(
                data.paths.phase4_summary
            ),
            "compressed_video_path": str(
                data.paths.phase4_compressed_video
            ),
            "status": phase4.get("status"),
            "compression_configuration": dict(
                phase4_compression
            ),
        },
        "phase5": {
            "report_json_path": str(
                data.paths.phase5_report_json
            ),
            "report_text_path": str(
                data.paths.phase5_report_text
            ),
            "status": phase5.get("status"),
            "validation": phase5.get("validation"),
        },
        "compression_results": dict(
            phase4_storage
        ),
        "quality_results": dict(
            phase5_quality
        ),
        "integrity_results": {
            "measured_source": (
                data.source_media.to_dict()
            ),
            "measured_compressed": (
                data.compressed_media.to_dict()
            ),
            "phase5_integrity_metrics": dict(
                phase5_integrity
            ),
        },
        "performance_results": {
            "phase4": dict(phase4_performance),
            "phase5": dict(phase5_performance),
        },
        "consistency_checks": [
            check.to_dict()
            for check in result.checks
        ],
        "final_artifacts": dict(final_artifacts),
        "warnings": list(result.warnings),
        "errors": list(result.errors),
        "limitations": limitations,
    }


class FinalReportWriter:
    """Atomically publish Phase 6 JSON and text reports."""

    def __init__(
        self,
        json_path: Path,
        text_path: Path,
        *,
        overwrite: bool,
    ) -> None:
        self.json_path = json_path.expanduser().resolve()
        self.text_path = text_path.expanduser().resolve()
        self.overwrite = overwrite

    def write(
        self,
        report: Mapping[str, Any],
    ) -> None:
        if self.json_path == self.text_path:
            raise FinalReportError(
                "JSON and text report paths must differ"
            )

        if not self.overwrite:
            existing = [
                path
                for path in (
                    self.json_path,
                    self.text_path,
                )
                if path.exists()
            ]

            if existing:
                raise FinalReportError(
                    "Report exists and overwrite is disabled: "
                    + ", ".join(
                        str(path)
                        for path in existing
                    )
                )

        json_text = json.dumps(
            report,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n"

        text = render_text_report(report)

        _write_text_atomic(
            self.json_path,
            json_text,
        )
        _write_text_atomic(
            self.text_path,
            text,
        )


def render_text_report(
    report: Mapping[str, Any],
) -> str:
    pipeline = _mapping(
        report.get("pipeline_status")
    )
    source = _mapping(
        report.get("source_video")
    )
    integrity = _mapping(
        report.get("integrity_results")
    )
    compressed = _mapping(
        integrity.get("measured_compressed")
    )
    phase3 = _mapping(
        report.get("phase3")
    )
    compression = _mapping(
        report.get("compression_results")
    )
    phase4 = _mapping(
        report.get("phase4")
    )
    configuration = _mapping(
        phase4.get("compression_configuration")
    )
    performance = _mapping(
        report.get("performance_results")
    )
    phase4_performance = _mapping(
        performance.get("phase4")
    )
    phase5_performance = _mapping(
        performance.get("phase5")
    )
    quality = _mapping(
        report.get("quality_results")
    )
    metric_values = _quality_values(
        quality.get("metrics")
    )

    lines = [
        "CCTV COMPRESSION PIPELINE - FINAL DELIVERY REPORT",
        "=" * 49,
        f"Final status: {report.get('status', 'unknown')}",
        (
            "All critical validation checks passed: "
            f"{pipeline.get('valid', False)}"
        ),
        (
            "Successful checks: "
            f"{_value(pipeline.get('successful_checks'))}"
        ),
        "",
        "VIDEO INTEGRITY",
        f"Source: {_media_line(source)}",
        f"Compressed: {_media_line(compressed)}",
        "",
        "COMPRESSION RESULTS",
        (
            "Original size: "
            f"{_value(compression.get('original_bytes'))} bytes"
        ),
        (
            "Compressed size: "
            f"{_value(compression.get('compressed_bytes'))} bytes"
        ),
        (
            "Bytes saved: "
            f"{_value(compression.get('bytes_saved'))}"
        ),
        (
            "Storage reduction: "
            f"{_value(compression.get('storage_reduction_percent'))}%"
        ),
        (
            "Compression ratio: "
            f"{_value(compression.get('compression_ratio'))}x"
        ),
        (
            "Priority CRF strategy: "
            f"{_value(configuration.get('priority_crf'))}"
        ),
        "",
        "PHASE 3 DECISIONS",
        (
            "Decision records: "
            f"{_value(phase3.get('total_decision_records'))}"
        ),
        (
            "Retained: "
            f"{_value(phase3.get('retained_decisions'))}"
        ),
        (
            "Dropped: "
            f"{_value(phase3.get('dropped_decisions'))}"
        ),
        (
            "Priorities: "
            f"{_value(phase3.get('priority_distribution'))}"
        ),
        "",
        "OBJECTIVE QUALITY",
        f"PSNR: {_value(metric_values.get('psnr'))}",
        f"SSIM: {_value(metric_values.get('ssim'))}",
        f"VMAF: {_value(metric_values.get('vmaf'))}",
        "",
        "PERFORMANCE",
        (
            "Phase 4 encoding time: "
            f"{_value(phase4_performance.get('encoding_duration_seconds'))} "
            "seconds"
        ),
        (
            "Phase 5 analysis time: "
            f"{_value(phase5_performance.get('phase5_analysis_duration_seconds'))} "
            "seconds"
        ),
        "",
        "WARNINGS",
    ]

    warnings = report.get("warnings")

    lines.extend(
        [
            f"- {item}"
            for item in warnings
        ]
        if isinstance(warnings, list) and warnings
        else ["None"]
    )

    lines.extend(
        [
            "",
            "LIMITATIONS",
        ]
    )

    limitations = report.get("limitations")

    lines.extend(
        [
            f"- {item}"
            for item in limitations
        ]
        if isinstance(limitations, list) and limitations
        else ["None"]
    )

    return "\n".join(lines) + "\n"


def _quality_values(value: Any) -> dict[str, Any]:
    result: dict[str, Any] = {}

    if not isinstance(value, list):
        return result

    for item in value:
        if not isinstance(item, Mapping):
            continue

        name = item.get("name")

        if not isinstance(name, str):
            continue

        if item.get("status") == "completed":
            result[name] = (
                "infinite"
                if item.get("infinite")
                else item.get("value")
            )
        else:
            result[name] = (
                "unavailable "
                f"({item.get('reason') or item.get('status')})"
            )

    return result


def _media_line(
    media: Mapping[str, Any],
) -> str:
    return (
        f"{_value(media.get('width'))}x"
        f"{_value(media.get('height'))}, "
        f"fps={_value(media.get('average_fps'))}, "
        f"duration={_value(media.get('duration_seconds'))} seconds, "
        f"frames={_value(media.get('frame_count'))}, "
        f"codec={_value(media.get('codec'))}"
    )


def _write_text_atomic(
    path: Path,
    content: str,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    handle, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(name)

    try:
        with os.fdopen(
            handle,
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.close(handle)
        except OSError:
            pass

        temporary.unlink(missing_ok=True)

        raise FinalReportError(
            f"Unable to write report {path}: {exc}"
        ) from exc


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _value(value: Any) -> str:
    return (
        "unavailable"
        if value is None
        else str(value)
    )