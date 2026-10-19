"""Reliable JSONL and summary writing for Phase 3."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, TextIO


class DecisionWriterError(RuntimeError):
    """Raised when Phase 3 output cannot be written safely."""


class DecisionWriter:
    """Write decisions incrementally and publish completed outputs.

    Decision records are streamed to a temporary sibling file. The
    public output files are replaced only during ``finalize``. Therefore,
    an interrupted run does not make a partially written JSONL file look
    like a completed result.
    """

    def __init__(
        self,
        decisions_path: Path,
        summary_path: Path,
    ) -> None:
        self.decisions_path = decisions_path
        self.summary_path = summary_path

        self._decisions_temp = decisions_path.with_suffix(
            decisions_path.suffix + ".tmp"
        )

        self._summary_temp = summary_path.with_suffix(
            summary_path.suffix + ".tmp"
        )

        self._stream: TextIO | None = None
        self._finalized = False

    def open(self) -> None:
        """Create output directories and open the JSONL temp file."""

        try:
            self.decisions_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self.summary_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            self._stream = self._decisions_temp.open(
                "w",
                encoding="utf-8",
                newline="\n",
            )
        except OSError as exc:
            raise DecisionWriterError(
                f"Unable to initialize outputs: {exc}"
            ) from exc

    def write_decision(
        self,
        record: Mapping[str, Any],
    ) -> None:
        """Serialize one decision as one compact JSON line."""

        if self._stream is None or self._stream.closed:
            raise DecisionWriterError(
                "Decision writer is not open"
            )

        try:
            safe_record = _json_safe(record)

            serialized = json.dumps(
                safe_record,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=False,
            )

            self._stream.write(serialized + "\n")
        except (OSError, TypeError, ValueError) as exc:
            raise DecisionWriterError(
                f"Unable to write decision record: {exc}"
            ) from exc

    def finalize(
        self,
        summary: Mapping[str, Any],
    ) -> None:
        """Flush and publish the decision and summary files."""

        if self._stream is None or self._stream.closed:
            raise DecisionWriterError(
                "Decision writer is not open"
            )

        try:
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()

            with self._summary_temp.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as summary_stream:
                json.dump(
                    _json_safe(summary),
                    summary_stream,
                    ensure_ascii=False,
                    allow_nan=False,
                    indent=2,
                    sort_keys=False,
                )

                summary_stream.write("\n")
                summary_stream.flush()
                os.fsync(summary_stream.fileno())

            os.replace(
                self._decisions_temp,
                self.decisions_path,
            )

            os.replace(
                self._summary_temp,
                self.summary_path,
            )

            self._finalized = True
        except (OSError, TypeError, ValueError) as exc:
            raise DecisionWriterError(
                f"Unable to finalize outputs: {exc}"
            ) from exc

    def abort(self) -> None:
        """Close an incomplete decision stream."""

        if (
            self._stream is not None
            and not self._stream.closed
        ):
            try:
                self._stream.close()
            except OSError:
                pass

    def __enter__(self) -> "DecisionWriter":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Any,
        exc: Any,
        traceback: Any,
    ) -> None:
        if not self._finalized:
            self.abort()


class SummaryAccumulator:
    """Calculate summary statistics using constant memory."""

    def __init__(self) -> None:
        self.total_frames = 0
        self.frames_retained = 0
        self.frames_dropped = 0

        self.priority_distribution: Counter[str] = Counter()
        self.reason_distribution: Counter[str] = Counter()
        self.class_distribution: Counter[str] = Counter()

        self.importance_sum = 0.0
        self.minimum_importance = math.inf
        self.maximum_importance = -math.inf
        self.maximum_consecutive_drops = 0

    def update(
        self,
        record: Mapping[str, Any],
    ) -> None:
        """Update statistics from the record sent to the writer."""

        score = float(record["importance_score"])
        decision = str(record["decision"])
        priority = str(record["priority"])
        reason = str(record["reason"])

        consecutive_drops = int(
            record["consecutive_drops"]
        )

        self.total_frames += 1

        if decision == "retain":
            self.frames_retained += 1
        else:
            self.frames_dropped += 1

        self.priority_distribution[priority] += 1
        self.reason_distribution[reason] += 1

        # Count the number of analyzed frames containing each class.
        for class_name in set(
            record.get("classes_present", [])
        ):
            self.class_distribution[
                str(class_name)
            ] += 1

        self.importance_sum += score

        self.minimum_importance = min(
            self.minimum_importance,
            score,
        )

        self.maximum_importance = max(
            self.maximum_importance,
            score,
        )

        self.maximum_consecutive_drops = max(
            self.maximum_consecutive_drops,
            consecutive_drops,
        )

    def build_summary(
        self,
        *,
        input_file: Path,
        total_input_lines: int,
        malformed_records_skipped: int,
        metadata_warnings: int,
        configured_max_consecutive_drops: int,
        processing_duration_seconds: float,
    ) -> dict[str, Any]:
        """Build the deterministic Phase 3 summary object."""

        total = self.total_frames

        return {
            "schema_version": "1.0",
            "phase": "phase_3_frame_decision",
            "status": (
                "completed_with_warnings"
                if (
                    malformed_records_skipped
                    or metadata_warnings
                )
                else "completed"
            ),
            "input_metadata_file": str(input_file),
            "total_input_lines": total_input_lines,
            "malformed_records_skipped": (
                malformed_records_skipped
            ),
            "metadata_warnings": metadata_warnings,
            "total_frames_analyzed": total,
            "frames_retained": self.frames_retained,
            "frames_dropped": self.frames_dropped,
            "retention_ratio": round(
                self.frames_retained / total,
                6,
            ),
            "drop_ratio": round(
                self.frames_dropped / total,
                6,
            ),
            "priority_distribution": dict(
                sorted(
                    self.priority_distribution.items()
                )
            ),
            "reason_distribution": dict(
                sorted(
                    self.reason_distribution.items()
                )
            ),
            "class_distribution": dict(
                sorted(
                    self.class_distribution.items()
                )
            ),
            "average_importance_score": round(
                self.importance_sum / total,
                6,
            ),
            "maximum_importance_score": round(
                self.maximum_importance,
                6,
            ),
            "minimum_importance_score": round(
                self.minimum_importance,
                6,
            ),
            "maximum_consecutive_drops": (
                self.maximum_consecutive_drops
            ),
            "configured_max_consecutive_drops": (
                configured_max_consecutive_drops
            ),
            "processing_duration_seconds": round(
                processing_duration_seconds,
                6,
            ),
        }


def _json_safe(value: Any) -> Any:
    """Recursively convert supported values into strict JSON data."""

    if is_dataclass(value) and not isinstance(value, type):
        return _json_safe(asdict(value))

    if isinstance(value, Enum):
        return value.value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item)
            for item in value
        ]

    if isinstance(value, (set, frozenset)):
        return [
            _json_safe(item)
            for item in sorted(value, key=str)
        ]

    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None

    if value is None or isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    raise TypeError(
        f"Unsupported JSON value: {type(value).__name__}"
    )