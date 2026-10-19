"""Phase 3 decision loading and temporal quality interval construction."""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


LOGGER = logging.getLogger("FrameSelector")


class FrameSelectorError(RuntimeError):
    """Raised when Phase 3 decisions cannot define a safe timeline."""


class Priority(str, Enum):
    """Supported Phase 3 priority levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class FrameDecision:
    """Validated Phase 3 sampled-frame decision."""

    frame_index: int
    timestamp_seconds: float
    importance_score: float
    decision: str
    priority: Priority
    reason: str


@dataclass(frozen=True, slots=True)
class TemporalInterval:
    """A contiguous source-frame interval sharing one quality priority."""

    start_frame: int
    end_frame_exclusive: int
    start_seconds: float
    end_seconds: float
    priority: Priority
    decision_count: int

    @property
    def frame_count(self) -> int:
        return self.end_frame_exclusive - self.start_frame

    @property
    def duration_seconds(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass(frozen=True, slots=True)
class DecisionStatistics:
    """Measured properties of the Phase 3 decision file."""

    total_lines: int
    valid_records: int
    malformed_records: int
    retain_decisions: int
    drop_decisions: int
    duplicate_indices: int
    gap_count: int
    missing_frame_indices_between_decisions: int
    priority_distribution: dict[str, int]


@dataclass(frozen=True, slots=True)
class TemporalPlan:
    """Complete, gap-free quality plan for the source video."""

    intervals: tuple[TemporalInterval, ...]
    source_frame_count: int
    covered_frame_count: int
    quality_segment_count: int

    def interval_for_frame(self, frame_index: int) -> TemporalInterval:
        """Return the quality interval containing one physical source frame."""

        if not 0 <= frame_index < self.source_frame_count:
            raise IndexError(
                f"Frame index {frame_index} is outside 0.."
                f"{self.source_frame_count - 1}"
            )

        for interval in self.intervals:
            if frame_index < interval.end_frame_exclusive:
                return interval

        raise FrameSelectorError(
            f"Temporal plan unexpectedly has no interval for frame {frame_index}"
        )


class FrameSelector:
    """Interpret sampled Phase 3 decisions as temporal quality signals."""

    def __init__(
        self,
        decisions: tuple[FrameDecision, ...],
        statistics: DecisionStatistics,
    ) -> None:
        self.decisions = decisions
        self.statistics = statistics

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        fail_on_malformed: bool,
    ) -> "FrameSelector":
        """Stream, validate, sort, and index Phase 3 decisions."""

        decision_path = path.expanduser().resolve()

        if not decision_path.is_file():
            raise FrameSelectorError(
                f"Phase 3 decision file not found: {decision_path}"
            )

        records: list[FrameDecision] = []
        total_lines = 0
        malformed = 0

        try:
            with decision_path.open(
                "r",
                encoding="utf-8-sig",
                errors="strict",
            ) as stream:
                for line_number, line in enumerate(stream, start=1):
                    total_lines += 1
                    stripped = line.strip()

                    if not stripped:
                        continue

                    try:
                        raw = json.loads(stripped)
                        records.append(_parse_record(raw))
                    except (json.JSONDecodeError, ValueError) as exc:
                        if fail_on_malformed:
                            raise FrameSelectorError(
                                f"Invalid Phase 3 record at line "
                                f"{line_number}: {exc}"
                            ) from exc

                        malformed += 1
                        LOGGER.warning(
                            "Skipping malformed decision at line %d: %s",
                            line_number,
                            exc,
                        )
        except UnicodeError as exc:
            raise FrameSelectorError(
                f"Decision file is not valid UTF-8: {decision_path}"
            ) from exc
        except OSError as exc:
            raise FrameSelectorError(
                f"Unable to read decision file {decision_path}: {exc}"
            ) from exc

        if not records:
            raise FrameSelectorError(
                "Phase 3 decision file contains no valid decisions"
            )

        records.sort(key=lambda item: item.frame_index)

        duplicates = 0
        gaps = 0
        missing_indices = 0
        previous: FrameDecision | None = None

        for record in records:
            if previous is not None:
                if record.frame_index == previous.frame_index:
                    duplicates += 1
                elif record.frame_index > previous.frame_index + 1:
                    gaps += 1
                    missing_indices += (
                        record.frame_index - previous.frame_index - 1
                    )

                if record.timestamp_seconds <= previous.timestamp_seconds:
                    raise FrameSelectorError(
                        "Decision timestamps must increase with frame indices: "
                        f"frame {record.frame_index} has timestamp "
                        f"{record.timestamp_seconds}"
                    )

            previous = record

        if duplicates:
            raise FrameSelectorError(
                f"Detected {duplicates} duplicate Phase 3 frame indices"
            )

        decision_counts = Counter(item.decision for item in records)
        priority_counts = Counter(item.priority.value for item in records)

        statistics = DecisionStatistics(
            total_lines=total_lines,
            valid_records=len(records),
            malformed_records=malformed,
            retain_decisions=decision_counts["retain"],
            drop_decisions=decision_counts["drop"],
            duplicate_indices=duplicates,
            gap_count=gaps,
            missing_frame_indices_between_decisions=missing_indices,
            priority_distribution=dict(sorted(priority_counts.items())),
        )

        LOGGER.info(
            "Loaded %d sampled decisions: retain=%d drop=%d gaps=%d",
            statistics.valid_records,
            statistics.retain_decisions,
            statistics.drop_decisions,
            statistics.gap_count,
        )

        return cls(tuple(records), statistics)

    def build_temporal_plan(
        self,
        *,
        source_frame_count: int,
        source_fps: float,
        source_duration_seconds: float,
    ) -> TemporalPlan:
        """Build gap-free intervals covering every source frame exactly once."""

        if source_frame_count <= 0:
            raise FrameSelectorError("Source frame count must be positive")

        if not math.isfinite(source_fps) or source_fps <= 0.0:
            raise FrameSelectorError("Source FPS must be positive and finite")

        if (
            not math.isfinite(source_duration_seconds)
            or source_duration_seconds <= 0.0
        ):
            raise FrameSelectorError(
                "Source duration must be positive and finite"
            )

        if self.decisions[0].frame_index != 0:
            raise FrameSelectorError(
                "The first Phase 3 decision must describe source frame 0"
            )

        last_index = self.decisions[-1].frame_index

        if last_index >= source_frame_count:
            raise FrameSelectorError(
                f"Decision frame {last_index} is outside source range "
                f"0..{source_frame_count - 1}"
            )

        raw_intervals: list[TemporalInterval] = []

        for index, decision in enumerate(self.decisions):
            start_frame = decision.frame_index
            end_frame = (
                self.decisions[index + 1].frame_index
                if index + 1 < len(self.decisions)
                else source_frame_count
            )

            if end_frame <= start_frame:
                raise FrameSelectorError(
                    f"Invalid temporal interval {start_frame}..{end_frame}"
                )

            start_seconds = start_frame / source_fps
            end_seconds = (
                end_frame / source_fps
                if end_frame < source_frame_count
                else source_duration_seconds
            )

            raw_intervals.append(
                TemporalInterval(
                    start_frame=start_frame,
                    end_frame_exclusive=end_frame,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    priority=decision.priority,
                    decision_count=1,
                )
            )

        intervals = _coalesce_intervals(raw_intervals)
        covered = sum(interval.frame_count for interval in intervals)

        if covered != source_frame_count:
            raise FrameSelectorError(
                "Temporal plan does not cover every source frame exactly once: "
                f"covered={covered}, source={source_frame_count}"
            )

        for previous, current in zip(intervals, intervals[1:]):
            if previous.end_frame_exclusive != current.start_frame:
                raise FrameSelectorError("Temporal plan contains a frame gap")

        LOGGER.info(
            "Built %d quality segments covering %d source frames",
            len(intervals),
            covered,
        )

        return TemporalPlan(
            intervals=tuple(intervals),
            source_frame_count=source_frame_count,
            covered_frame_count=covered,
            quality_segment_count=len(intervals),
        )


def _parse_record(raw: Any) -> FrameDecision:
    if not isinstance(raw, Mapping):
        raise ValueError("JSON value must be an object")

    frame_index = _nonnegative_integer(
        raw.get("frame_index"),
        "frame_index",
    )
    timestamp = _nonnegative_number(
        raw.get("timestamp_seconds"),
        "timestamp_seconds",
    )
    importance = _bounded_number(
        raw.get("importance_score"),
        "importance_score",
    )

    decision_value = raw.get("decision")

    if not isinstance(decision_value, str):
        raise ValueError("decision must be 'retain' or 'drop'")

    decision = decision_value.strip().lower()

    if decision not in {"retain", "drop"}:
        raise ValueError("decision must be 'retain' or 'drop'")

    priority_value = raw.get("priority")

    if not isinstance(priority_value, str):
        raise ValueError(
            "priority must be critical, high, medium, or low"
        )

    try:
        priority = Priority(priority_value.strip().lower())
    except ValueError as exc:
        raise ValueError(
            "priority must be critical, high, medium, or low"
        ) from exc

    reason_value = raw.get("reason")

    if not isinstance(reason_value, str) or not reason_value.strip():
        raise ValueError("reason must be a nonempty string")

    schema = raw.get("schema_version")

    if schema is not None and schema != "1.0":
        raise ValueError(f"unsupported schema_version: {schema}")

    phase = raw.get("phase")

    if phase is not None and phase != "phase_3_frame_decision":
        raise ValueError(f"unexpected phase identifier: {phase}")

    return FrameDecision(
        frame_index=frame_index,
        timestamp_seconds=timestamp,
        importance_score=importance,
        decision=decision,
        priority=priority,
        reason=reason_value.strip(),
    )


def _coalesce_intervals(
    intervals: list[TemporalInterval],
) -> list[TemporalInterval]:
    result: list[TemporalInterval] = []

    for interval in intervals:
        if result and result[-1].priority is interval.priority:
            previous = result[-1]

            result[-1] = TemporalInterval(
                start_frame=previous.start_frame,
                end_frame_exclusive=interval.end_frame_exclusive,
                start_seconds=previous.start_seconds,
                end_seconds=interval.end_seconds,
                priority=previous.priority,
                decision_count=(
                    previous.decision_count + interval.decision_count
                ),
            )
        else:
            result.append(interval)

    return result


def _nonnegative_integer(value: Any, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a nonnegative integer")

    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{field} must be a nonnegative integer"
        ) from exc

    if (
        not math.isfinite(numeric)
        or not numeric.is_integer()
        or numeric < 0
    ):
        raise ValueError(f"{field} must be a nonnegative integer")

    return int(numeric)


def _nonnegative_number(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be a nonnegative number")

    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{field} must be a nonnegative number"
        ) from exc

    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{field} must be a nonnegative number")

    return result


def _bounded_number(value: Any, field: str) -> float:
    result = _nonnegative_number(value, field)

    if result > 1.0:
        raise ValueError(f"{field} must be between 0 and 1")

    return result

