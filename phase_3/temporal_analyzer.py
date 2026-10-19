"""Bounded temporal state for Phase 3 frame decisions."""

from __future__ import annotations

from dataclasses import dataclass

from importance_engine import ImportanceAssessment, Priority


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    """Configuration used by the temporal analyzer."""

    max_consecutive_drops: int

    def __post_init__(self) -> None:
        if self.max_consecutive_drops < 0:
            raise ValueError(
                "max_consecutive_drops must be zero or greater"
            )


@dataclass(frozen=True, slots=True)
class TemporalContext:
    """Temporal state immediately before the current decision."""

    consecutive_drops: int
    consecutive_low_frames: int
    consecutive_active_frames: int
    newly_appearing_important_classes: tuple[str, ...]
    scene_changed: bool
    drop_limit_reached: bool


class TemporalAnalyzer:
    """Track constant-size state across perception records.

    The analyzer stores only the previous frame's object information and
    a small set of counters. Its memory usage therefore remains constant
    regardless of the input JSONL size.
    """

    def __init__(self, config: TemporalConfig) -> None:
        self.config = config

        self._consecutive_drops = 0
        self._consecutive_low_frames = 0
        self._consecutive_active_frames = 0

        self._previous_classes: frozenset[str] = frozenset()
        self._previous_object_count: int | None = None

    @property
    def consecutive_drops(self) -> int:
        """Return the current number of consecutive dropped frames."""

        return self._consecutive_drops

    def analyze(
        self,
        assessment: ImportanceAssessment,
    ) -> TemporalContext:
        """Return temporal context without modifying internal state."""

        current_classes = frozenset(
            assessment.classes_present
        )

        important_classes_now = frozenset(
            assessment.critical_classes_present
            + assessment.important_classes_present
        )

        newly_appearing_classes = tuple(
            sorted(
                important_classes_now
                - self._previous_classes
            )
        )

        scene_changed = False

        if self._previous_object_count is not None:
            class_union = (
                current_classes
                | self._previous_classes
            )

            if class_union:
                class_delta_ratio = (
                    len(
                        current_classes
                        ^ self._previous_classes
                    )
                    / len(class_union)
                )
            else:
                class_delta_ratio = 0.0

            object_count_delta = abs(
                assessment.number_of_objects
                - self._previous_object_count
            )

            scene_changed = (
                class_delta_ratio >= 0.5
                or object_count_delta >= 3
            )

        next_low_count = (
            self._consecutive_low_frames + 1
            if assessment.priority is Priority.LOW
            else 0
        )

        next_active_count = (
            self._consecutive_active_frames + 1
            if assessment.priority
            in {Priority.CRITICAL, Priority.HIGH}
            else 0
        )

        return TemporalContext(
            consecutive_drops=self._consecutive_drops,
            consecutive_low_frames=next_low_count,
            consecutive_active_frames=next_active_count,
            newly_appearing_important_classes=(
                newly_appearing_classes
            ),
            scene_changed=scene_changed,
            drop_limit_reached=(
                self._consecutive_drops
                >= self.config.max_consecutive_drops
            ),
        )

    def update(
        self,
        assessment: ImportanceAssessment,
        *,
        retained: bool,
    ) -> None:
        """Commit the current assessment and decision to state."""

        if retained:
            self._consecutive_drops = 0
        else:
            self._consecutive_drops += 1

        if assessment.priority is Priority.LOW:
            self._consecutive_low_frames += 1
        else:
            self._consecutive_low_frames = 0

        if assessment.priority in {
            Priority.CRITICAL,
            Priority.HIGH,
        }:
            self._consecutive_active_frames += 1
        else:
            self._consecutive_active_frames = 0

        self._previous_classes = frozenset(
            assessment.classes_present
        )

        self._previous_object_count = (
            assessment.number_of_objects
        )