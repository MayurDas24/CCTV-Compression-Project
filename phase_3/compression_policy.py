"""Explainable frame-retention policy for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from importance_engine import ImportanceAssessment, Priority
from temporal_analyzer import TemporalContext


class Decision(str, Enum):
    """Downstream frame-retention action."""

    RETAIN = "retain"
    DROP = "drop"


class DecisionReason(str, Enum):
    """Stable reason codes written to Phase 3 output."""

    CRITICAL_OBJECT = "critical_object"
    NEW_IMPORTANT_OBJECT = "new_important_object"
    HIGH_IMPORTANCE = "high_importance"
    MEDIUM_IMPORTANCE = "medium_importance"
    LOW_IMPORTANCE = "low_importance"
    TEMPORAL_PROTECTION = "temporal_protection"
    MAXIMUM_DROP_LIMIT = "maximum_drop_limit"
    EMPTY_FRAME = "empty_frame"
    EMPTY_FRAME_RETAINED = "empty_frame_retained"


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Configuration owned by the compression policy."""

    retain_empty_frames: bool
    max_consecutive_drops: int

    def __post_init__(self) -> None:
        if self.max_consecutive_drops < 0:
            raise ValueError(
                "max_consecutive_drops must be zero or greater"
            )


@dataclass(frozen=True, slots=True)
class FrameDecision:
    """Typed policy result for one frame."""

    decision: Decision
    priority: Priority
    reason: DecisionReason

    @property
    def retained(self) -> bool:
        """Return whether the policy retained the frame."""

        return self.decision is Decision.RETAIN


class CompressionPolicy:
    """Produce deterministic and explainable retention decisions."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config

    def decide(
        self,
        assessment: ImportanceAssessment,
        temporal: TemporalContext,
    ) -> FrameDecision:
        """Apply the configured retention rules in safety-first order."""

        if assessment.critical_classes_present:
            return self._retain(
                assessment.priority,
                DecisionReason.CRITICAL_OBJECT,
            )

        if temporal.newly_appearing_important_classes:
            return self._retain(
                assessment.priority,
                DecisionReason.NEW_IMPORTANT_OBJECT,
            )

        if assessment.priority is Priority.HIGH:
            return self._retain(
                assessment.priority,
                DecisionReason.HIGH_IMPORTANCE,
            )

        if assessment.priority is Priority.MEDIUM:
            return self._retain(
                assessment.priority,
                DecisionReason.MEDIUM_IMPORTANCE,
            )

        if (
            assessment.is_empty
            and self.config.retain_empty_frames
        ):
            return self._retain(
                assessment.priority,
                DecisionReason.EMPTY_FRAME_RETAINED,
            )

        # If the configured limit is 10, ten consecutive drops are
        # allowed. The next frame is retained so the run never becomes
        # longer than ten.
        if (
            temporal.drop_limit_reached
            or temporal.consecutive_drops
            >= self.config.max_consecutive_drops
        ):
            return self._retain(
                assessment.priority,
                DecisionReason.MAXIMUM_DROP_LIMIT,
            )

        # A class-set or substantial object-count transition can be
        # operationally meaningful even if its raw score is low.
        if temporal.scene_changed:
            return self._retain(
                assessment.priority,
                DecisionReason.TEMPORAL_PROTECTION,
            )

        reason = (
            DecisionReason.EMPTY_FRAME
            if assessment.is_empty
            else DecisionReason.LOW_IMPORTANCE
        )

        return FrameDecision(
            decision=Decision.DROP,
            priority=assessment.priority,
            reason=reason,
        )

    @staticmethod
    def _retain(
        priority: Priority,
        reason: DecisionReason,
    ) -> FrameDecision:
        """Construct a retain decision."""

        return FrameDecision(
            decision=Decision.RETAIN,
            priority=priority,
            reason=reason,
        )