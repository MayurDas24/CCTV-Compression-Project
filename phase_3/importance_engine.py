"""Importance scoring for Phase 3 frame-retention decisions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping, Sequence


class Priority(str, Enum):
    """Normalized priority assigned to a perception record."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass(frozen=True, slots=True)
class ImportanceConfig:
    """Configuration used by ImportanceEngine."""

    high_importance_threshold: float
    low_importance_threshold: float
    minimum_detection_confidence: float
    critical_classes: frozenset[str]
    important_classes: frozenset[str]

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_importance_threshold <= 1.0:
            raise ValueError(
                "low_importance_threshold must be between 0 and 1"
            )

        if not 0.0 <= self.high_importance_threshold <= 1.0:
            raise ValueError(
                "high_importance_threshold must be between 0 and 1"
            )

        if (
            self.low_importance_threshold
            > self.high_importance_threshold
        ):
            raise ValueError(
                "low_importance_threshold cannot exceed "
                "high_importance_threshold"
            )

        if not 0.0 <= self.minimum_detection_confidence <= 1.0:
            raise ValueError(
                "minimum_detection_confidence must be between 0 and 1"
            )


@dataclass(frozen=True, slots=True)
class DetectionSignal:
    """Validated detection fields used for importance scoring."""

    class_name: str
    confidence: float
    normalized_area: float
    importance_score: float


@dataclass(frozen=True, slots=True)
class ImportanceAssessment:
    """Typed importance assessment for one Phase 2 frame record."""

    score: float
    source_score: float
    priority: Priority
    number_of_objects: int
    classes_present: tuple[str, ...]
    critical_classes_present: tuple[str, ...]
    important_classes_present: tuple[str, ...]
    qualifying_detections: tuple[DetectionSignal, ...]
    is_empty: bool
    issues: tuple[str, ...]


class ImportanceEngine:
    """Convert Phase 2 perception metadata into an importance assessment.

    The score uses the stronger of:

    - The Phase 2 frame importance score.
    - An additive object signal based on detection importance, confidence,
      and normalized area.

    A bounded object-count bonus recognizes busy frames without letting
    object count dominate the decision.

    Additive scoring prevents an important detection from disappearing
    merely because one signal, such as normalized area, is small.
    """

    _DETECTION_IMPORTANCE_WEIGHT = 0.50
    _CONFIDENCE_WEIGHT = 0.30
    _AREA_WEIGHT = 0.20

    _AREA_SATURATION = 0.10
    _MAX_OBJECT_COUNT_BONUS = 0.10
    _OBJECT_COUNT_SATURATION = 5

    def __init__(self, config: ImportanceConfig) -> None:
        self.config = config

    def assess(
        self,
        metadata: Mapping[str, Any],
    ) -> ImportanceAssessment:
        """Assess a Phase 2 record defensively.

        Record-level problems are returned through ``issues`` instead of
        terminating the pipeline.
        """

        issues: list[str] = []

        source_score = self._bounded_float(
            metadata.get("frame_importance_score"),
            default=0.0,
            field="frame_importance_score",
            issues=issues,
        )

        raw_detections = metadata.get("detections", [])

        if not isinstance(raw_detections, Sequence) or isinstance(
            raw_detections,
            (str, bytes, bytearray),
        ):
            issues.append("detections is not a list; treated as empty")
            raw_detections = []

        parsed_detections: list[DetectionSignal] = []
        all_detection_classes: set[str] = set()

        for index, raw_detection in enumerate(raw_detections):
            if not isinstance(raw_detection, Mapping):
                issues.append(
                    f"detections[{index}] is not an object; ignored"
                )
                continue

            signal = self._parse_detection(
                raw_detection,
                index,
                issues,
            )

            if signal is None:
                continue

            parsed_detections.append(signal)
            all_detection_classes.add(signal.class_name)

        qualifying_detections = tuple(
            detection
            for detection in parsed_detections
            if detection.confidence
            >= self.config.minimum_detection_confidence
        )

        declared_classes = self._parse_classes(
            metadata.get("classes_present"),
            issues,
        )

        classes_present = tuple(
            sorted(declared_classes | all_detection_classes)
        )

        number_of_objects = self._nonnegative_int(
            metadata.get("number_of_objects"),
            default=len(parsed_detections),
            field="number_of_objects",
            issues=issues,
        )

        number_of_objects = max(
            number_of_objects,
            len(parsed_detections),
        )

        qualifying_classes = {
            detection.class_name
            for detection in qualifying_detections
        }

        if not parsed_detections and number_of_objects > 0:
            # When detailed detections are missing, use the declared
            # classes conservatively. This prevents a declared critical
            # object such as "person" from being silently discarded.
            qualifying_classes.update(declared_classes)

            if declared_classes:
                issues.append(
                    "detection details missing; declared classes "
                    "used conservatively"
                )

        critical_classes_present = tuple(
            sorted(
                qualifying_classes
                & self.config.critical_classes
            )
        )

        important_classes_present = tuple(
            sorted(
                qualifying_classes
                & self.config.important_classes
            )
        )

        score = max(
            source_score,
            self._object_signal(qualifying_detections),
        )

        score = min(
            1.0,
            score
            + self._object_count_bonus(
                len(qualifying_detections)
            ),
        )

        if critical_classes_present:
            score = max(
                score,
                self.config.high_importance_threshold,
            )
            priority = Priority.CRITICAL
        else:
            if important_classes_present:
                score = max(
                    score,
                    self.config.low_importance_threshold,
                )

            priority = self._priority_for_score(score)

        is_empty = (
            number_of_objects == 0
            and not classes_present
            and not parsed_detections
        )

        return ImportanceAssessment(
            score=round(score, 6),
            source_score=round(source_score, 6),
            priority=priority,
            number_of_objects=number_of_objects,
            classes_present=classes_present,
            critical_classes_present=critical_classes_present,
            important_classes_present=important_classes_present,
            qualifying_detections=qualifying_detections,
            is_empty=is_empty,
            issues=tuple(issues),
        )

    def _parse_detection(
        self,
        detection: Mapping[str, Any],
        index: int,
        issues: list[str],
    ) -> DetectionSignal | None:
        """Validate and normalize one Phase 2 detection."""

        raw_class_name = detection.get("class_name")

        if (
            not isinstance(raw_class_name, str)
            or not raw_class_name.strip()
        ):
            issues.append(
                f"detections[{index}].class_name is invalid; ignored"
            )
            return None

        class_name = raw_class_name.strip().lower()

        confidence = self._bounded_float(
            detection.get("confidence"),
            default=0.0,
            field=f"detections[{index}].confidence",
            issues=issues,
        )

        normalized_area = self._bounded_float(
            detection.get("normalized_area"),
            default=0.0,
            field=f"detections[{index}].normalized_area",
            issues=issues,
        )

        importance_score = self._bounded_float(
            detection.get("importance_score"),
            default=0.0,
            field=f"detections[{index}].importance_score",
            issues=issues,
        )

        return DetectionSignal(
            class_name=class_name,
            confidence=confidence,
            normalized_area=normalized_area,
            importance_score=importance_score,
        )

    @staticmethod
    def _parse_classes(
        value: Any,
        issues: list[str],
    ) -> set[str]:
        """Normalize the Phase 2 classes_present collection."""

        if value is None:
            return set()

        if not isinstance(value, Sequence) or isinstance(
            value,
            (str, bytes, bytearray),
        ):
            issues.append(
                "classes_present is not a list; treated as empty"
            )
            return set()

        classes: set[str] = set()

        for item in value:
            if isinstance(item, str) and item.strip():
                classes.add(item.strip().lower())
            else:
                issues.append(
                    "classes_present contains an invalid value; ignored"
                )

        return classes

    def _object_signal(
        self,
        detections: tuple[DetectionSignal, ...],
    ) -> float:
        """Calculate the explainable additive object signal."""

        if not detections:
            return 0.0

        maximum_importance = max(
            detection.importance_score
            for detection in detections
        )

        maximum_confidence = max(
            detection.confidence
            for detection in detections
        )

        maximum_area = max(
            detection.normalized_area
            for detection in detections
        )

        area_signal = min(
            1.0,
            maximum_area / self._AREA_SATURATION,
        )

        return (
            self._DETECTION_IMPORTANCE_WEIGHT
            * maximum_importance
            + self._CONFIDENCE_WEIGHT
            * maximum_confidence
            + self._AREA_WEIGHT
            * area_signal
        )

    def _object_count_bonus(self, count: int) -> float:
        """Return a bounded bonus for frames containing many objects."""

        if count <= 0:
            return 0.0

        saturated_count = min(
            count,
            self._OBJECT_COUNT_SATURATION,
        )

        return self._MAX_OBJECT_COUNT_BONUS * (
            saturated_count
            / self._OBJECT_COUNT_SATURATION
        )

    def _priority_for_score(self, score: float) -> Priority:
        """Convert a normalized score into a configured priority."""

        if score >= self.config.high_importance_threshold:
            return Priority.HIGH

        if score >= self.config.low_importance_threshold:
            return Priority.MEDIUM

        return Priority.LOW

    @staticmethod
    def _bounded_float(
        value: Any,
        *,
        default: float,
        field: str,
        issues: list[str],
    ) -> float:
        """Convert a value into a finite float clamped to [0, 1]."""

        try:
            result = float(value)
        except (TypeError, ValueError):
            issues.append(
                f"{field} is invalid; defaulted to {default}"
            )
            return default

        if not math.isfinite(result):
            issues.append(
                f"{field} is not finite; defaulted to {default}"
            )
            return default

        if result < 0.0 or result > 1.0:
            issues.append(
                f"{field} is outside [0, 1]; clamped"
            )

        return min(1.0, max(0.0, result))

    @staticmethod
    def _nonnegative_int(
        value: Any,
        *,
        default: int,
        field: str,
        issues: list[str],
    ) -> int:
        """Convert a value into a nonnegative integer safely."""

        if isinstance(value, bool):
            issues.append(
                f"{field} is invalid; defaulted to {default}"
            )
            return default

        try:
            numeric = float(value)
        except (TypeError, ValueError, OverflowError):
            issues.append(
                f"{field} is invalid; defaulted to {default}"
            )
            return default

        if not math.isfinite(numeric) or not numeric.is_integer():
            issues.append(
                f"{field} is invalid; defaulted to {default}"
            )
            return default

        result = int(numeric)

        if result < 0:
            issues.append(
                f"{field} is negative; clamped to 0"
            )
            return 0

        return result