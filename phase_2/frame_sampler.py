"""
Frame sampling utilities for the Phase 2 AI perception pipeline.

The sampler reduces the number of frames sent to the AI detector while
preserving a predictable sampling rate. This is important for CCTV because
the camera may produce 25-30+ FPS while object detection does not necessarily
need to run on every frame.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict


logger = logging.getLogger("phase_2.frame_sampler")


@dataclass
class FrameSampler:
    """
    Determines which video frames should be analyzed by the AI detector.

    Supported modes:
        - "fixed_fps": sample at a target FPS
        - "every_n": sample every Nth frame
        - "all": analyze every frame
    """

    mode: str = "fixed_fps"
    target_fps: float = 5.0
    step_interval: int = 1

    def __post_init__(self) -> None:
        """Validate and normalize sampler configuration."""

        self.mode = str(self.mode).strip().lower()

        valid_modes = {"fixed_fps", "every_n", "all"}

        if self.mode not in valid_modes:
            raise ValueError(
                f"Unsupported sampling mode '{self.mode}'. "
                f"Expected one of: {sorted(valid_modes)}"
            )

        try:
            self.target_fps = float(self.target_fps)
        except (TypeError, ValueError) as exc:
            raise ValueError("target_fps must be numeric.") from exc

        if self.target_fps < 0:
            raise ValueError("target_fps cannot be negative.")

        try:
            self.step_interval = int(self.step_interval)
        except (TypeError, ValueError) as exc:
            raise ValueError("step_interval must be an integer.") from exc

        if self.step_interval < 1:
            raise ValueError("step_interval must be >= 1.")

        if self.mode == "all":
            self.step_interval = 1

    @classmethod
    def from_config(
        cls,
        config: Dict[str, Any],
        video_fps: float,
    ) -> "FrameSampler":
        """
        Create a FrameSampler from the project's YAML configuration.

        Expected configuration:

            sampling:
              mode: fixed_fps
              target_fps: 5

        Or:

            sampling:
              mode: every_n
              step_interval: 6
        """

        sampling_cfg = config.get("sampling", {})

        if not isinstance(sampling_cfg, dict):
            raise ValueError("'sampling' configuration must be a mapping.")

        mode = str(sampling_cfg.get("mode", "fixed_fps")).strip().lower()

        if mode == "all":
            return cls(
                mode="all",
                target_fps=float(video_fps),
                step_interval=1,
            )

        if mode == "every_n":
            step_interval = sampling_cfg.get("step_interval", 1)

            try:
                step_interval = int(step_interval)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "'sampling.step_interval' must be an integer."
                ) from exc

            if step_interval < 1:
                raise ValueError("'sampling.step_interval' must be >= 1.")

            effective_fps = (
                float(video_fps) / step_interval
                if video_fps > 0
                else 0.0
            )

            logger.info(
                "Frame sampler configured: every %d frame(s) | "
                "Source FPS: %.2f | Effective sampling FPS: %.2f",
                step_interval,
                video_fps,
                effective_fps,
            )

            return cls(
                mode="every_n",
                target_fps=effective_fps,
                step_interval=step_interval,
            )

        # Default / fixed_fps mode
        target_fps = sampling_cfg.get("target_fps", 5.0)

        try:
            target_fps = float(target_fps)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "'sampling.target_fps' must be numeric."
            ) from exc

        if target_fps <= 0:
            raise ValueError(
                "'sampling.target_fps' must be greater than 0."
            )

        if video_fps <= 0:
            logger.warning(
                "Invalid video FPS %.2f. Using one frame every frame.",
                video_fps,
            )
            step_interval = 1
            effective_fps = target_fps
        else:
            # Example:
            # source = 30 FPS
            # target = 5 FPS
            # interval = ceil(30 / 5) = 6
            step_interval = max(1, math.ceil(video_fps / target_fps))
            effective_fps = video_fps / step_interval

        logger.info(
            "Frame sampler configured: fixed FPS | "
            "Source FPS: %.2f | Requested: %.2f | "
            "Interval: every %d frame(s) | Effective FPS: %.2f",
            video_fps,
            target_fps,
            step_interval,
            effective_fps,
        )

        return cls(
            mode="fixed_fps",
            target_fps=target_fps,
            step_interval=step_interval,
        )

    def should_sample(self, frame_index: int) -> bool:
        """
        Return True if the specified frame should be analyzed.

        Frame index is zero-based.

        Examples:
            step_interval = 1 -> every frame
            step_interval = 2 -> frames 0,2,4,6...
            step_interval = 6 -> frames 0,6,12,18...
        """

        if frame_index < 0:
            raise ValueError("frame_index cannot be negative.")

        if self.mode == "all":
            return True

        return frame_index % self.step_interval == 0

    def effective_fps(self, source_fps: float) -> float:
        """Return the actual expected sampling FPS."""

        if source_fps <= 0:
            return 0.0

        if self.mode == "all":
            return float(source_fps)

        return float(source_fps / self.step_interval)

    def to_dict(self, source_fps: float | None = None) -> Dict[str, Any]:
        """Return sampler configuration as JSON-serializable metadata."""

        data: Dict[str, Any] = {
            "mode": self.mode,
            "target_fps": float(self.target_fps),
            "step_interval": int(self.step_interval),
        }

        if source_fps is not None:
            data["source_fps"] = float(source_fps)
            data["effective_fps"] = float(
                self.effective_fps(source_fps)
            )

        return data