"""Focused tests for authoritative Phase 5 frame counting."""

from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


PHASE5_DIRECTORY = Path(__file__).resolve().parents[1]

if str(PHASE5_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PHASE5_DIRECTORY))


from quality_analysis import QualityConfig, assess_timeline_alignment
from validation import MediaInspector


def _metadata_payload(frame_count: str = "3640") -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "25/1",
                "r_frame_rate": "25/1",
                "duration": "145.372800",
                "nb_frames": frame_count,
            }
        ],
        "format": {
            "duration": "145.372800",
            "bit_rate": "8000000",
        },
    }


def _count_payload(frame_count: str | None) -> dict[str, object]:
    stream: dict[str, object] = {}

    if frame_count is not None:
        stream["nb_read_frames"] = frame_count

    return {"streams": [stream]}


def _inspector() -> MediaInspector:
    inspector = MediaInspector.__new__(MediaInspector)
    inspector.ffprobe = "ffprobe"
    inspector.timeout_seconds = 30.0
    return inspector


def _quality_config() -> QualityConfig:
    return QualityConfig(
        enabled=True,
        metrics=(),
        maximum_fps_difference=0.05,
        maximum_duration_difference_seconds=0.25,
        timeout_seconds=30.0,
    )


class FrameCountValidationTests(unittest.TestCase):
    """Exercise counted-frame preference and safe fallback behavior."""

    def test_decoded_count_overrides_conflicting_container_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.mp4"
            source.write_bytes(b"not-empty")
            inspector = _inspector()

            with patch.object(
                inspector,
                "_run_ffprobe",
                side_effect=[
                    _metadata_payload("3640"),
                    _count_payload("3632"),
                ],
            ) as run_ffprobe:
                information = inspector.inspect(source, "Source video")

            self.assertEqual(information.frame_count, 3632)
            self.assertEqual(
                information.frame_count_method,
                "ffprobe_count_frames",
            )
            self.assertEqual(run_ffprobe.call_count, 2)
            counted_command = run_ffprobe.call_args_list[1].args[0]
            self.assertIn("-count_frames", counted_command)
            self.assertIn(
                "stream=nb_read_frames",
                counted_command,
            )

    def test_container_fallback_cannot_prove_alignment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_path = Path(temp) / "source.mp4"
            source_path.write_bytes(b"not-empty")
            inspector = _inspector()

            with patch.object(
                inspector,
                "_run_ffprobe",
                side_effect=[
                    _metadata_payload("3640"),
                    _count_payload(None),
                ],
            ):
                source = inspector.inspect(
                    source_path,
                    "Source video",
                )

            compressed = replace(
                source,
                path=Path(temp) / "compressed.mp4",
            )
            alignment = assess_timeline_alignment(
                source,
                compressed,
                _quality_config(),
            )

            self.assertEqual(source.frame_count, 3640)
            self.assertEqual(
                source.frame_count_method,
                "container_metadata_fallback",
            )
            self.assertFalse(alignment.aligned)
            self.assertIn(
                "Decoded frame counts are unavailable",
                alignment.reasons[0],
            )

    def test_authoritative_frame_count_mismatch_still_fails_alignment(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source_path = Path(temp) / "source.mp4"
            source_path.write_bytes(b"not-empty")
            inspector = _inspector()

            with patch.object(
                inspector,
                "_run_ffprobe",
                side_effect=[
                    _metadata_payload("3640"),
                    _count_payload("3632"),
                ],
            ):
                source = inspector.inspect(
                    source_path,
                    "Source video",
                )

            compressed = replace(
                source,
                path=Path(temp) / "compressed.mp4",
                frame_count=3631,
            )
            alignment = assess_timeline_alignment(
                source,
                compressed,
                _quality_config(),
            )

            self.assertFalse(alignment.aligned)
            self.assertIn(
                "Source and compressed frame counts differ",
                alignment.reasons,
            )


if __name__ == "__main__":
    unittest.main()
