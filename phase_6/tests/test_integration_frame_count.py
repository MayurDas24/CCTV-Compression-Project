"""Focused tests for Phase 6 frame-count consistency."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PHASE6_DIRECTORY = Path(__file__).resolve().parents[1]

if str(PHASE6_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PHASE6_DIRECTORY))


from integration import (
    IntegrationError,
    IntegrationPaths,
    IntegrationValidator,
    LoadedPipelineData,
    MediaInspector,
    MediaMetadata,
    Phase3Statistics,
    ValidationTolerances,
)


def _metadata_payload() -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "h264",
                "pix_fmt": "yuv420p",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": "25/1",
                "duration": "145.372800",
                "nb_frames": "3640",
            }
        ],
        "format": {"duration": "145.372800"},
    }


def _count_payload(value: str | None) -> dict[str, object]:
    stream: dict[str, object] = {}

    if value is not None:
        stream["nb_read_frames"] = value

    return {"streams": [stream]}


def _media(path: Path, frame_count: int) -> MediaMetadata:
    return MediaMetadata(
        path=path,
        size_bytes=100,
        width=1920,
        height=1080,
        average_fps=25.0,
        duration_seconds=145.3728,
        frame_count=frame_count,
        frame_count_method="ffprobe_count_frames",
        codec="h264",
        pixel_format="yuv420p",
    )


def _reported_media(media: MediaMetadata) -> dict[str, object]:
    return {
        "width": media.width,
        "height": media.height,
        "average_fps": media.average_fps,
        "duration_seconds": media.duration_seconds,
        "frame_count": media.frame_count,
        "codec": media.codec,
        "pixel_format": media.pixel_format,
    }


class Phase6FrameCountTests(unittest.TestCase):
    """Verify decoded counting and strict cross-media comparison."""

    def test_inspector_uses_decoded_count_when_metadata_conflicts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.mp4"
            source.write_bytes(b"not-empty")
            inspector = MediaInspector.__new__(MediaInspector)
            inspector.ffprobe = "ffprobe"
            inspector.timeout_seconds = 120.0

            with patch.object(
                inspector,
                "_probe",
                side_effect=[
                    _metadata_payload(),
                    _count_payload("3632"),
                ],
            ) as probe:
                metadata = inspector.inspect(source, "source video")

            self.assertEqual(metadata.frame_count, 3632)
            self.assertEqual(
                metadata.frame_count_method,
                "ffprobe_count_frames",
            )
            self.assertEqual(probe.call_count, 2)
            counted_command = probe.call_args_list[1].args[0]
            self.assertIn("-count_frames", counted_command)
            self.assertIn(
                "stream=nb_read_frames",
                counted_command,
            )

    def test_inspector_rejects_missing_decoded_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.mp4"
            source.write_bytes(b"not-empty")
            inspector = MediaInspector.__new__(MediaInspector)
            inspector.ffprobe = "ffprobe"
            inspector.timeout_seconds = 120.0

            with patch.object(
                inspector,
                "_probe",
                side_effect=[
                    _metadata_payload(),
                    _count_payload(None),
                ],
            ):
                with self.assertRaisesRegex(
                    IntegrationError,
                    "no measured frame count",
                ):
                    inspector.inspect(source, "source video")

    def test_genuine_frame_count_mismatch_remains_an_error(
        self,
    ) -> None:
        source = _media(Path("source.mp4"), 3632)
        compressed = _media(Path("compressed.mp4"), 3631)
        phase4 = {
            "video": {
                "source": _reported_media(source),
                "output": _reported_media(compressed),
            }
        }
        phase5 = {
            "source_video": _reported_media(source),
            "compressed_video": _reported_media(compressed),
        }
        paths = IntegrationPaths(
            source_video=source.path,
            phase3_decisions=Path("decisions.jsonl"),
            phase4_compressed_video=compressed.path,
            phase4_summary=Path("phase4.json"),
            phase5_report_json=Path("phase5.json"),
            phase5_report_text=Path("phase5.txt"),
        )
        data = LoadedPipelineData(
            paths=paths,
            phase3=Phase3Statistics(
                path=paths.phase3_decisions,
                total_records=1,
                retained=1,
                dropped=0,
                malformed=0,
                priority_distribution={"critical": 1},
            ),
            phase4=phase4,
            phase5=phase5,
            source_media=source,
            compressed_media=compressed,
        )
        validator = IntegrationValidator.__new__(
            IntegrationValidator
        )
        validator.tolerances = ValidationTolerances(
            fps=0.05,
            duration_seconds=0.25,
            compression_ratio=0.000001,
            storage_reduction_percent=0.000001,
        )
        checks = []

        validator._check_media(data, checks)

        frame_check = next(
            check
            for check in checks
            if check.name == "source_compressed_frame_count"
        )
        self.assertEqual(frame_check.status, "error")
        self.assertEqual(frame_check.expected, 3632)
        self.assertEqual(frame_check.actual, 3631)


if __name__ == "__main__":
    unittest.main()
