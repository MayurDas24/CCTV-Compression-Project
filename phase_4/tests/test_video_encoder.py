"""Focused tests for Phase 4 media probing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PHASE4_DIRECTORY = Path(__file__).resolve().parents[1]

if str(PHASE4_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PHASE4_DIRECTORY))


from video_encoder import VideoEncoder, VideoEncoderError


def _probe_payload(
    *,
    metadata_frames: str = "3640",
    counted_frames: str | None = "3632",
) -> dict[str, object]:
    video: dict[str, object] = {
        "codec_type": "video",
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": 1920,
        "height": 1080,
        "avg_frame_rate": "25/1",
        "r_frame_rate": "25/1",
        "duration": "145.372800",
        "nb_frames": metadata_frames,
    }

    if counted_frames is not None:
        video["nb_read_frames"] = counted_frames

    return {
        "streams": [video],
        "format": {
            "duration": "145.372800",
            "bit_rate": "8000000",
        },
    }


class VideoEncoderProbeTests(unittest.TestCase):
    """Verify that decoded-frame counts are authoritative."""

    def test_probe_uses_counted_frames_when_metadata_conflicts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.mp4"
            source.write_bytes(b"not-empty")
            encoder = VideoEncoder.__new__(VideoEncoder)

            with patch.object(
                VideoEncoder,
                "_run_ffprobe",
                return_value=_probe_payload(),
            ) as run_ffprobe:
                properties = encoder.probe(source)

            run_ffprobe.assert_called_once_with(
                source.resolve(),
                count_frames=True,
            )
            self.assertEqual(properties.frame_count, 3632)

    def test_probe_does_not_fall_back_to_container_frame_count(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.mp4"
            source.write_bytes(b"not-empty")
            encoder = VideoEncoder.__new__(VideoEncoder)

            with patch.object(
                VideoEncoder,
                "_run_ffprobe",
                return_value=_probe_payload(counted_frames=None),
            ) as run_ffprobe:
                with self.assertRaisesRegex(
                    VideoEncoderError,
                    "Unable to determine a reliable frame count",
                ):
                    encoder.probe(source)

            run_ffprobe.assert_called_once_with(
                source.resolve(),
                count_frames=True,
            )


if __name__ == "__main__":
    unittest.main()
