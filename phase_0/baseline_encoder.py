import argparse
import json
import logging
import os
import pathlib
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Dict, Any, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("BaselineEncoder")


@dataclass
class EncodingMetrics:
    input_file: str
    output_file: str
    input_size_bytes: int
    output_size_bytes: int
    duration_seconds: float
    bitrate_kbps: float
    compression_ratio: float
    storage_reduction_percent: float


def run_ffprobe(filepath: pathlib.Path) -> Dict[str, Any]:
    """Extract metadata from video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(filepath)
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        logger.error(f"Failed to execute ffprobe on {filepath}: {e}")
        return {}


class BaselineEncoder:
    def __init__(self, codec: str = "libx265", preset: str = "medium", crf: int = 28, video_only: bool = True):
        self.codec = codec
        self.preset = preset
        self.crf = crf
        self.video_only = video_only

    def encode(self, input_path: pathlib.Path, output_path: pathlib.Path, overwrite: bool = False) -> EncodingMetrics:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"Output file already exists: {output_path}. Use overwrite flag.")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "ffmpeg", "-y" if overwrite else "-n",
            "-i", str(input_path),
            "-c:v", self.codec,
            "-preset", self.preset,
            "-crf", str(self.crf),
            "-pix_fmt", "yuv420p"
        ]
        
        if self.video_only:
            cmd.append("-an")

        cmd.append(str(output_path))

        logger.info(f"Encoding {input_path} -> {output_path} (CRF {self.crf}, Preset {self.preset})")
        start_time = time.time()
        
        process = subprocess.run(cmd, capture_output=True, text=True)
        if process.returncode != 0:
            logger.error(f"FFmpeg error output: {process.stderr}")
            raise RuntimeError("FFmpeg encoding failed.")

        elapsed = time.time() - start_time
        
        input_size = input_path.stat().st_size
        output_size = output_path.stat().st_size
        
        comp_ratio = input_size / output_size if output_size > 0 else 0.0
        storage_reduction = ((input_size - output_size) / input_size) * 100.0 if input_size > 0 else 0.0
        
        probe_data = run_ffprobe(output_path)
        duration = float(probe_data.get("format", {}).get("duration", elapsed))
        bitrate_kbps = (output_size * 8) / (duration * 1000) if duration > 0 else 0.0

        metrics = EncodingMetrics(
            input_file=str(input_path),
            output_file=str(output_path),
            input_size_bytes=input_size,
            output_size_bytes=output_size,
            duration_seconds=elapsed,
            bitrate_kbps=bitrate_kbps,
            compression_ratio=comp_ratio,
            storage_reduction_percent=storage_reduction
        )
        
        logger.info(f"Encoding completed in {elapsed:.2f}s. Bitrate: {bitrate_kbps:.2f} kbps. Reduction: {storage_reduction:.2f}%")
        return metrics


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Baseline H.265 Encoder")
    parser.add_argument("--input", required=True, type=pathlib.Path, help="Input video path")
    parser.add_argument("--output", required=True, type=pathlib.Path, help="Output video path")
    parser.add_argument("--crf", type=int, default=28, help="Constant Rate Factor (default: 28)")
    parser.add_argument("--preset", type=str, default="medium", help="FFmpeg encoding preset")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output file")
    
    args = parser.parse_args()
    
    encoder = BaselineEncoder(crf=args.crf, preset=args.preset)
    try:
        metrics = encoder.encode(args.input, args.output, overwrite=args.overwrite)
        print(json.dumps(asdict(metrics), indent=2))
    except Exception as e:
        logger.error(f"Encoding failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()