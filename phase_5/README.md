# Phase 5 - Quality and Forensic Validation

## Purpose

Phase 5 is the measurement and validation gate for the CCTV compression core. It consumes the existing Phase 3 and Phase 4 artifacts and determines whether the system produced measurable storage savings without hiding changes to video timing, frame rate, media properties, or decision integrity.

Phase 5 does not modify or rerun Phases 2-4. It does not encode video and does not attempt to fix the known Phase 4 sampled-frame temporal behavior.

## Architecture

```text
Phase 3 frame_decisions.jsonl ----+
Phase 4 compression_summary.json -+--> validation.py
Source video ---------------------+        |
Compressed video -----------------+        v
                                      metrics.py
                                          |
                                          v
                                  quality_analysis.py
                                          |
                                          v
                                     reporting.py
                                          |
                                          v
                             phase5_report.json + .txt
```

- `validation.py` validates decision records, the Phase 4 summary, cross-phase counts, and media readability.
- `metrics.py` calculates storage, video, integrity, frame-decision, and performance metrics.
- `quality_analysis.py` runs optional PSNR, SSIM, and VMAF only when timeline alignment can be proven.
- `reporting.py` writes machine-readable and human-readable reports.
- `phase5_pipeline.py` orchestrates the complete Phase 5 workflow.

Video files are inspected through ffprobe metadata. They are not loaded into Python memory.

## Inputs

Default inputs are:

```text
../sample_videos/test1.mp4
../phase_3/outputs/frame_decisions.jsonl
../phase_4/outputs/compressed_video.mp4
../phase_4/outputs/compression_summary.json
```

All paths are configurable. Relative paths are resolved against the directory containing `config.yaml`.

Phase 3 decision records require:

```json
{
  "frame_index": 0,
  "decision": "retain"
}
```

`frame_index` must be a nonnegative integer. `decision` must be `retain` or `drop`. Records must be strictly ordered.

Additional Phase 3 fields are tolerated. Priority and reason values are aggregated when present.

The Phase 4 summary must contain:

- `schema_version: "1.0"`
- `phase: "phase_4_intelligent_compression"`
- A status of `completed` or `completed_with_warnings`

## Outputs

```text
outputs/
├── phase5_report.json
└── phase5_report.txt
```

The JSON report is the machine-readable source of truth. The text report provides a concise engineering-manager summary.

## Metrics

The report includes:

- Original and compressed file sizes.
- Bytes saved.
- Compression ratio.
- Storage reduction percentage.
- Phase 3 retain and drop counts.
- Decision retention and drop ratios.
- Source and compressed resolution.
- Source and compressed average and nominal FPS.
- Source and compressed duration.
- Source and compressed codec.
- Pixel formats.
- Bitrates.
- Frame counts when available.
- Audio presence and audio codec.
- FPS difference.
- Duration difference.
- Bitrate reduction.
- Codec, resolution, pixel-format, and audio changes.
- Phase 4 encoding duration.
- Phase 4 encoding throughput when source frame count is available.
- Phase 5 analysis duration.
- Phase 5 decision-validation throughput.
- Cross-phase consistency warnings.

Unavailable ffprobe metadata is represented by JSON `null`. Phase 5 does not fabricate or estimate missing values.

Frame-decision retention ratio and physical whole-video frame reduction are not the same measurement. Phase 3 normally contains sampled-frame decisions, so its retain/drop ratio must not be interpreted as complete source-frame coverage.

## Optional quality metrics

Phase 5 supports:

- PSNR
- SSIM
- VMAF

These are full-reference metrics and require corresponding source and compressed frames.

Before calculating them, Phase 5 requires:

- Identical source and compressed resolution.
- Average FPS difference within the configured tolerance.
- Duration difference within the configured tolerance.
- Available and identical frame counts.

If these requirements are not satisfied, the metric status is `unavailable`, with an explanation. The pipeline continues because these metrics are optional.

If alignment is established:

- PSNR uses FFmpeg's `psnr` filter.
- SSIM uses FFmpeg's `ssim` filter.
- VMAF uses FFmpeg's `libvmaf` filter.

If an FFmpeg build does not include a requested filter, only that metric becomes unavailable.

Setting:

```yaml
quality:
  timeout_seconds: 0
```

disables the optional metric timeout. Use a positive value in automated environments.

## Known Phase 4 limitation

The current Phase 4 output is built from retained sampled frames. Its average FPS can therefore be substantially lower than the source FPS, producing visible temporal lag or stuttering.

Phase 5 exposes the FPS, duration, and frame-count differences and refuses to produce misleading PSNR, SSIM, or VMAF scores when frame alignment is not proven.

This is a Phase 3/4 optimization issue, not a Phase 5 processing failure.

Phase 5 also does not rerun YOLO on the compressed video. It therefore does not claim to measure post-compression object-detection accuracy. That requires a future controlled forensic evaluation using the same model and aligned evidence set.

## Requirements

- Python 3.11 or newer.
- FFprobe for required media validation.
- FFmpeg for optional PSNR, SSIM, and VMAF.
- PyYAML.

FFmpeg and ffprobe are external executables and are not installed by pip.

Verify them:

```powershell
ffprobe -version
ffmpeg -version
ffmpeg -hide_banner -filters
```

If they are not available on `PATH`, configure their absolute paths in `config.yaml`.

## Installation

From `phase_5`:

```powershell
python -m pip install -r requirements.txt
```

## Configuration

```yaml
inputs:
  source_video: "../sample_videos/test1.mp4"
  phase3_decisions: "../phase_3/outputs/frame_decisions.jsonl"
  phase4_compressed_video: "../phase_4/outputs/compressed_video.mp4"
  phase4_compression_summary: "../phase_4/outputs/compression_summary.json"

output:
  directory: "outputs"
  json_report: "outputs/phase5_report.json"
  text_report: "outputs/phase5_report.txt"

validation:
  fail_on_malformed_decisions: true
  fail_on_phase4_count_mismatch: true

quality:
  enabled: true
  metrics:
    - psnr
    - ssim
    - vmaf
  maximum_fps_difference: 0.01
  maximum_duration_difference_seconds: 0.10
  timeout_seconds: 0

runtime:
  ffmpeg_path: "ffmpeg"
  ffprobe_path: "ffprobe"
  ffprobe_timeout_seconds: 30

logging:
  level: "INFO"
```

Set `fail_on_malformed_decisions: false` to skip malformed Phase 3 lines while counting and reporting them. Duplicate and out-of-order indices remain fatal.

Set `fail_on_phase4_count_mismatch: false` to report Phase 3 versus Phase 4 decision-count mismatches as warnings.

## Running

From `phase_5`:

```powershell
python -u phase5_pipeline.py
```

With another configuration:

```powershell
python -u phase5_pipeline.py --config C:\path\to\config.yaml
```

## Syntax validation

```powershell
python -m py_compile validation.py metrics.py quality_analysis.py reporting.py phase5_pipeline.py
```

## Error behavior

The program returns exit code `1` for fatal errors such as:

- Missing required input.
- Invalid required JSON.
- Unsupported Phase 4 schema.
- Duplicate or out-of-order decisions.
- Unreadable source or compressed video.
- Missing ffprobe.
- Output write failure.
- Configured fatal count mismatch.

Optional quality-metric unavailability does not terminate the pipeline.

## Report status

- `completed`: required validation completed without warnings.
- `completed_with_warnings`: required validation completed, but optional quality or nonfatal consistency checks produced warnings.

The report never contains fabricated quality values or benchmark results.