# Phase 4 — Temporal-Preserving Intelligent Compression

Phase 4 compresses the source CCTV video with different HEVC quality levels while preserving every source frame, the original frame order, source resolution, source frame rate, and duration.

Phase 3 records are sampled decisions, not deletion commands. A decision at frame `N` becomes the temporal quality signal from frame `N` up to, but not including, the next sampled frame. The final decision continues through the end of the source video.

The `retain` and `drop` fields remain available for reporting, but Phase 4 never physically removes frames.

## Why the previous output was about 2 FPS

The previous implementation encoded only the roughly 35 sampled records marked `retain` from a source containing about 526 physical frames.

Spreading those 35 frames over the original 17.6-second duration produced roughly 2 FPS and visible stuttering.

The revised implementation reads the same decisions as quality signals and encodes all roughly 526 physical source frames.

## Files

- `frame_selector.py` streams and validates Phase 3 JSONL records, reports malformed records and sampling gaps, and converts decisions into complete temporal intervals.
- `video_encoder.py` probes media, encodes exact frame intervals with priority-specific CRF values, concatenates the encoded intervals, remuxes source audio, and validates the result before publishing it.
- `compression_metrics.py` separates Phase 3 decision statistics from actual output-frame statistics and builds the Phase 5-compatible summary.
- `phase4_pipeline.py` loads configuration and coordinates the complete pipeline.
- `config.yaml` contains paths, CRF values, temporal safety requirements, validation tolerances, and runtime settings.

## Input contract

The Phase 3 JSONL file must contain one JSON object per analyzed frame.

Example:

```json
{"schema_version":"1.0","phase":"phase_3_frame_decision","frame_index":0,"timestamp_seconds":0.0,"importance_score":0.91,"decision":"retain","priority":"critical","reason":"person detected"}
```

Supported priorities are:

- `critical`
- `high`
- `medium`
- `low`

Records are sorted by `frame_index`.

Duplicate indices, invalid values, decisions outside the source frame range, and non-monotonic timestamps are rejected.

When `validation.fail_on_malformed_decisions` is `true`, any malformed JSONL record stops the run.

The first decision must describe source frame `0`.

## Temporal interpretation

If Phase 3 contains decisions for frames:

```text
0
15
30
45
```

Phase 4 interprets them as:

```text
frame 0 decision  -> frames 0 through 14
frame 15 decision -> frames 15 through 29
frame 30 decision -> frames 30 through 44
frame 45 decision -> frame 45 through the end
```

Every physical source frame belongs to exactly one interval.

A Phase 3 `drop` decision means that the interval receives stronger compression. It does not mean the source frames in that interval are removed.

## Encoding architecture

Each coalesced priority interval is encoded independently with FFmpeg's `libx265` encoder and the CRF configured for that priority.

Exact `trim=start_frame:end_frame` filters define the interval boundaries.

Each temporary MPEG-TS segment uses the same:

- source frame rate;
- source resolution;
- frame ordering;
- `libx265` codec;
- `yuv420p` pixel format;
- constant frame cadence.

The compatible segments are concatenated without another video re-encode. The original audio stream is copied or converted to AAC according to the configuration.

This architecture applies temporal quality without dropping, duplicating, or interpolating source frames. It does not change MP4 metadata to create artificial smoothness.

## CRF configuration

Default mapping:

| Priority | CRF | Intended quality |
|---|---:|---|
| critical | 22 | highest |
| high | 25 | high |
| medium | 28 | normal |
| low | 32 | strongest compression |

Lower CRF values produce higher visual quality and larger files.

Higher CRF values produce stronger compression and smaller files.

All CRF values are configurable from `0` through `51`.

## Requirements

- Python 3.11 or newer
- FFmpeg and FFprobe available on `PATH`
- FFmpeg compiled with the `libx265` encoder
- The Python dependency listed in `requirements.txt`

Verify FFmpeg:

```bash
ffmpeg -version
ffprobe -version
ffmpeg -hide_banner -encoders
```

The encoder list must contain `libx265`.

Install the Python dependency:

```bash
python -m pip install -r requirements.txt
```

If FFmpeg is not on `PATH`, set absolute Windows paths in `config.yaml`:

```yaml
runtime:
  ffmpeg_path: "C:/ffmpeg/bin/ffmpeg.exe"
  ffprobe_path: "C:/ffmpeg/bin/ffprobe.exe"
```

## Project layout

Expected layout:

```text
CCTV_Compression/
├── phase_3/
│   └── outputs/
│       └── frame_decisions.jsonl
├── phase_4/
│   ├── frame_selector.py
│   ├── video_encoder.py
│   ├── compression_metrics.py
│   ├── phase4_pipeline.py
│   ├── config.yaml
│   ├── requirements.txt
│   └── outputs/
└── sample_videos/
    └── test1.mp4
```

## Running Phase 4

From the `phase_4` directory:

```bash
python phase4_pipeline.py --config config.yaml
```

The pipeline writes:

```text
phase_4/outputs/compressed_video.mp4
phase_4/outputs/compression_summary.json
```

Temporary encoded segments are deleted after successful or failed execution unless:

```yaml
runtime:
  keep_temporary_files: true
```

## Output validation

Before replacing the public output video, the pipeline verifies:

- output resolution equals source resolution;
- output frame rate is within `validation.fps_tolerance`;
- output duration is within `validation.duration_tolerance_seconds`;
- output frame count exactly equals source frame count.

Inspect the source metadata:

```bash
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,codec_name,pix_fmt,r_frame_rate,avg_frame_rate,duration,nb_frames,nb_read_frames -of json ../sample_videos/test1.mp4
```

Inspect the compressed output:

```bash
ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=width,height,codec_name,pix_fmt,r_frame_rate,avg_frame_rate,duration,nb_frames,nb_read_frames -of json outputs/compressed_video.mp4
```

For the described test video, the expected result is approximately:

```text
Resolution: 3840x2160
FPS:        29.97
Duration:   17.6 seconds
Frames:     526
Codec:      HEVC/H.265
```

Small timestamp differences within the configured tolerances are acceptable. An approximately 2 FPS output is not acceptable.

## Compression summary

`compression_summary.json` distinguishes Phase 3 decision records from physical source and output frames.

Important fields include:

```text
decisions.frames_analyzed
decisions.frames_retained
decisions.frames_dropped

frame_reduction.source_frame_count
frame_reduction.output_frame_count
frame_reduction.actual_physically_removed_frames

validation.absolute_fps_difference
validation.absolute_duration_difference_seconds
validation.all_source_frames_preserved
```

For this implementation:

```text
actual_physically_removed_frames = 0
```

The number of Phase 3 retained decisions must not be interpreted as the number of frames in the compressed video.

## Configuration notes

The required temporal safety configuration is:

```yaml
temporal:
  preserve_source_fps: true
  preserve_duration: true
  physically_drop_frames: false
```

Other useful options:

- Set `audio.enabled: false` to create video-only output.
- `audio.mode: copy` preserves the original audio codec when MP4 supports it.
- Use `audio.mode: aac` when the source audio codec cannot be copied into MP4.
- `runtime.overwrite: false` prevents replacement of existing output files.
- Output video and summary paths must remain inside the Phase 4 `outputs` directory.

## Known limitations

- Exact frame trimming can require decoding the source again for each quality segment. Many rapidly changing priorities therefore increase encoding time.
- Segment boundaries start independently decodable HEVC groups of pictures and can add modest storage overhead.
- The output preserves the source average frame cadence as a constant frame rate. Unusual variable-frame-rate input should be normalized before Phase 4.
- Direct audio copying depends on MP4 codec compatibility. Select AAC mode if audio copying fails.
- CRF controls perceptual quality and does not guarantee a specific final file size.

## Relationship to Phase 3 and Phase 5

Phase 3 supplies sampled temporal importance decisions.

Phase 4 converts those decisions into quality intervals, preserves all physical source frames, and writes:

```text
compressed_video.mp4
compression_summary.json
```

Phase 5 can then measure quality, integrity, storage, and forensic properties using those two Phase 4 artifacts.

Phase 4 does not modify Phase 3 data and does not perform Phase 5 validation.