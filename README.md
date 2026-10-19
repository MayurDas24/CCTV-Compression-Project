\# CCTV Compression Project



\### AI-Driven Video Compression with Computer Vision



An intelligent CCTV video-compression pipeline that combines object detection, frame-importance analysis, adaptive quality encoding, and validation to reduce storage requirements while preserving useful visual information.



\*\*Built with:\*\* Python · OpenCV · Ultralytics YOLOv8 · FFmpeg · HEVC/H.265



\---



\## Overview



CCTV systems generate large volumes of video data, creating challenges around storage capacity, retention periods, and processing costs.



This project explores a computer-vision-assisted approach to video compression. It analyzes sampled video frames, estimates their importance, applies segment-based encoding, and validates the resulting video and compression metrics.



The goal is to reduce video storage requirements while maintaining good visual quality and preserving the video timeline.



\## Key Features



\- \*\*Computer vision:\*\* Uses YOLOv8 for object detection on sampled video frames.

\- \*\*Frame-importance analysis:\*\* Produces importance and retention decisions to inform downstream processing.

\- \*\*Segment-based encoding:\*\* Uses FFmpeg and HEVC/H.265 encoding to apply different quality settings across video segments.

\- \*\*Quality validation:\*\* Evaluates video properties and objective quality metrics when source and compressed timelines can be aligned.

\- \*\*Integration checks:\*\* Consolidates phase outputs and validates consistency before publishing final artifacts.

\- \*\*Pipeline orchestration:\*\* Provides a Phase 7 adapter for running the processing pipeline on local video files.

\- \*\*Webcam experiment:\*\* Tested the workflow on a short webcam recording.



\## Architecture



The project is organized into phases, with Phases 2–7 forming the demonstrated processing workflow.



| Phase | Responsibility |

|---|---|

| Phase 2 — Perception | Samples frames and runs YOLOv8 object detection. |

| Phase 3 — Importance Analysis | Generates frame-level importance and retention decisions. |

| Phase 4 — Compression | Selects quality segments and encodes the output video. |

| Phase 5 — Quality Validation | Checks video properties, compression metrics, and objective quality metrics when alignment is valid. |

| Phase 6 — Integration | Consolidates outputs, checks consistency, and publishes final artifacts when validation passes. |

| Phase 7 — Orchestration | Manages processing jobs and invokes the pipeline for local video inputs. |



\*\*Important design detail:\*\* Frame-importance decisions inform the compression strategy; the demonstrated Phase 4 implementation uses segment-based quality encoding rather than physically removing every frame marked unimportant.



\## Technology Stack



\- \*\*Language:\*\* Python

\- \*\*Computer vision:\*\* OpenCV, Ultralytics YOLOv8

\- \*\*Video processing:\*\* FFmpeg, FFprobe

\- \*\*Encoding:\*\* HEVC/H.265

\- \*\*Detection model:\*\* YOLOv8n

\- \*\*Orchestration:\*\* Python pipeline adapter



\## Repository Structure



```text

CCTV-Compression-Project/

├── phase\_2/                 # Perception and object detection

├── phase\_3/                 # Importance analysis and decisions

├── phase\_4/                 # Segment selection and video encoding

├── phase\_5/                 # Quality analysis and validation

├── phase\_6/                 # Integration and final artifacts

├── phase\_7/                 # Orchestration and pipeline adapter

├── sample\_videos/           # Local test inputs, if included

├── live\_test/               # Webcam experiment scripts, if included

├── README.md

└── requirements.txt         # Python dependencies, if provided

```



The exact files and optional folders may vary by project version.



\## Getting Started



\### Prerequisites



\- Windows, Linux, or macOS with a compatible Python environment

\- Python 3.11 recommended for the tested Windows setup

\- FFmpeg and FFprobe installed and accessible on `PATH`

\- The Python dependencies used by the project's individual phases

\- Sufficient CPU, memory, and disk space for video decoding and encoding



\### 1. Clone the repository



```bash

git clone https://github.com/MayurDas24/CCTV-Compression-Project.git

cd CCTV-Compression-Project

```



\### 2. Create a virtual environment



\*\*Windows PowerShell:\*\*



```powershell

python -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell blocks activation, use the Python executable inside the environment directly or follow your system's virtual-environment policy.



\### 3. Install dependencies



If the repository contains a complete `requirements.txt`:



```powershell

python -m pip install --upgrade pip

pip install -r requirements.txt

```



Install and configure FFmpeg separately if it is not already available. Verify the installation with:



```powershell

ffmpeg -version

ffprobe -version

```



\### 4. Validate the pipeline configuration



From the repository root:



```powershell

python .\\phase\_7\\phase7\_pipeline.py --validate-config

```



\### 5. Run the smoke test



```powershell

python .\\phase\_7\\phase7\_pipeline.py --smoke-test

```



\### 6. Process a local video



The Phase 7 adapter supports finite local-video processing. Example:



```powershell

python .\\phase\_7\\pipeline\_adapter.py `

&#x20; --project-root "." `

&#x20; --source ".\\sample\_videos\\test1.mp4" `

&#x20; --camera-id "test1\_run" `

&#x20; --camera-output-dir ".\\phase\_7\\outputs\\test1\_run"

```



Replace the source path with a video available on your machine.



Check the generated `pipeline\_adapter\_result.json`, phase reports, and final output directory to confirm whether the run completed successfully.



\## Experimental Results



The following measurements were observed in local test runs. They are illustrative results from specific inputs, not guaranteed performance for every video.



| Test | Original size | Compressed size | Storage reduction |

|---|---:|---:|---:|

| Test 1 | 63.47 MB | 19.02 MB | 70.03% |

| Test 3 | 157.02 MB | 109.17 MB | 30.48% |

| Webcam recording — 30 seconds | 4.18 MB | 1.32 MB | Approximately 68.3% |



\### Test 1 — Objective quality metrics



| Metric | Observed value |

|---|---:|

| PSNR | 43.67 dB |

| SSIM | 0.9844 |

| VMAF | 96.71 |



The Test 1 output was played back and visually inspected. The observed metrics and playback supported good visual-quality retention for that input.



\*\*Test 3 status:\*\* The compression stage generated an output, but final integration encountered a frame-count consistency mismatch. Its compression figures above describe the Phase 4 output and should not be interpreted as a successful end-to-end Phase 6 run.



\## Output Artifacts



Depending on the run and phase, generated artifacts may include:



\- `perception.jsonl` and `summary.json`

\- `frame\_decisions.jsonl`

\- Compressed video output

\- Compression and quality reports

\- Phase 6 integration report and artifact manifest

\- Phase 7 pipeline adapter result and execution logs



Generated files are typically stored under the corresponding phase's `outputs` directory. Consult the run-specific report for the authoritative status and paths.



\## Limitations and Future Work



This project is a prototype and has not been validated as a production-ready, continuously operating CCTV deployment.



Current limitations include:



\- Real RTSP camera streaming and reconnection behavior have not been comprehensively validated.

\- Long-duration, multi-camera operation and production resource behavior require further testing.

\- Detection accuracy after compression has not been comprehensively benchmarked against the original video.

\- Objective quality metrics depend on valid source-to-output timeline alignment.

\- A frame-count consistency issue remains unresolved for one Test 3 integration run.

\- Compression savings depend on source content, resolution, bitrate, and encoding settings.



Potential next steps include testing more camera feeds, measuring downstream detection accuracy, improving runtime performance, and expanding automated regression tests.



\## Reproducibility



For meaningful comparisons, record the source video, source and output sizes, encoding settings, output duration, frame counts, quality metrics, and the pipeline's final validation status.



Only report an end-to-end test as successful when its final integration checks pass.



\## Author



\*\*Mayur R Das\*\*



GitHub: \[@MayurDas24](https://github.com/MayurDas24)



\## License



Add a license file before describing this project as open source. Until then, no license is specified.



