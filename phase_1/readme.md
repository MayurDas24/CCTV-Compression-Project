# Phase 1 — Real-Time CCTV Ingestion & Encoding Pipeline

Phase 1 provides a non-blocking ingestion and encoding foundation for streaming RTSP sources.

## Architecture
```text
RTSP Camera -> Ingestion Manager (Queue Thread) -> Core Real-time Engine (Pipe) -> FFmpeg (H.265)