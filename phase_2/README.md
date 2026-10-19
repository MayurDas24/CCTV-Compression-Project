# Phase 2: Perception-Aware Processing Layer

This directory contains the Phase 2 perception layer for the **Real-Time Intelligent CCTV Video Compression System**.

## 1. What Phase 2 Does

Phase 2 acts as the vision analysis engine for the CCTV compression pipeline. It processes uncompressed or ingested frames from Phase 1, runs lightweight deep-learning object detection, and calculates spatial-temporal importance metrics across all detected Regions of Interest (ROIs).

Key responsibilities:
- Configurable sampling of incoming high-FPS video frames.
- Object detection using Ultralytics YOLO models (supporting CPU and CUDA GPU).
- Heuristic perception scoring based on object class priority, box area, confidence, and central frame proximity.
- Region of Interest (ROI) bounding coordinate generation.
- Structured output logging into streaming `.jsonl` lines and summary execution metrics.

---

## 2. Architecture & Pipeline Flow