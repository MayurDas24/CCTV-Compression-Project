# Phase 0 — Baseline & Feasibility Setup

This module establishes a baseline metric evaluation pipeline comparing standard global video compression against ROI-aware spatial pre-filtered adaptive video compression.

## Prerequisites
* Python 3.11+
* FFmpeg & FFprobe system binaries accessible via environment `PATH`.

## Installation & Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt