"""
Phase 7: multi-source orchestration for CCTV_Compression.

Phase 7 coordinates independent video sources without importing or modifying
the internal implementation of Phases 3 through 6. Existing pipeline execution
is exposed through an isolated, configuration-driven subprocess adapter.

The default configuration uses probe-only mode. Probe-only results confirm
that orchestration, scheduling, media probing, health monitoring, and reporting
worked; they do not claim that compression was performed.
"""

from __future__ import annotations

PHASE_NAME = "phase_7"
SCHEMA_VERSION = "1.0"
__version__ = "1.0.0"

__all__ = [
    "PHASE_NAME",
    "SCHEMA_VERSION",
    "__version__",
]