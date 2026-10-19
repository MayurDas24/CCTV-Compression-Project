"""
Metadata writer for Phase 2 perception outputs.

Writes:
    1. Per-frame metadata as JSON Lines (.jsonl)
    2. Final processing summary as JSON (.json)

The implementation is designed to be:
    - deterministic
    - UTF-8 compatible
    - safe for repeated runs
    - easy to consume downstream
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class MetadataWriter:
    """
    Writes perception metadata and final run summaries.

    Per-frame records are stored as JSON Lines:

        outputs/perception.jsonl

    Each line represents one sampled frame.

    The final summary is stored as standard JSON:

        outputs/summary.json
    """

    def __init__(
        self,
        metadata_file: str = "outputs/perception.jsonl",
        summary_file: str = "outputs/summary.json",
    ) -> None:
        """
        Initialize metadata writer.

        Args:
            metadata_file:
                Destination for per-frame JSONL records.

            summary_file:
                Destination for final summary JSON.
        """

        self.metadata_path = Path(metadata_file)
        self.summary_path = Path(summary_file)

        self._metadata_file: Optional[Any] = None
        self._closed = False
        self._records_written = 0

        self._prepare_directories()
        self._open_metadata_file()

    def _prepare_directories(self) -> None:
        """Create parent directories for output files."""

        self.metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def _open_metadata_file(self) -> None:
        """Open JSONL metadata file for writing."""

        try:
            self._metadata_file = self.metadata_path.open(
                mode="w",
                encoding="utf-8",
            )

            logger.info(
                "Metadata output initialized: %s",
                self.metadata_path,
            )

        except OSError as err:
            logger.error(
                "Failed to open metadata output '%s': %s",
                self.metadata_path,
                err,
            )
            raise RuntimeError(
                f"Could not open metadata output: {self.metadata_path}"
            ) from err

    @staticmethod
    def _sanitize_for_json(value: Any) -> Any:
        """
        Convert common non-JSON-native values into JSON-compatible types.

        Handles:
            - numpy integer/float/bool
            - dictionaries
            - lists/tuples
            - dataclasses exposing to_dict()
        """

        # Objects such as ROI and Detection may provide to_dict().
        if hasattr(value, "to_dict") and callable(value.to_dict):
            return MetadataWriter._sanitize_for_json(value.to_dict())

        # Avoid importing numpy unless necessary.
        try:
            import numpy as np

            if isinstance(value, np.integer):
                return int(value)

            if isinstance(value, np.floating):
                return float(value)

            if isinstance(value, np.bool_):
                return bool(value)

            if isinstance(value, np.ndarray):
                return value.tolist()

        except ImportError:
            pass

        if isinstance(value, dict):
            return {
                str(key): MetadataWriter._sanitize_for_json(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                MetadataWriter._sanitize_for_json(item)
                for item in value
            ]

        if isinstance(value, Path):
            return str(value)

        return value

    def write_record(self, record: Dict[str, Any]) -> None:
        """
        Write one perception frame record to JSONL.

        Args:
            record:
                Dictionary containing metadata for a sampled frame.
        """

        if self._closed:
            raise RuntimeError(
                "Cannot write metadata after MetadataWriter has been closed."
            )

        if not isinstance(record, dict):
            raise TypeError(
                f"Metadata record must be a dictionary, "
                f"got {type(record).__name__}"
            )

        if self._metadata_file is None:
            raise RuntimeError("Metadata file is not open.")

        try:
            sanitized_record = self._sanitize_for_json(record)

            json_line = json.dumps(
                sanitized_record,
                ensure_ascii=False,
                separators=(",", ":"),
            )

            self._metadata_file.write(json_line)
            self._metadata_file.write("\n")

            # Flush periodically so metadata is not lost if the process
            # is interrupted during a long-running CCTV stream.
            self._records_written += 1

            if self._records_written % 10 == 0:
                self._metadata_file.flush()

        except (TypeError, ValueError) as err:
            logger.error(
                "Failed to serialize metadata record #%d: %s",
                self._records_written + 1,
                err,
            )
            raise

        except OSError as err:
            logger.error(
                "Failed to write metadata record #%d: %s",
                self._records_written + 1,
                err,
            )
            raise RuntimeError(
                "Failed to write perception metadata."
            ) from err

    def write_summary(self, summary: Dict[str, Any]) -> None:
        """
        Write final Phase 2 summary as formatted JSON.

        Args:
            summary:
                Dictionary containing run-level statistics.
        """

        if not isinstance(summary, dict):
            raise TypeError(
                f"Summary must be a dictionary, "
                f"got {type(summary).__name__}"
            )

        try:
            sanitized_summary = self._sanitize_for_json(summary)

            # Write to temporary file first.
            #
            # This prevents leaving a partially written summary.json
            # if the process is interrupted while writing.
            temporary_path = self.summary_path.with_suffix(
                self.summary_path.suffix + ".tmp"
            )

            with temporary_path.open(
                mode="w",
                encoding="utf-8",
            ) as summary_file:

                json.dump(
                    sanitized_summary,
                    summary_file,
                    ensure_ascii=False,
                    indent=2,
                )

                summary_file.write("\n")
                summary_file.flush()

            # Replace old summary atomically where supported.
            temporary_path.replace(self.summary_path)

            logger.info(
                "Summary written successfully: %s",
                self.summary_path,
            )

        except (TypeError, ValueError) as err:
            logger.error(
                "Failed to serialize summary: %s",
                err,
            )
            raise

        except OSError as err:
            logger.error(
                "Failed to write summary '%s': %s",
                self.summary_path,
                err,
            )
            raise RuntimeError(
                f"Could not write summary: {self.summary_path}"
            ) from err

    def flush(self) -> None:
        """Flush pending metadata to disk."""

        if self._metadata_file is not None:
            self._metadata_file.flush()

    def close(self) -> None:
        """
        Close the metadata output file.

        Safe to call multiple times.
        """

        if self._closed:
            return

        try:
            if self._metadata_file is not None:
                self._metadata_file.flush()
                self._metadata_file.close()

            logger.info(
                "Metadata writer closed. Records written: %d",
                self._records_written,
            )

        finally:
            self._metadata_file = None
            self._closed = True

    def __enter__(self) -> "MetadataWriter":
        """Support context-manager usage."""

        return self

    def __exit__(
        self,
        exc_type: Any,
        exc_value: Any,
        traceback: Any,
    ) -> None:
        """Close writer when leaving context manager."""

        self.close()

    @property
    def records_written(self) -> int:
        """Return number of frame records written."""

        return self._records_written

    @property
    def is_closed(self) -> bool:
        """Return whether the writer has been closed."""

        return self._closed