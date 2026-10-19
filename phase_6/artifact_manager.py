"""Atomic copying and manifest generation for Phase 6 deliverables."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


class ArtifactError(RuntimeError):
    """Raised when a final artifact cannot be safely published."""


@dataclass(frozen=True, slots=True)
class ArtifactRecord:
    path: Path
    artifact_type: str
    source_phase: str
    size_bytes: int | None
    sha256: str | None
    copied_from: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "type": self.artifact_type,
            "source_phase": self.source_phase,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "copied_from": (
                str(self.copied_from)
                if self.copied_from
                else None
            ),
        }


class ArtifactManager:
    """Publish byte-identical copies without moving source files."""

    def __init__(
        self,
        output_directory: Path,
        *,
        overwrite: bool,
    ) -> None:
        self.output_directory = (
            output_directory.expanduser().resolve()
        )
        self.overwrite = overwrite

    def prepare(self) -> None:
        try:
            self.output_directory.mkdir(
                parents=True,
                exist_ok=True,
            )
        except OSError as exc:
            raise ArtifactError(
                "Unable to create output directory "
                f"{self.output_directory}: {exc}"
            ) from exc

    def copy_video(
        self,
        source: Path,
        destination: Path,
    ) -> ArtifactRecord:
        source_path = _required_file(
            source,
            "Phase 4 compressed video",
        )
        target = self._validate_destination(destination)

        if target.exists() and not self.overwrite:
            raise ArtifactError(
                "Artifact exists and overwrite is disabled: "
                f"{target}"
            )

        temporary = _temporary_sibling(target)

        try:
            shutil.copy2(source_path, temporary)

            if (
                temporary.stat().st_size
                != source_path.stat().st_size
            ):
                raise ArtifactError(
                    "Copied video size differs from its source"
                )

            source_hash = sha256_file(source_path)
            copied_hash = sha256_file(temporary)

            if source_hash != copied_hash:
                raise ArtifactError(
                    "Copied video checksum differs from its source"
                )

            os.replace(temporary, target)

            return ArtifactRecord(
                path=target,
                artifact_type="compressed_video",
                source_phase="phase_4",
                size_bytes=target.stat().st_size,
                sha256=copied_hash,
                copied_from=source_path,
            )
        except (OSError, ArtifactError) as exc:
            _remove_if_present(temporary)

            if isinstance(exc, ArtifactError):
                raise

            raise ArtifactError(
                f"Unable to publish final video: {exc}"
            ) from exc

    def describe_reference(
        self,
        path: Path,
        artifact_type: str,
        source_phase: str,
    ) -> ArtifactRecord:
        resolved = _required_file(
            path,
            artifact_type,
        )

        return ArtifactRecord(
            path=resolved,
            artifact_type=artifact_type,
            source_phase=source_phase,
            size_bytes=resolved.stat().st_size,
            sha256=sha256_file(resolved),
        )

    def describe_output(
        self,
        path: Path,
        artifact_type: str,
        source_phase: str = "phase_6",
    ) -> ArtifactRecord:
        resolved = _required_file(
            path,
            artifact_type,
        )

        return ArtifactRecord(
            path=resolved,
            artifact_type=artifact_type,
            source_phase=source_phase,
            size_bytes=resolved.stat().st_size,
            sha256=sha256_file(resolved),
        )

    def write_manifest(
        self,
        destination: Path,
        artifacts: Iterable[ArtifactRecord],
    ) -> ArtifactRecord:
        target = self._validate_destination(destination)

        payload = {
            "schema_version": "1.0",
            "phase": "phase_6_final_integration",
            "artifacts": [
                artifact.to_dict()
                for artifact in artifacts
            ],
        }

        _write_json_atomic(
            target,
            payload,
            overwrite=self.overwrite,
        )

        return self.describe_output(
            target,
            "artifact_manifest",
        )

    def _validate_destination(
        self,
        destination: Path,
    ) -> Path:
        target = destination.expanduser().resolve()

        if not target.is_relative_to(
            self.output_directory
        ):
            raise ArtifactError(
                "Final artifact must be inside "
                f"{self.output_directory}: {target}"
            )

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        return target


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    try:
        with path.open("rb") as stream:
            while chunk := stream.read(chunk_size):
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactError(
            f"Unable to hash artifact {path}: {exc}"
        ) from exc

    return digest.hexdigest()


def _write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    overwrite: bool,
) -> None:
    if path.exists() and not overwrite:
        raise ArtifactError(
            "Artifact exists and overwrite is disabled: "
            f"{path}"
        )

    temporary = _temporary_sibling(path)

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            json.dump(
                payload,
                stream,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        _remove_if_present(temporary)

        raise ArtifactError(
            f"Unable to write manifest {path}: {exc}"
        ) from exc


def _temporary_sibling(path: Path) -> Path:
    handle, name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(handle)

    return Path(name)


def _required_file(
    path: Path,
    label: str,
) -> Path:
    resolved = path.expanduser().resolve()

    if (
        not resolved.is_file()
        or resolved.stat().st_size <= 0
    ):
        raise ArtifactError(
            f"Required {label} is missing or empty: {resolved}"
        )

    return resolved


def _remove_if_present(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
