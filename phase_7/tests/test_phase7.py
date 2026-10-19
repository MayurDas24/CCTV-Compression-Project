"""
Unit and integration tests for CCTV Compression Phase 7.

These tests do not require a real RTSP camera and do not run the real
Phase 2-6 compression pipeline. They validate:

- configuration loading;
- safe source redaction;
- reliable FFprobe frame-count fallback behavior;
- optional missing-camera handling;
- probe-only scheduling;
- external-command placeholder expansion;
- explicit artifact and JSON-report validation;
- rejection of missing artifacts;
- explicit, sequential real-pipeline mode;
- honest separation of pipeline and compression success;
- smoke-test report generation.

Run from the project root:

    python -m unittest discover \
        -s phase_7/tests \
        -p "test_*.py" \
        -v
"""

from __future__ import annotations

import json
import logging
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


PHASE7_DIRECTORY = Path(__file__).resolve().parents[1]

if str(PHASE7_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PHASE7_DIRECTORY))


import phase7_pipeline
from camera_manager import CameraManager
from health_monitor import (
    HealthMonitor,
    ReconnectPolicy,
)
from phase7_pipeline import (
    Phase7Application,
    load_config,
)
from processing_scheduler import (
    CaptureConfig,
    CommandAdapterConfig,
    ProcessingConfig,
    ProcessingMode,
    ProcessingScheduler,
    SchedulerConfig,
    SuccessValidationConfig,
)
from resource_monitor import ResourceMonitor
from stream_manager import (
    MediaProbe,
    SourceType,
    StreamDefinition,
    StreamManager,
    probe_media,
    redact_source,
)


def _successful_probe() -> MediaProbe:
    """Return deterministic successful probe data."""

    return MediaProbe(
        width=3840,
        height=2160,
        codec_name="h264",
        pixel_format="yuv420p",
        average_fps=29.97002997002997,
        duration_seconds=17.6176,
        frame_count=528,
        frame_count_source="ffprobe_count_frames",
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        size_bytes=63_469_839,
        bit_rate_bps=28_820_000,
        has_audio=True,
        probed_at_utc="2026-01-01T00:00:00+00:00",
    )


def _scheduler_config(
    *,
    maximum_workers: int = 1,
) -> SchedulerConfig:
    """Build a scheduler configuration suitable for tests."""

    return SchedulerConfig(
        max_concurrent_cameras=maximum_workers,
        queue_policy="round_robin",
        queue_poll_interval_seconds=0.05,
        fail_fast=False,
        cpu_admission_threshold_percent=100.0,
        memory_admission_threshold_percent=100.0,
        resource_admission_timeout_seconds=2.0,
    )


def _probe_only_processing_config(
    working_directory: Path,
) -> ProcessingConfig:
    """Build a probe-only processing configuration."""

    return ProcessingConfig(
        mode=ProcessingMode.PROBE_ONLY,
        command_adapter=CommandAdapterConfig(
            working_directory=working_directory,
            command=(),
            environment={},
            success_validation=SuccessValidationConfig(
                require_zero_exit_code=True,
                required_artifacts=(),
                report_json=None,
                status_fields=(),
                accepted_status_values=frozenset(),
            ),
            capture=CaptureConfig(
                save_stdout=True,
                save_stderr=True,
                maximum_in_memory_characters=10_000,
            ),
        ),
    )


def _external_processing_config(
    working_directory: Path,
    command: tuple[str, ...],
) -> ProcessingConfig:
    """Build a harmless external-command test configuration."""

    return ProcessingConfig(
        mode=ProcessingMode.EXTERNAL_PIPELINE,
        command_adapter=CommandAdapterConfig(
            working_directory=working_directory,
            command=command,
            environment={},
            success_validation=SuccessValidationConfig(
                require_zero_exit_code=True,
                required_artifacts=("artifact.bin",),
                report_json="report.json",
                status_fields=("status",),
                accepted_status_values=frozenset(
                    {"success"}
                ),
                compression_status_field=None,
            ),
            capture=CaptureConfig(
                save_stdout=True,
                save_stderr=True,
                maximum_in_memory_characters=10_000,
            ),
        ),
    )


def _run_scheduler(
    *,
    root: Path,
    definitions: tuple[StreamDefinition, ...],
    processing_config: ProcessingConfig,
    maximum_workers: int = 1,
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
]:
    """Run an isolated scheduler instance for a test."""

    shutdown_event = threading.Event()

    stream_registry = StreamManager(
        base_directory=root,
        ffprobe_path="ffprobe",
        probe_timeout_seconds=10.0,
        stop_event=shutdown_event,
    )

    camera_registry = CameraManager(
        stream_manager=stream_registry,
        run_output_directory=root / "run",
    )
    camera_registry.register_all(definitions)

    resource_monitor = ResourceMonitor(
        enabled=False,
        sample_interval_seconds=1.0,
        include_process_metrics=False,
        gpu_enabled=False,
        nvidia_smi_path="nvidia-smi",
        gpu_query_timeout_seconds=1.0,
        shutdown_event=shutdown_event,
    )

    reconnect_policy = ReconnectPolicy(
        enabled=False,
        eligible_source_types=frozenset(
            {SourceType.RTSP}
        ),
        maximum_attempts=0,
        initial_delay_seconds=0.0,
        backoff_multiplier=1.0,
        maximum_delay_seconds=0.0,
    )

    health_monitor = HealthMonitor(
        camera_manager=camera_registry,
        enabled=False,
        check_interval_seconds=1.0,
        default_stale_timeout_seconds=30.0,
        maximum_consecutive_health_failures=3,
        stale_detection_source_types=frozenset(
            {SourceType.RTSP}
        ),
        reconnect_policy=reconnect_policy,
        shutdown_event=shutdown_event,
    )

    scheduler = ProcessingScheduler(
        camera_manager=camera_registry,
        resource_monitor=resource_monitor,
        health_monitor=health_monitor,
        scheduler_config=_scheduler_config(
            maximum_workers=maximum_workers
        ),
        processing_config=processing_config,
        project_root=root,
        run_output_directory=root / "run",
        processing_timeout_seconds=20.0,
        shutdown_grace_seconds=2.0,
        shutdown_event=shutdown_event,
    )

    try:
        resource_monitor.start()
        health_monitor.start()

        result = scheduler.run()
        snapshots = camera_registry.snapshots()

        return result.to_dict(), snapshots
    finally:
        scheduler.close()
        health_monitor.close()
        resource_monitor.close()
        camera_registry.close()


class Phase7Tests(unittest.TestCase):
    """Focused Phase 7 orchestration tests."""

    def tearDown(self) -> None:
        """
        Close handlers created by application tests.

        Windows may otherwise keep temporary log files open and prevent
        TemporaryDirectory cleanup.
        """

        root_logger = logging.getLogger()

        for handler in list(root_logger.handlers):
            try:
                handler.flush()
                handler.close()
            finally:
                root_logger.removeHandler(handler)

    @unittest.skipIf(
        phase7_pipeline.yaml is None,
        "PyYAML is not installed",
    )
    def test_default_configuration_loads(self) -> None:
        config_path = (
            PHASE7_DIRECTORY / "config.yaml"
        )

        self.assertTrue(
            config_path.is_file(),
            f"Missing configuration file: {config_path}",
        )

        config = load_config(config_path)

        self.assertEqual(
            config.phase,
            "phase_7",
        )
        self.assertEqual(
            config.processing.mode,
            ProcessingMode.PROBE_ONLY,
        )
        self.assertEqual(
            config.scheduler.max_concurrent_cameras,
            1,
        )
        self.assertGreaterEqual(
            len(config.sources),
            1,
        )

        smoke_sources = [
            source
            for source in config.sources
            if "smoke-test" in source.tags
        ]

        self.assertGreaterEqual(
            len(smoke_sources),
            1,
        )
        self.assertEqual(
            smoke_sources[0].camera_id,
            "camera_01",
        )
        self.assertEqual(
            smoke_sources[0].source_type,
            SourceType.LOCAL_FILE,
        )

    def test_rtsp_credentials_are_redacted(self) -> None:
        source = (
            "rtsp://security-user:secret-password"
            "@192.0.2.10:554/live/main"
        )

        redacted = redact_source(source)

        self.assertNotIn(
            "security-user",
            redacted,
        )
        self.assertNotIn(
            "secret-password",
            redacted,
        )
        self.assertIn(
            "<redacted>@192.0.2.10:554",
            redacted,
        )

    def test_ffprobe_frame_count_fallback_is_used(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "video.mp4"
            source.write_bytes(b"not-a-real-video")

            normal_payload = {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 3840,
                        "height": 2160,
                        "codec_name": "h264",
                        "pix_fmt": "yuv420p",
                        "avg_frame_rate": "30000/1001",
                        "r_frame_rate": "30000/1001",
                        "duration": "17.617600",
                        "nb_frames": "N/A",
                        "bit_rate": "28820000",
                    },
                    {
                        "codec_type": "audio",
                        "codec_name": "aac",
                    },
                ],
                "format": {
                    "format_name": (
                        "mov,mp4,m4a,3gp,3g2,mj2"
                    ),
                    "duration": "17.617600",
                    "size": "63469839",
                    "bit_rate": "28820000",
                },
            }

            counted_payload = {
                "streams": [
                    {
                        "nb_read_frames": "528",
                    }
                ]
            }

            with patch(
                "stream_manager._run_ffprobe_json",
                side_effect=[
                    normal_payload,
                    counted_payload,
                ],
            ) as mocked_probe:
                result = probe_media(
                    source=str(source),
                    source_type=SourceType.LOCAL_FILE,
                    ffprobe_path="ffprobe",
                    timeout_seconds=10.0,
                    stop_event=threading.Event(),
                    count_frames_fallback=True,
                )

            self.assertEqual(
                mocked_probe.call_count,
                2,
            )
            self.assertEqual(
                result.frame_count,
                528,
            )
            self.assertEqual(
                result.frame_count_source,
                "ffprobe_count_frames",
            )
            self.assertAlmostEqual(
                result.average_fps or 0.0,
                30000 / 1001,
                places=8,
            )

    def test_frame_count_is_never_estimated(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "video.mp4"
            source.write_bytes(b"not-a-real-video")

            normal_payload = {
                "streams": [
                    {
                        "codec_type": "video",
                        "width": 1920,
                        "height": 1080,
                        "codec_name": "h264",
                        "pix_fmt": "yuv420p",
                        "avg_frame_rate": "30/1",
                        "duration": "10.0",
                        "nb_frames": "N/A",
                    }
                ],
                "format": {
                    "format_name": "mov,mp4",
                    "duration": "10.0",
                    "size": "1000",
                },
            }

            unavailable_count_payload = {
                "streams": [
                    {
                        "nb_read_frames": "N/A",
                    }
                ]
            }

            with patch(
                "stream_manager._run_ffprobe_json",
                side_effect=[
                    normal_payload,
                    unavailable_count_payload,
                ],
            ):
                result = probe_media(
                    source=str(source),
                    source_type=SourceType.LOCAL_FILE,
                    ffprobe_path="ffprobe",
                    timeout_seconds=10.0,
                    stop_event=threading.Event(),
                    count_frames_fallback=True,
                )

            # FPS multiplied by duration would be 300, but Phase 7
            # must never manufacture that value.
            self.assertIsNone(
                result.frame_count
            )
            self.assertIsNone(
                result.frame_count_source
            )

    def test_optional_missing_source_is_skipped(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)

            definition = StreamDefinition(
                camera_id="optional_camera",
                source_type=SourceType.LOCAL_FILE,
                source="missing.mp4",
                required=False,
                enabled=True,
            )

            scheduler_result, snapshots = (
                _run_scheduler(
                    root=root,
                    definitions=(definition,),
                    processing_config=(
                        _probe_only_processing_config(
                            root
                        )
                    ),
                    maximum_workers=1,
                )
            )

            self.assertTrue(
                scheduler_result["successful"]
            )
            self.assertEqual(
                scheduler_result["required_failures"],
                [],
            )
            self.assertEqual(
                len(snapshots),
                1,
            )
            self.assertEqual(
                snapshots[0]["status"],
                "skipped",
            )
            self.assertIsNone(
                snapshots[0]["processing_result"]
            )
            self.assertTrue(
                snapshots[0]["warnings"]
            )

    def test_probe_only_scheduler_does_not_claim_compression(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "camera.mp4"
            source.write_bytes(b"placeholder")

            definition = StreamDefinition(
                camera_id="camera_01",
                source_type=SourceType.LOCAL_FILE,
                source=str(source),
                required=True,
                enabled=True,
                tags=("smoke-test",),
            )

            with patch(
                "stream_manager.probe_media",
                return_value=_successful_probe(),
            ):
                scheduler_result, snapshots = (
                    _run_scheduler(
                        root=root,
                        definitions=(definition,),
                        processing_config=(
                            _probe_only_processing_config(
                                root
                            )
                        ),
                        maximum_workers=1,
                    )
                )

            self.assertTrue(
                scheduler_result["successful"]
            )
            self.assertEqual(
                snapshots[0]["status"],
                "completed",
            )

            processing = snapshots[0][
                "processing_result"
            ]

            self.assertIsInstance(
                processing,
                dict,
            )
            self.assertEqual(
                processing["mode"],
                "probe_only",
            )
            self.assertFalse(
                processing["processing_invoked"]
            )
            self.assertFalse(
                processing["compression_ran"]
            )
            self.assertFalse(
                processing["compression_success"]
            )

    def test_external_pipeline_requires_explicit_evidence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "camera.mp4"
            source.write_bytes(b"placeholder")

            definition = StreamDefinition(
                camera_id="camera_external",
                source_type=SourceType.LOCAL_FILE,
                source=str(source),
                required=True,
                enabled=True,
            )

            child_code = (
                "import json, pathlib, sys; "
                "output = pathlib.Path(sys.argv[1]); "
                "(output / 'artifact.bin').write_bytes(b'ok'); "
                "(output / 'report.json').write_text("
                "json.dumps(dict(status='success')), "
                "encoding='utf-8'); "
                "print('external pipeline finished')"
            )

            processing_config = (
                _external_processing_config(
                    root,
                    (
                        "{python_executable}",
                        "-c",
                        child_code,
                        "{camera_output_dir}",
                    ),
                )
            )

            with patch(
                "stream_manager.probe_media",
                return_value=_successful_probe(),
            ):
                scheduler_result, snapshots = (
                    _run_scheduler(
                        root=root,
                        definitions=(definition,),
                        processing_config=(
                            processing_config
                        ),
                        maximum_workers=1,
                    )
                )

            self.assertTrue(
                scheduler_result["successful"]
            )
            self.assertEqual(
                snapshots[0]["status"],
                "completed",
            )

            result = snapshots[0][
                "processing_result"
            ]

            self.assertIsInstance(
                result,
                dict,
            )
            self.assertTrue(
                result["process_started"]
            )
            self.assertTrue(
                result["processing_invoked"]
            )
            self.assertEqual(
                result["exit_code"],
                0,
            )
            self.assertTrue(
                result["pipeline_success_verified"]
            )

            # The generic child process proves the configured
            # artifact/report contract. It does not prove that
            # real video compression occurred.
            self.assertFalse(
                result["compression_success_verified"]
            )
            self.assertIn(
                "not separately verified",
                result[
                    "compression_verification_message"
                ],
            )

            artifacts = result[
                "required_artifacts"
            ]

            self.assertEqual(
                len(artifacts),
                1,
            )
            self.assertTrue(
                artifacts[0]["exists"]
            )
            self.assertTrue(
                artifacts[0]["nonempty"]
            )

    def test_missing_required_artifact_rejects_job(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "camera.mp4"
            source.write_bytes(b"placeholder")

            definition = StreamDefinition(
                camera_id="camera_missing_artifact",
                source_type=SourceType.LOCAL_FILE,
                source=str(source),
                required=True,
                enabled=True,
            )

            # This command exits successfully and writes a success
            # report, but deliberately does not create artifact.bin.
            child_code = (
                "import json, pathlib, sys; "
                "output = pathlib.Path(sys.argv[1]); "
                "(output / 'report.json').write_text("
                "json.dumps(dict(status='success')), "
                "encoding='utf-8')"
            )

            processing_config = (
                _external_processing_config(
                    root,
                    (
                        "{python_executable}",
                        "-c",
                        child_code,
                        "{camera_output_dir}",
                    ),
                )
            )

            with patch(
                "stream_manager.probe_media",
                return_value=_successful_probe(),
            ):
                scheduler_result, snapshots = (
                    _run_scheduler(
                        root=root,
                        definitions=(definition,),
                        processing_config=(
                            processing_config
                        ),
                        maximum_workers=1,
                    )
                )

            self.assertFalse(
                scheduler_result["successful"]
            )
            self.assertEqual(
                snapshots[0]["status"],
                "failed",
            )

            result = snapshots[0][
                "processing_result"
            ]

            self.assertIsInstance(
                result,
                dict,
            )
            self.assertEqual(
                result["exit_code"],
                0,
            )
            self.assertFalse(
                result["pipeline_success_verified"]
            )
            self.assertTrue(
                any(
                    "artifact" in error.lower()
                    and (
                        "missing" in error.lower()
                        or "empty" in error.lower()
                    )
                    for error in result["errors"]
                )
            )

    @unittest.skipIf(
        phase7_pipeline.yaml is None,
        "PyYAML is not installed",
    )
    def test_real_pipeline_mode_is_explicit_and_sequential(
        self,
    ) -> None:
        config = load_config(
            PHASE7_DIRECTORY / "config.yaml"
        )

        # Normal configuration loading must remain safe.
        self.assertEqual(
            config.processing.mode,
            ProcessingMode.PROBE_ONLY,
        )

        overridden = (
            phase7_pipeline._apply_runtime_overrides(
                config,
                smoke_test=False,
                selected_camera_ids=("camera_01",),
                force_probe_only=False,
                force_external_pipeline=True,
            )
        )

        self.assertEqual(
            overridden.processing.mode,
            ProcessingMode.EXTERNAL_PIPELINE,
        )
        self.assertEqual(
            overridden.scheduler.max_concurrent_cameras,
            1,
        )
        self.assertEqual(
            overridden.processing
            .command_adapter.command[0],
            "{python_executable}",
        )

        command_text = " ".join(
            overridden.processing
            .command_adapter.command
        )

        self.assertIn(
            "pipeline_adapter.py",
            command_text,
        )
        self.assertIn(
            "{camera_output_dir}",
            command_text,
        )

        validation = (
            overridden.processing
            .command_adapter.success_validation
        )

        self.assertEqual(
            validation.report_json,
            "pipeline_adapter_result.json",
        )
        self.assertEqual(
            validation.compression_status_field,
            "compression_success_verified",
        )
        self.assertIn(
            "final/compressed_video.mp4",
            validation.required_artifacts,
        )
        self.assertIn(
            "final/phase_6_final_report.json",
            validation.required_artifacts,
        )
        self.assertIn(
            "final/phase_6_final_report.txt",
            validation.required_artifacts,
        )
        self.assertIn(
            "final/phase_6_artifact_manifest.json",
            validation.required_artifacts,
        )

        self.assertTrue(
            (
                overridden.project_root
                / "phase_7"
                / "pipeline_adapter.py"
            ).is_file()
        )

    @unittest.skipIf(
        phase7_pipeline.yaml is None,
        "PyYAML is not installed",
    )
    def test_smoke_application_writes_honest_reports(
        self,
    ) -> None:
        source_config_path = (
            PHASE7_DIRECTORY / "config.yaml"
        )

        with tempfile.TemporaryDirectory() as temp:
            project_root = Path(temp)
            phase_directory = (
                project_root / "phase_7"
            )
            sample_directory = (
                project_root / "sample_videos"
            )

            phase_directory.mkdir()
            sample_directory.mkdir()

            test_video = (
                sample_directory / "test1.mp4"
            )
            test_video.write_bytes(b"placeholder")

            with source_config_path.open(
                "r",
                encoding="utf-8",
            ) as stream:
                raw_config = (
                    phase7_pipeline.yaml.safe_load(
                        stream
                    )
                )

            raw_config["paths"]["output_root"] = (
                "outputs"
            )
            raw_config["processing"]["mode"] = (
                "probe_only"
            )
            raw_config["resource_monitor"][
                "enabled"
            ] = False
            raw_config["resource_monitor"]["gpu"][
                "enabled"
            ] = False
            raw_config["health_monitor"][
                "enabled"
            ] = False
            raw_config["sources"] = [
                {
                    "camera_id": "camera_01",
                    "enabled": True,
                    "source_type": "local_file",
                    "source": (
                        "../sample_videos/test1.mp4"
                    ),
                    "required": True,
                    "display_name": (
                        "Smoke test camera"
                    ),
                    "tags": ["smoke-test"],
                }
            ]

            config_path = (
                phase_directory / "config.yaml"
            )

            with config_path.open(
                "w",
                encoding="utf-8",
                newline="\n",
            ) as stream:
                phase7_pipeline.yaml.safe_dump(
                    raw_config,
                    stream,
                    sort_keys=False,
                )

            config = load_config(config_path)

            with patch(
                "stream_manager.probe_media",
                return_value=_successful_probe(),
            ):
                application = Phase7Application(
                    config,
                    smoke_test=True,
                    selected_camera_ids=None,
                    force_probe_only=False,
                )
                exit_code = application.run()

            logging.shutdown()

            self.assertEqual(
                exit_code,
                0,
            )

            output_root = (
                phase_directory / "outputs"
            )
            run_directories = [
                path
                for path in output_root.iterdir()
                if path.is_dir()
            ]

            self.assertEqual(
                len(run_directories),
                1,
            )

            report_path = (
                run_directories[0]
                / "phase7_run_report.json"
            )
            text_report_path = (
                run_directories[0]
                / "phase7_run_report.txt"
            )

            self.assertTrue(
                report_path.is_file()
            )
            self.assertTrue(
                text_report_path.is_file()
            )

            with report_path.open(
                "r",
                encoding="utf-8",
            ) as stream:
                report = json.load(stream)

            self.assertTrue(
                report["successful"]
            )
            self.assertEqual(
                report["status"],
                "probe_only_completed",
            )
            self.assertEqual(
                report["mode"],
                "probe_only",
            )
            self.assertEqual(
                report["processing_summary"][
                    "compression_success_verified_camera_count"
                ],
                0,
            )
            self.assertEqual(
                report["cameras"][0]["status"],
                "completed",
            )
            self.assertFalse(
                report["cameras"][0][
                    "processing_result"
                ]["compression_success"]
            )

            text_report = (
                text_report_path.read_text(
                    encoding="utf-8"
                )
            )

            self.assertIn(
                "Compression was not run or claimed.",
                text_report,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
