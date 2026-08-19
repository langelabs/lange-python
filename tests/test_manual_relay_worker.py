"""Tests for the authenticated manual relay worker."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def test_manual_relay_worker_loads_adjacent_secret_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load credentials from the adjacent dotenv file and pass them to the worker.
    
    :param monkeypatch: Pytest patch helper used to isolate the manual runner.
    """
    module_path = Path(__file__).parent / "manual" / "run_relay_worker.py"
    spec = importlib.util.spec_from_file_location("manual_relay_worker", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded_paths: list[Path] = []
    worker_arguments: list[dict[str, object]] = []

    class FakeWorker:
        """Capture worker configuration without opening a connection."""

        remote_relay_address: str | None = None

        def __init__(self, **kwargs: object) -> None:
            """Record the worker constructor arguments.

            :param kwargs: Worker configuration under test.
            """
            worker_arguments.append(kwargs)

        def start(self) -> None:
            """Simulate worker startup."""

        def is_alive(self) -> bool:
            """Stop the runner loop immediately.

            :returns: Always ``False``.
            """
            return False

        async def stop(self) -> None:
            """Simulate idempotent worker shutdown."""

        def join(self, timeout: float | None = None) -> None:
            """Simulate joining the worker thread.

            :param timeout: Optional join timeout.
            """

    def load_test_dotenv(path: Path) -> bool:
        """Populate test credentials as if they came from dotenv.

        :param path: Dotenv path selected by the runner.
        :returns: Always ``True``.
        """
        loaded_paths.append(path)
        monkeypatch.setenv("LANGE_LABS_API_KEY", "test-secret")
        monkeypatch.setenv("MESH_PROJECT_ID", "00000000-0000-0000-0000-000000000001")
        monkeypatch.setenv("MESH_RELAY_TARGET", "http://localhost:3000")
        return True

    monkeypatch.setattr(module, "load_dotenv", load_test_dotenv)
    monkeypatch.setattr(module, "MeshWorker", FakeWorker)

    module.main()

    assert loaded_paths == [Path(module.__file__).with_name(".env")]
    assert len(worker_arguments) == 1
    arguments = worker_arguments[0]
    assert arguments["project_id"] == "00000000-0000-0000-0000-000000000001"
    assert arguments["remote_base_url"] == "wss://mesh.lange-labs.com"
    assert arguments["api_key"] == "test-secret"
    plugins = arguments["plugins"]
    assert isinstance(plugins, list)
    assert len(plugins) == 1
    assert isinstance(plugins[0], module.MeshRelayPlugin)
    assert plugins[0].relay_target == "http://localhost:3000"
