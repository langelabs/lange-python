"""Tests for AI inference mesh plugins."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from lange.ai import MeshAiPlugin
from lange.ai.contracts import AiModelConfig
from lange.mesh import MeshWorker


def model_config(alias: str) -> AiModelConfig:
    """Create a minimal AI model configuration.

    :param alias: Model alias.
    :returns: Minimal model configuration.
    """
    return AiModelConfig(
        model_name="example/model",
        model_alias=alias,
        model_type="LLM",
        registration=None,
    )


class FakeInferenceServer:
    """Record inference server lifecycle calls."""

    def __init__(self, port: int, events: list[str]) -> None:
        """Create a fake server.

        :param port: Assigned server port.
        :param events: Shared event collection.
        """
        self.port = port
        self.events = events

    def start(self) -> None:
        """Record server startup."""
        self.events.append(f"server-start:{self.port}")

    def stop(self) -> None:
        """Record server shutdown."""
        self.events.append(f"server-stop:{self.port}")

    def join(self, timeout: float | None = None) -> None:
        """Record server joining.

        :param timeout: Optional join timeout.
        """
        self.events.append(f"server-join:{self.port}")


class FakeWorkerThread:
    """Avoid opening a real worker connection in plugin lifecycle tests."""

    def __init__(self, **kwargs: Any) -> None:
        """Create a fake worker thread.

        :param kwargs: Ignored thread configuration.
        """
        self.alive = False

    def start(self) -> None:
        """Mark the fake thread alive."""
        self.alive = True

    def is_alive(self) -> bool:
        """Return fake liveness.

        :returns: Whether the fake thread is alive.
        """
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        """Mark the fake thread stopped.

        :param timeout: Optional join timeout.
        """
        self.alive = False


def test_ai_plugins_assign_sequential_ports_and_manage_servers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assign default ports by AI plugin order and manage each server."""
    events: list[str] = []
    created_ports: list[int] = []

    def create_inference_server(
        model: AiModelConfig,
        *,
        host: str,
        port: int,
    ) -> FakeInferenceServer:
        """Create a fake server for the requested port.

        :param model: Model served by the fake.
        :param host: Server bind host.
        :param port: Server bind port.
        :returns: Fake inference server.
        """
        assert host == "127.0.0.1"
        created_ports.append(port)
        return FakeInferenceServer(port, events)

    monkeypatch.setattr(
        "lange.ai.plugin.create_inference_server",
        create_inference_server,
        raising=False,
    )
    monkeypatch.setattr("lange.mesh.worker.threading.Thread", FakeWorkerThread)
    worker = MeshWorker(
        plugins=[
            MeshAiPlugin(model_config("first")),
            MeshAiPlugin(model_config("second")),
        ],
    )

    worker.start()
    monkeypatch.undo()
    asyncio.run(worker.stop())

    assert created_ports == [8500, 8501]
    assert events == [
        "server-start:8500",
        "server-start:8501",
        "server-stop:8501",
        "server-join:8501",
        "server-stop:8500",
        "server-join:8500",
    ]


def test_worker_rejects_duplicate_resolved_ai_ports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject explicit AI ports that collide with assigned default ports."""
    starts: list[int] = []

    def create_inference_server(
        model: AiModelConfig,
        *,
        host: str,
        port: int,
    ) -> FakeInferenceServer:
        """Record an unexpected server creation.

        :param model: Model served by the fake.
        :param host: Server bind host.
        :param port: Server bind port.
        :returns: Fake inference server.
        """
        starts.append(port)
        return FakeInferenceServer(port, [])

    monkeypatch.setattr(
        "lange.ai.plugin.create_inference_server",
        create_inference_server,
        raising=False,
    )
    worker = MeshWorker(
        plugins=[
            MeshAiPlugin(model_config("default")),
            MeshAiPlugin(model_config("explicit"), port=8500),
        ],
    )

    with pytest.raises(ValueError, match="plugin startup resource"):
        worker.start()

    assert starts == []
