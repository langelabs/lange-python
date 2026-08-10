"""Tests for shared AI inference server startup."""

from __future__ import annotations

import importlib
import sys
import types

import pytest

import lange.ai as ai_package
from lange.ai.contracts import AiModelConfig


def model_config(alias: str) -> AiModelConfig:
    """Create a minimal model configuration.

    :param alias: Model alias.
    :returns: Minimal model configuration.
    """
    return AiModelConfig(
        model_name="example/model",
        model_alias=alias,
        model_type="LLM",
        registration=None,
    )


def test_start_ai_models_reuses_server_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create model servers through the shared factory with sequential ports."""
    fake_base = types.ModuleType("lange.ai.servers.__base")
    fake_base.InferenceServer = object
    monkeypatch.setattr(ai_package, "servers", None, raising=False)
    monkeypatch.setitem(sys.modules, "lange.ai.servers.__base", fake_base)
    monkeypatch.delitem(sys.modules, "lange.ai.servers", raising=False)
    servers = importlib.import_module("lange.ai.servers")
    calls: list[tuple[str, str, int]] = []

    class FakeServer:
        """Record server startup and join operations."""

        def __init__(self) -> None:
            """Create an unstarted fake server."""
            self.started = False
            self.joined = False

        def start(self) -> None:
            """Record server startup."""
            self.started = True

        def join(self, timeout: float | None = None) -> None:
            """Record server joining.

            :param timeout: Optional join timeout.
            """
            self.joined = True

    def create_inference_server(
        model: AiModelConfig,
        *,
        host: str,
        port: int,
    ) -> FakeServer:
        """Capture factory arguments.

        :param model: Model being served.
        :param host: Server bind host.
        :param port: Server bind port.
        :returns: Fake inference server.
        """
        calls.append((model.model_alias, host, port))
        return FakeServer()

    monkeypatch.setattr(
        "lange.ai.plugin.create_inference_server",
        create_inference_server,
    )
    models = [model_config("first"), model_config("second")]

    workers = servers.start_ai_models(
        models,
        blocking=True,
        host="0.0.0.0",
        start_port=9100,
    )

    assert calls == [
        ("first", "0.0.0.0", 9100),
        ("second", "0.0.0.0", 9101),
    ]
    assert all(worker.started and worker.joined for worker in workers)
