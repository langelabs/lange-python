"""Tests for mesh worker plugin composition."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
import pytest

from lange.mesh import MeshPlugin, MeshRelayPlugin, MeshWorker
from lange.mesh.contracts import MeshMessage, MeshRelayRequest, MeshRelayResponse


class RecordingPlugin(MeshPlugin):
    """Record lifecycle and message operations for worker tests."""

    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        handles_messages: bool = False,
        fail_start: bool = False,
    ) -> None:
        """Configure a recording plugin.

        :param name: Plugin name recorded in events.
        :param events: Shared event collection.
        :param handles_messages: Whether the plugin supports every message.
        :param fail_start: Whether startup should fail.
        """
        self.name = name
        self.events = events
        self.handles_messages = handles_messages
        self.fail_start = fail_start

    def start(self, *, instance_index: int) -> None:
        """Record plugin startup.

        :param instance_index: Index assigned among matching plugin types.
        """
        self.events.append(f"start:{self.name}:{instance_index}")
        if self.fail_start:
            raise RuntimeError(f"{self.name} failed")

    def stop(self) -> None:
        """Record plugin shutdown."""
        self.events.append(f"stop:{self.name}")

    def supports(self, message: MeshMessage) -> bool:
        """Return the configured message support state.

        :param message: Candidate message.
        :returns: Configured support state.
        """
        return self.handles_messages

    async def handle(self, message: MeshMessage) -> MeshMessage | None:
        """Return a ready response and record handling.

        :param message: Supported message.
        :returns: Ready response.
        """
        self.events.append(f"handle:{self.name}")
        return MeshMessage(status="ready", type="manage", data=None)


class FakeThread:
    """Thread test double that records starts without executing its target."""

    def __init__(self) -> None:
        """Initialize a stopped fake thread."""
        self.started = False

    def start(self) -> None:
        """Record that the thread was started."""
        self.started = True

    def is_alive(self) -> bool:
        """Return whether the thread was started.

        :returns: Fake thread liveness.
        """
        return self.started

    def join(self, timeout: float | None = None) -> None:
        """Mark the fake thread stopped.

        :param timeout: Ignored join timeout.
        """
        self.started = False


def test_worker_starts_plugins_before_connection_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Start plugins in constructor order before opening the connection."""
    events: list[str] = []
    plugins = [RecordingPlugin("first", events), RecordingPlugin("second", events)]
    worker = MeshWorker(plugins=plugins)
    fake_thread = FakeThread()

    monkeypatch.setattr("lange.mesh.worker.threading.Thread", lambda **_: fake_thread)
    worker.start()

    assert events == ["start:first:0", "start:second:1"]
    assert fake_thread.started


def test_worker_rolls_back_started_plugins_when_startup_fails() -> None:
    """Stop already-started plugins when a later plugin cannot start."""
    events: list[str] = []
    worker = MeshWorker(
        plugins=[
            RecordingPlugin("first", events),
            RecordingPlugin("broken", events, fail_start=True),
        ],
    )

    with pytest.raises(RuntimeError, match="broken failed"):
        worker.start()

    assert events == ["start:first:0", "start:broken:1", "stop:first"]
    assert not worker.is_alive()


def test_worker_rolls_back_plugins_when_client_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stop started plugins if connection setup cannot be created."""
    events: list[str] = []
    worker = MeshWorker(
        plugins=[RecordingPlugin("first", events)],
    )

    def fail_client_creation() -> None:
        """Simulate invalid connection setup.

        :raises RuntimeError: Always.
        """
        raise RuntimeError("client failed")

    monkeypatch.setattr(worker, "_create_client", fail_client_creation)

    with pytest.raises(RuntimeError, match="client failed"):
        worker.start()

    assert events == ["start:first:0", "stop:first"]


def test_worker_stops_plugins_in_reverse_order() -> None:
    """Stop plugins in reverse order after connection shutdown."""
    events: list[str] = []
    first = RecordingPlugin("first", events)
    second = RecordingPlugin("second", events)
    worker = MeshWorker(plugins=[first, second])
    worker._started_plugins = [first, second]

    asyncio.run(worker.stop())

    assert events == ["stop:second", "stop:first"]


def test_worker_rejects_multiple_relay_plugins() -> None:
    """Reject ambiguous relay targets on one worker connection."""
    with pytest.raises(ValueError, match="one MeshRelayPlugin"):
        MeshWorker(
            plugins=[
                MeshRelayPlugin("http://localhost:3000"),
                MeshRelayPlugin("http://localhost:4000"),
            ],
        )


def test_pluginless_worker_returns_correlated_relay_error() -> None:
    """Return a bounded relay error when no plugin handles a request."""
    worker = MeshWorker()
    request = MeshMessage(
        status="request",
        type="relay",
        data=MeshRelayRequest(method="GET", path="/health"),
    )

    response = asyncio.run(worker.handle(request))

    assert response is not None
    assert response.id == request.id
    assert response.status == "response"
    assert isinstance(response.data, MeshRelayResponse)
    assert response.data.status == 502
    assert response.data.error == "No mesh plugin supports this relay request."


def test_worker_rejects_ambiguous_plugin_handlers() -> None:
    """Reject a message claimed by more than one plugin."""
    events: list[str] = []
    worker = MeshWorker(
        plugins=[
            RecordingPlugin("first", events, handles_messages=True),
            RecordingPlugin("second", events, handles_messages=True),
        ],
    )
    message = MeshMessage(status="request", type="compute", data=None)

    with pytest.raises(RuntimeError, match="Multiple mesh plugins"):
        asyncio.run(worker.handle(message))

    assert events == []


def test_relay_plugin_forwards_http_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward a mesh request with safe headers, body, and repeated query values."""
    captured: dict[str, Any] = {}

    class FakeAsyncClient:
        """Capture one relayed HTTP request."""

        def __init__(self, *, timeout: httpx.Timeout) -> None:
            """Capture the configured timeout.

            :param timeout: Relay HTTP timeout.
            """
            captured["timeout"] = timeout

        async def __aenter__(self) -> "FakeAsyncClient":
            """Enter the fake client context.

            :returns: This fake client.
            """
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            """Exit the fake client context.

            :param exc_type: Optional exception type.
            :param exc: Optional exception.
            :param traceback: Optional traceback.
            """

        async def request(self, **kwargs: Any) -> httpx.Response:
            """Capture request arguments and return a binary response.

            :param kwargs: Request keyword arguments.
            :returns: Fake local response.
            """
            captured.update(kwargs)
            return httpx.Response(
                201,
                headers=[
                    ("set-cookie", "session=one; Path=/"),
                    ("set-cookie", "theme=dark; Path=/"),
                ],
                content=b"\x00result",
            )

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    plugin = MeshRelayPlugin("http://localhost:5173/api", timeout=12.0)
    request = MeshMessage(
        status="request",
        type="relay",
        data=MeshRelayRequest(
            method="post",
            path="/items",
            header_pairs=[
                ("content-type", "application/octet-stream"),
                ("x-tag", "one"),
                ("x-tag", "two"),
                ("Host", "remote"),
            ],
            body="AHBheWxvYWQ=",
            body_encoding="base64",
            query_params={"tag": ["one", "two"]},
            query_string="ignored=true",
        ),
    )

    response = asyncio.run(plugin.handle(request))

    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:5173/api/items?tag=one&tag=two"
    assert captured["headers"] == [
        ("content-type", "application/octet-stream"),
        ("x-tag", "one"),
        ("x-tag", "two"),
    ]
    assert captured["content"] == b"\x00payload"
    assert response.id == request.id
    assert isinstance(response.data, MeshRelayResponse)
    assert response.data.status == 201
    assert response.data.header_pairs == [
        ("set-cookie", "session=one; Path=/"),
        ("set-cookie", "theme=dark; Path=/"),
    ]
    assert response.data.body == "AHJlc3VsdA=="
    assert response.data.body_encoding == "base64"


def test_relay_plugin_confines_absolute_url_paths_to_relay_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep attacker-controlled absolute URL paths on the configured origin.

    :param monkeypatch: Pytest patch helper used to capture the target URL.
    """
    captured: dict[str, Any] = {}

    class FakeAsyncClient:
        """Capture the URL selected by the relay plugin."""

        def __init__(self, *, timeout: httpx.Timeout) -> None:
            """Accept the configured timeout.

            :param timeout: Relay HTTP timeout.
            """

        async def __aenter__(self) -> "FakeAsyncClient":
            """Return this fake client.

            :returns: Active fake client.
            """
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            """Exit the fake client context.

            :param exc_type: Optional exception type.
            :param exc: Optional exception.
            :param traceback: Optional traceback.
            """

        async def request(self, **kwargs: Any) -> httpx.Response:
            """Capture request arguments and return an empty response.

            :param kwargs: Request keyword arguments.
            :returns: Successful fake response.
            """
            captured.update(kwargs)
            return httpx.Response(204)

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)
    plugin = MeshRelayPlugin("http://127.0.0.1:5173/api")
    request = MeshMessage(
        status="request",
        type="relay",
        data=MeshRelayRequest(
            method="GET",
            path="/https://attacker.example/private",
        ),
    )

    asyncio.run(plugin.handle(request))

    assert captured["url"] == (
        "http://127.0.0.1:5173/api/https://attacker.example/private"
    )


def test_relay_plugin_returns_bounded_gateway_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate local transport failures into correlated relay responses."""

    class FailingAsyncClient:
        """Raise a transport error for every request."""

        def __init__(self, *, timeout: httpx.Timeout) -> None:
            """Accept the configured timeout.

            :param timeout: Relay HTTP timeout.
            """

        async def __aenter__(self) -> "FailingAsyncClient":
            """Enter the fake client context.

            :returns: This fake client.
            """
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            """Exit the fake client context.

            :param exc_type: Optional exception type.
            :param exc: Optional exception.
            :param traceback: Optional traceback.
            """

        async def request(self, **kwargs: Any) -> httpx.Response:
            """Raise a simulated connection failure.

            :param kwargs: Ignored request arguments.
            :raises ConnectError: Always.
            """
            raise httpx.ConnectError("x" * 600)

    monkeypatch.setattr(httpx, "AsyncClient", FailingAsyncClient)
    plugin = MeshRelayPlugin("http://localhost:5173")
    request = MeshMessage(
        status="request",
        type="relay",
        data=MeshRelayRequest(method="GET", path="/health"),
    )

    response = asyncio.run(plugin.handle(request))

    assert response.id == request.id
    assert isinstance(response.data, MeshRelayResponse)
    assert response.data.status == 502
    assert response.data.error is not None
    assert len(response.data.error) == 512
