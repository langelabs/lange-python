"""Tests for current mesh client and worker behavior."""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any
import uuid

import pytest

from lange.mesh.client import MeshClient
from lange.mesh.contracts import (
    MeshMessage,
    MeshRelayRequest,
    MeshWorkerConfig,
    MeshWorkerRegistration,
)
from lange.mesh.worker import MeshWorker

PROJECT_ID = uuid.UUID("4c705310-f74d-4a13-8f39-8ebf052e70aa")


def test_mesh_client_send_serializes_messages_on_client_loop() -> None:
    """Send ``MeshMessage`` payloads as JSON through the active websocket."""

    class FakeWebSocket:
        """Capture websocket payloads sent by the mesh client."""

        def __init__(self) -> None:
            """Create an empty sent-payload list."""
            self.sent: list[str] = []

        async def send(self, payload: str) -> None:
            """Record one serialized websocket payload.

            :param payload: Serialized message payload.
            """
            self.sent.append(payload)

    async def run() -> FakeWebSocket:
        """Send one ready message from the client-owned loop.

        :returns: Fake websocket containing sent payloads.
        """
        websocket = FakeWebSocket()
        client = MeshClient(
            handler=lambda _message: asyncio.sleep(0),
            remote_base_url="ws://example.test",
            project_id=PROJECT_ID,
        )
        client.loop = asyncio.get_running_loop()
        client.websocket = websocket  # type: ignore[assignment]

        await client.send(MeshMessage(status="ready", type="manage", data=None))
        return websocket

    fake_websocket = asyncio.run(run())

    sent_message = MeshMessage.model_validate_json(fake_websocket.sent[0])
    assert sent_message.status == "ready"
    assert sent_message.type == "manage"
    assert sent_message.data is None


def test_mesh_client_send_serializes_worker_registration() -> None:
    """Send registration payloads with their websocket field names."""

    class FakeWebSocket:
        """Capture websocket payloads sent by the mesh client."""

        def __init__(self) -> None:
            """Create an empty sent-payload list."""
            self.sent: list[str] = []

        async def send(self, payload: str) -> None:
            """Record one serialized websocket payload.

            :param payload: Serialized message payload.
            """
            self.sent.append(payload)

    async def run() -> FakeWebSocket:
        """Send one worker registration from the client-owned loop.

        :returns: Fake websocket containing sent payloads.
        """
        websocket = FakeWebSocket()
        client = MeshClient(
            handler=lambda _message: asyncio.sleep(0),
            remote_base_url="ws://example.test",
            project_id=PROJECT_ID,
        )
        client.loop = asyncio.get_running_loop()
        client.websocket = websocket  # type: ignore[assignment]

        await client.send(
            MeshMessage(
                status="hello",
                type="manage",
                data=MeshWorkerRegistration(
                    timeout=12.5,
                ),
            )
        )
        return websocket

    fake_websocket = asyncio.run(run())

    raw_message = json.loads(fake_websocket.sent[0])
    assert raw_message["type"] == "manage"
    assert raw_message["data"] == {"timeout": 12.5, "platform": None}


def test_mesh_client_accept_requests_decodes_messages_for_handler() -> None:
    """Decode inbound websocket JSON and pass messages to the handler."""
    request = MeshMessage(
        status="request",
        type="relay",
        data=MeshRelayRequest(method="GET", path="/health"),
    )
    handled_messages: list[MeshMessage] = []

    class FakeWebSocket:
        """Async iterable websocket with one inbound payload."""

        def __init__(self) -> None:
            """Create the pending message list."""
            self.messages = [request.model_dump_json()]

        def __aiter__(self) -> "FakeWebSocket":
            """Return the async iterator.

            :returns: The fake websocket itself.
            """
            return self

        async def __anext__(self) -> str:
            """Return the next websocket payload.

            :returns: Serialized mesh message.
            """
            if self.messages:
                return self.messages.pop(0)
            raise StopAsyncIteration

    async def handle(message: MeshMessage) -> None:
        """Capture one decoded mesh message.

        :param message: Decoded mesh message.
        """
        handled_messages.append(message)

    async def run() -> None:
        """Consume the fake websocket."""
        client = MeshClient(
            handler=handle,
            remote_base_url="ws://example.test",
            project_id=PROJECT_ID,
        )
        await client.accept_requests(FakeWebSocket())  # type: ignore[arg-type]

    asyncio.run(run())

    assert len(handled_messages) == 1
    assert handled_messages[0].status == "request"
    assert handled_messages[0].type == "relay"
    assert isinstance(handled_messages[0].data, MeshRelayRequest)
    assert handled_messages[0].data.path == "/health"


def test_mesh_client_accept_requests_handles_relay_messages_concurrently() -> None:
    """Start a second relay request while the first request remains pending."""
    requests = [
        MeshMessage(
            status="request",
            type="relay",
            data=MeshRelayRequest(method="GET", path=f"/{index}"),
        )
        for index in range(2)
    ]
    started: list[uuid.UUID] = []

    class FakeWebSocket:
        """Async iterable websocket containing concurrent relay requests."""

        def __init__(self) -> None:
            """Create serialized inbound messages."""
            self.messages = [request.model_dump_json() for request in requests]

        def __aiter__(self) -> "FakeWebSocket":
            """Return this async iterator.

            :returns: Active fake websocket iterator.
            """
            return self

        async def __anext__(self) -> str:
            """Return the next inbound request.

            :returns: Serialized Mesh message.
            """
            if self.messages:
                return self.messages.pop(0)
            raise StopAsyncIteration

    async def run() -> None:
        """Require both handlers to start before either one completes."""
        release = asyncio.Event()
        both_started = asyncio.Event()

        async def handle(message: MeshMessage) -> None:
            """Block each request until concurrent dispatch is observed.

            :param message: Concurrently dispatched relay request.
            """
            started.append(message.id)
            if len(started) == 2:
                both_started.set()
            await release.wait()

        client = MeshClient(
            handler=handle,
            remote_base_url="ws://example.test",
            project_id=PROJECT_ID,
        )
        accept_task = asyncio.create_task(
            client.accept_requests(FakeWebSocket())  # type: ignore[arg-type]
        )
        try:
            await asyncio.wait_for(both_started.wait(), timeout=0.1)
            release.set()
            await accept_task
        finally:
            if not accept_task.done():
                accept_task.cancel()
                await asyncio.gather(accept_task, return_exceptions=True)

    asyncio.run(run())

    assert started == [request.id for request in requests]


def test_mesh_client_stop_closes_websocket_on_client_loop() -> None:
    """Close the active websocket from a caller-owned event loop."""

    class FakeWebSocket:
        """Capture websocket close calls."""

        def __init__(self) -> None:
            """Create an open fake websocket."""
            self.closed = False

        async def close(self) -> None:
            """Mark the websocket as closed."""
            self.closed = True

    loop = asyncio.new_event_loop()
    loop_ready = threading.Event()

    def run_loop() -> None:
        """Run the client-owned event loop until stopped."""
        asyncio.set_event_loop(loop)
        loop_ready.set()
        loop.run_forever()

    thread = threading.Thread(target=run_loop, daemon=True)
    thread.start()
    loop_ready.wait(timeout=1.0)

    websocket = FakeWebSocket()
    client = MeshClient(
        handler=lambda _message: asyncio.sleep(0),
        remote_base_url="ws://example.test",
        project_id=PROJECT_ID,
    )
    client.loop = loop
    client.websocket = websocket  # type: ignore[assignment]
    client.ready = True

    try:
        asyncio.run(client.stop())
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=1.0)
        loop.close()

    assert websocket.closed is True
    assert client.ready is False
    assert client.websocket is None


def test_mesh_client_stop_is_idempotent_when_not_connected() -> None:
    """Ignore stop calls when the client has no active loop or websocket."""
    client = MeshClient(
        handler=lambda _message: asyncio.sleep(0),
        remote_base_url="ws://example.test",
        project_id=PROJECT_ID,
    )

    asyncio.run(client.stop())
    asyncio.run(client.stop())

    assert client.ready is False
    assert client.websocket is None


def test_mesh_client_stop_before_run_skips_websocket_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avoid opening a websocket when stop was requested before run."""

    def fail_connect(**_kwargs: Any) -> None:
        """Fail if the client attempts a websocket connection."""
        raise AssertionError("websocket connection should not start")

    monkeypatch.setattr("lange.mesh.client.websockets.connect", fail_connect)
    client = MeshClient(
        handler=lambda _message: asyncio.sleep(0),
        remote_base_url="ws://example.test",
        project_id=PROJECT_ID,
    )

    async def run() -> None:
        """Stop the client before its async runner starts."""
        await client.stop()
        await client._run_async()

    asyncio.run(run())

    assert client.ready is False
    assert client.websocket is None
    assert client.loop is None


def test_mesh_client_connects_to_standalone_mesh_entrypoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Connect workers to the deployed standalone mesh websocket path."""
    captured: dict[str, Any] = {}

    class FakeWebSocket:
        """Async websocket context manager that stops after connecting."""

        async def __aenter__(self) -> "FakeWebSocket":
            """Return this fake websocket as the context manager value."""
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            """Exit the fake websocket context."""
            return None

        def __aiter__(self) -> "FakeWebSocket":
            """Return the async iterator."""
            return self

        async def __anext__(self) -> str:
            """Stop request processing after the connection is established."""
            raise StopAsyncIteration

    def fake_connect(**kwargs: Any) -> FakeWebSocket:
        """Capture websocket connection kwargs and return a fake connection."""
        captured.update(kwargs)
        return FakeWebSocket()

    monkeypatch.setattr("lange.mesh.client.websockets.connect", fake_connect)
    client = MeshClient(
        handler=lambda _message: asyncio.sleep(0),
        remote_base_url="wss://worker.mesh.lange-labs.com",
        project_id=PROJECT_ID,
        api_key="secret-token",
    )

    asyncio.run(client._run_async())

    assert captured["uri"] == (
        "wss://worker.mesh.lange-labs.com/worker/proxy"
    )
    assert captured["additional_headers"] == {"Authorization": "Bearer secret-token"}


def test_mesh_worker_defaults_to_standalone_mesh_service() -> None:
    """Default worker connections target the deployed standalone mesh service."""
    worker = MeshWorker(project_id=PROJECT_ID)

    assert worker._remote_base_url == "wss://worker.mesh.lange-labs.com"


def test_mesh_worker_hello_stores_runtime_config_and_returns_ready() -> None:
    """Handle mesh hello config without starting AI clients for relay-only workers."""
    worker = MeshWorker(
        project_id=PROJECT_ID,
        remote_base_url="ws://example.test",
    )
    config = MeshWorkerConfig(
        remote_relay_address="https://local-relay.mesh.example.test/",
        type="REST",
    )

    async def run() -> MeshMessage:
        """Handle one hello message.

        :returns: Ready response message.
        """
        response = await worker.handle(
            MeshMessage(status="hello", type="manage", data=config)
        )
        assert response is not None
        return response

    response = asyncio.run(run())

    assert worker.remote_relay_address == config.remote_relay_address
    assert response.status == "ready"
    assert response.data is None


def test_mesh_worker_sends_project_registration_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register a project worker and pass bearer auth to its client."""

    class FakeClient:
        """Thread-compatible fake mesh client."""

        instances: list["FakeClient"] = []

        def __init__(self, **kwargs: Any) -> None:
            """Capture mesh client constructor kwargs."""
            self.kwargs = kwargs
            self.sent: list[MeshMessage] = []
            FakeClient.instances.append(self)

        def start(self) -> None:
            """Accept worker startup."""
            return None

        async def block_until_ready(self) -> None:
            """Return immediately for tests."""
            return None

        async def send(self, message: MeshMessage) -> None:
            """Capture one outgoing mesh message."""
            self.sent.append(message)

        def join(self) -> None:
            """End the direct worker run after one registered connection."""
            worker._stop_requested.set()

    monkeypatch.setattr("lange.mesh.worker.MeshClient", FakeClient)
    worker = MeshWorker(
        project_id=PROJECT_ID,
        timeout=12.5,
        api_key="secret-token",
    )

    asyncio.run(worker._run_async())

    client = FakeClient.instances[0]
    assert client.kwargs["remote_base_url"] == "wss://worker.mesh.lange-labs.com"
    assert client.kwargs["project_id"] == PROJECT_ID
    assert client.kwargs["api_key"] == "secret-token"
    assert isinstance(client.sent[0].data, MeshWorkerRegistration)
    assert client.sent[0].type == "manage"
    assert client.sent[0].data.timeout == 12.5
    assert client.sent[0].data.model_dump() == {
        "timeout": 12.5,
        "platform": worker.platform,
    }


def test_mesh_worker_reconnects_and_registers_after_client_disconnect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create and register a new client after an unexpected disconnect.

    :param monkeypatch: Pytest patch helper used to replace transport timing.
    """

    class FakeClient:
        """Thread-compatible client that disconnects immediately."""

        instances: list["FakeClient"] = []

        def __init__(self, **_kwargs: Any) -> None:
            """Record each reconnect attempt."""
            self.sent: list[MeshMessage] = []
            FakeClient.instances.append(self)

        def start(self) -> None:
            """Accept client startup."""

        async def block_until_ready(self) -> None:
            """Report an established WebSocket immediately."""

        async def send(self, message: MeshMessage) -> None:
            """Capture the registration sent for this connection.

            :param message: Outbound worker registration.
            """
            self.sent.append(message)

        def join(self) -> None:
            """Disconnect and stop the worker after the second connection."""
            if len(FakeClient.instances) == 2:
                worker._stop_requested.set()

    async def skip_delay(_delay: float) -> None:
        """Skip reconnect backoff in this unit test.

        :param _delay: Ignored reconnect delay.
        """

    monkeypatch.setattr("lange.mesh.worker.MeshClient", FakeClient)
    monkeypatch.setattr("lange.mesh.worker.asyncio.sleep", skip_delay)
    worker = MeshWorker(project_id=PROJECT_ID, remote_base_url="ws://example.test")

    asyncio.run(worker._run_async())

    assert len(FakeClient.instances) == 2
    assert all(len(client.sent) == 1 for client in FakeClient.instances)
    assert all(
        isinstance(client.sent[0].data, MeshWorkerRegistration)
        for client in FakeClient.instances
    )


def test_mesh_worker_stop_stops_client_and_joins_thread() -> None:
    """Stop the active client and join the worker thread without blocking."""

    class FakeClient:
        """Capture async stop calls from the worker."""

        def __init__(self) -> None:
            """Create an unstopped fake client."""
            self.stop_calls = 0

        async def stop(self) -> None:
            """Record one stop request."""
            self.stop_calls += 1

    class FakeThread:
        """Capture delegated thread joins."""

        def __init__(self) -> None:
            """Create an unjoined fake thread."""
            self.join_timeout: float | None = None

        def join(self, timeout: float | None = None) -> None:
            """Record the join timeout.

            :param timeout: Maximum time to wait for thread exit.
            """
            self.join_timeout = timeout

    worker = MeshWorker(project_id=PROJECT_ID, remote_base_url="ws://example.test")
    client = FakeClient()
    thread = FakeThread()
    worker._client = client  # type: ignore[assignment]
    worker._thread = thread  # type: ignore[attr-defined]

    asyncio.run(worker.stop())

    assert client.stop_calls == 1
    assert thread.join_timeout == 5.0


def test_mesh_worker_start_can_restart_after_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Create a fresh client and thread for each worker start."""

    class FakeClient:
        """Thread-compatible fake mesh client."""

        instances: list["FakeClient"] = []

        def __init__(self, **_kwargs: Any) -> None:
            """Record a new fake client instance."""
            self.started = False
            self.stopped = False
            self.disconnected = threading.Event()
            FakeClient.instances.append(self)

        def start(self) -> None:
            """Mark the fake client as started."""
            self.started = True

        async def block_until_ready(self) -> None:
            """Return immediately for tests."""
            return None

        async def send(self, _message: MeshMessage) -> None:
            """Accept outgoing messages without transport."""
            return None

        def join(self) -> None:
            """Wait until the test explicitly stops this connection."""
            self.disconnected.wait(timeout=2.0)

        async def stop(self) -> None:
            """Mark the fake client as stopped."""
            self.stopped = True
            self.disconnected.set()

    monkeypatch.setattr("lange.mesh.worker.MeshClient", FakeClient)
    worker = MeshWorker(project_id=PROJECT_ID, remote_base_url="ws://example.test")

    worker.start()
    first_thread = worker._thread
    worker.join(timeout=1.0)
    asyncio.run(worker.stop())

    worker.start()
    second_thread = worker._thread
    worker.join(timeout=1.0)
    asyncio.run(worker.stop())

    assert first_thread is not second_thread
    assert len(FakeClient.instances) == 2
    assert all(client.started and client.stopped for client in FakeClient.instances)
