"""Tests for current mesh client and worker behavior."""

from __future__ import annotations

import asyncio
import base64
import threading
from typing import Any

import httpx
import pytest

from lange.contracts import MeshMessage
from lange.contracts.relay import MeshRelayRequest, MeshRelayResponse
from lange.contracts.worker import MeshWorkerConfig
from lange.mesh.client import MeshClient
from lange.mesh.worker import MeshWorker


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
        client = MeshClient(handler=handle, remote_base_url="ws://example.test")
        await client.accept_requests(FakeWebSocket())  # type: ignore[arg-type]

    asyncio.run(run())

    assert len(handled_messages) == 1
    assert handled_messages[0].status == "request"
    assert handled_messages[0].type == "relay"
    assert isinstance(handled_messages[0].data, MeshRelayRequest)
    assert handled_messages[0].data.path == "/health"


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
    )

    async def run() -> None:
        """Stop the client before its async runner starts."""
        await client.stop()
        await client._run_async()

    asyncio.run(run())

    assert client.ready is False
    assert client.websocket is None
    assert client.loop is None


def test_mesh_worker_hello_stores_runtime_config_and_returns_ready() -> None:
    """Handle mesh hello config without starting AI clients for relay-only workers."""
    worker = MeshWorker(
        name="local-relay",
        relay_target="http://localhost:5173",
        remote_base_url="ws://example.test",
    )
    config = MeshWorkerConfig(
        relay_address="https://api.example.test/api/v1/mesh/relay/rest/local-relay",
        ai_models=[],
    )

    async def run() -> MeshMessage:
        """Handle one hello message.

        :returns: Ready response message.
        """
        return await worker._handle_hello(
            MeshMessage(status="hello", type="manage", data=config)
        )

    response = asyncio.run(run())

    assert worker.remote_relay_address == config.relay_address
    assert worker.ai_models == []
    assert response.status == "ready"
    assert response.type == "manage"
    assert response.data is None


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

    worker = MeshWorker(name="local-relay", remote_base_url="ws://example.test")
    client = FakeClient()
    thread = FakeThread()
    worker.client = client  # type: ignore[assignment]
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
            """Return immediately for tests."""
            return None

        async def stop(self) -> None:
            """Mark the fake client as stopped."""
            self.stopped = True

    monkeypatch.setattr("lange.mesh.worker.MeshClient", FakeClient)
    worker = MeshWorker(name="local-relay", remote_base_url="ws://example.test")

    worker.start()
    worker.join(timeout=1.0)
    asyncio.run(worker.stop())
    first_thread = worker._thread

    worker.start()
    worker.join(timeout=1.0)
    asyncio.run(worker.stop())
    second_thread = worker._thread

    assert first_thread is not second_thread
    assert len(FakeClient.instances) == 2
    assert all(client.started and client.stopped for client in FakeClient.instances)


def test_mesh_worker_forwards_relay_requests_to_local_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forward relay requests to the configured local HTTP target."""
    calls: list[dict[str, Any]] = []
    timeouts: list[httpx.Timeout] = []

    class FakeAsyncClient:
        """Fake local target HTTP client."""

        def __init__(self, *, timeout: httpx.Timeout) -> None:
            """Capture the configured timeout.

            :param timeout: Timeout configured by the mesh worker.
            """
            timeouts.append(timeout)

        async def __aenter__(self) -> "FakeAsyncClient":
            """Return this fake client as the context manager value.

            :returns: Fake async client.
            """
            return self

        async def __aexit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            traceback: object | None,
        ) -> None:
            """Exit the fake async context manager.

            :param exc_type: Optional exception type.
            :param exc: Optional exception value.
            :param traceback: Optional traceback.
            """
            return None

        async def request(self, **kwargs: Any) -> httpx.Response:
            """Capture request kwargs and return a small JSON response.

            :param kwargs: Request keyword arguments.
            :returns: Fake HTTP response.
            """
            calls.append(kwargs)
            return httpx.Response(
                201,
                headers={"Content-Type": "application/json"},
                content=b'{"ok":true}',
            )

    monkeypatch.setattr("lange.mesh.worker.httpx.AsyncClient", FakeAsyncClient)
    worker = MeshWorker(
        name="local-relay",
        relay_target="http://localhost:5173/api",
        timeout=12.5,
        remote_base_url="ws://example.test",
    )
    request = MeshRelayRequest(
        method="POST",
        path="/jobs",
        query_params={"tag": ["alpha", "beta"]},
        headers={
            "Content-Type": "application/json",
            "Connection": "keep-alive",
        },
        body=base64.b64encode(b'{"job":"run"}').decode("ascii"),
        body_encoding="base64",
    )

    async def run() -> MeshMessage | None:
        """Handle one relay request.

        :returns: Relay response message.
        """
        return await worker._handle_relay_request(
            MeshMessage(status="request", type="relay", data=request)
        )

    response = asyncio.run(run())

    assert calls == [
        {
            "method": "POST",
            "url": "http://localhost:5173/api/jobs?tag=alpha&tag=beta",
            "headers": {"Content-Type": "application/json"},
            "content": b'{"job":"run"}',
        }
    ]
    assert timeouts[0].connect == 12.5
    assert response is not None
    assert response.status == "response"
    assert response.type == "relay"
    assert isinstance(response.data, MeshRelayResponse)
    assert response.data.status == 201
    assert response.data.body_encoding == "base64"
    assert base64.b64decode(response.data.body or "") == b'{"ok":true}'
