import ssl
import threading
import time
from collections.abc import Callable
from typing import Any, Coroutine

import certifi
from websockets import ClientConnection

from lange.contracts import MeshMessage
import asyncio
import websockets


class MeshClient(threading.Thread):
    def __init__(
        self,
        handler: Callable[[MeshMessage], Coroutine[Any, Any, None]],
        remote_base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        """Create a mesh websocket client thread.

        :param handler: Async callback for decoded mesh messages.
        :param remote_base_url: Base websocket URL for the Lange API.
        :param api_key: Optional bearer token used for authentication.
        :param timeout: Connection and readiness timeout in seconds.
        """
        super().__init__(daemon=True)
        self.handler = handler
        self.remote_base_url = remote_base_url

        self.api_key = api_key
        self.timeout = timeout

        self.websocket: ClientConnection | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

        self.ready: bool = False
        self._stop_requested = threading.Event()

    async def send(self, message: MeshMessage) -> None:
        """Send a ``MeshMessage`` to the remote websocket.

        This method can be awaited from another event loop. The actual websocket
        send is always executed on the MeshClient's internal event loop.

        :param message: Message to serialize and send.
        """
        if self.loop is None:
            raise RuntimeError("MeshClient event loop is not running")

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is self.loop:
            await self._send_on_client_loop(message)
            return

        future = asyncio.run_coroutine_threadsafe(
            self._send_on_client_loop(message),
            self.loop,
        )
        await asyncio.wrap_future(future)

    async def _send_on_client_loop(self, message: MeshMessage) -> None:
        """Send a message from the client-owned event loop.

        :param message: Message to serialize and send.
        """
        if self.websocket is None:
            raise RuntimeError("WebSocket is not connected")

        await self.websocket.send(message.model_dump_json())

    def run(self) -> None:
        """Run the websocket client until the connection closes."""
        asyncio.run(self._run_async())

    async def block_until_ready(self) -> None:
        """Wait until the websocket connection is ready.

        :raises RuntimeError: If the client is stopped before becoming ready.
        :raises TimeoutError: If the client does not become ready in time.
        """
        start_time = time.time()

        while (
            time.time() < start_time + self.timeout
            and not self.ready
            and not self._stop_requested.is_set()
        ):
            await asyncio.sleep(1)

        if self._stop_requested.is_set():
            raise RuntimeError("MeshClient was stopped before becoming ready")

        if not self.ready:
            raise TimeoutError("MeshClient did not become ready in time")

    async def _run_async(self) -> None:
        """Connect to the mesh websocket and process inbound requests."""
        self.loop = asyncio.get_running_loop()
        if self._stop_requested.is_set():
            self.loop = None
            return

        headers = (
            {"Authorization": f"Bearer {self.api_key}"}
            if self.api_key
            else {}
        )

        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.set_alpn_protocols(["http/1.1"])

        try:
            async with websockets.connect(
                uri=f"{self.remote_base_url}/api/v1/mesh/relay/entrypoint",
                additional_headers=headers,
                ssl=ssl_context,
                proxy=None,
                open_timeout=self.timeout,
            ) as websocket:
                self.websocket = websocket
                self.ready = True

                if self._stop_requested.is_set():
                    await websocket.close()
                    return

                await self.accept_requests(websocket)

        finally:
            self.ready = False
            self.websocket = None
            self.loop = None

    async def accept_requests(self, websocket: ClientConnection) -> None:
        """Receive encoded ``MeshMessage`` requests from the websocket and forward them
        to the configured handler.

        :param websocket: Connected websocket to consume.
        """
        async for raw_message in websocket:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")

            request = MeshMessage.model_validate_json(raw_message)
            await self.handler(request)

    async def stop(self) -> None:
        """Close the active websocket connection.

        The close operation is scheduled on the client-owned event loop when the
        caller awaits this method from another loop. Calling this method before
        the client connects or after it has stopped is safe.
        """
        self._stop_requested.set()
        loop = self.loop

        if loop is None or not loop.is_running():
            self.ready = False
            self.websocket = None
            return

        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        if current_loop is loop:
            await self._close_websocket_on_client_loop()
            return

        future = asyncio.run_coroutine_threadsafe(
            self._close_websocket_on_client_loop(),
            loop,
        )
        await asyncio.wrap_future(future)

    async def _close_websocket_on_client_loop(self) -> None:
        """Close and clear the websocket from the client-owned event loop."""
        websocket = self.websocket
        self.ready = False
        self.websocket = None

        if websocket is not None:
            await websocket.close()
