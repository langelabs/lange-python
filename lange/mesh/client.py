import ssl
import threading
import time
from collections.abc import Callable
from typing import Coroutine, Any

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
    ):
        super().__init__(daemon=True)
        self.handler = handler
        self.remote_base_url = remote_base_url
        
        self.api_key = api_key
        self.timeout = timeout

        self.websocket: ClientConnection | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

        self.ready: bool = False
        self._stop_event = asyncio.Event()

    async def send(self, message: MeshMessage) -> None:
        """
        Send a MeshMessage to the remote websocket.

        This method can be awaited from another event loop. The actual websocket
        send is always executed on the MeshClient's internal event loop.
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
        if self.websocket is None:
            raise RuntimeError("WebSocket is not connected")

        await self.websocket.send(message.model_dump_json())

    def run(self) -> None:
        asyncio.run(self._run_async())


    async def block_until_ready(self) -> None:
        start_time = time.time()

        while time.time() < start_time + self.timeout and not self.ready:
            await asyncio.sleep(1)

        if not self.ready:
            raise TimeoutError("MeshClient did not become ready in time")


    async def _run_async(self) -> None:
        self.loop = asyncio.get_running_loop()

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

                await self.accept_requests(websocket)

        finally:
            self.ready = False
            self.websocket = None

    async def accept_requests(self, websocket: ClientConnection) -> None:
        """
        Receive encoded MeshMessage requests from the websocket and forward them
        to the configured handler.
        """
        async for raw_message in websocket:
            if isinstance(raw_message, bytes):
                raw_message = raw_message.decode("utf-8")

            request = MeshMessage.model_validate_json(raw_message)
            await self.handler(request)