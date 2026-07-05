import asyncio
import threading
import platform

from lange.contracts import MeshRelayRequest
from lange.contracts.worker import MeshWorkerConfig, MeshWorkerRegistration
from lange.mesh.client import MeshClient
from lange.contracts import MeshMessage, AiModelConfig
import base64
import httpx
from httpx import Timeout
from lange.contracts import (
    MeshRelayResponse,
)
from .utils.body import decode_request_body
from .utils.url import build_url
from .utils.headers import filter_hop_by_hop_headers


class MeshWorker:
    def __init__(
        self,
        name: str,
        relay_target: str | None = None,
        timeout: float = 60,
        remote_base_url: str = "wss://mesh.lange-labs.com",
        api_key: str | None = None,
        is_ai_worker: bool = False,
    ) -> None:
        """Create a restartable mesh worker.

        :param name: Worker name registered with the mesh API.
        :param relay_target: Optional local HTTP target for relay requests.
        :param timeout: Connection and relay request timeout in seconds.
        :param remote_base_url: Base websocket URL for the Lange mesh service.
        :param api_key: Optional bearer token used for mesh worker authentication.
        :param is_ai_worker: Whether this worker should start AI model clients.
        """
        self.client: MeshClient | None = None
        self._thread: threading.Thread | None = None
        self._remote_base_url = remote_base_url
        self._api_key = api_key

        # worker config
        self.name = name
        self.platform = platform.system()

        # relay config
        self.timeout = timeout
        self.relay_target = relay_target
        self.remote_relay_address: str | None = None

        # ai config
        self.ai_models: list[AiModelConfig] = []
        self.ai_worker: list = []
        self.is_ai_worker = is_ai_worker

    def _create_client(self) -> MeshClient:
        """Create a new mesh client for one worker run.

        :returns: Mesh client bound to this worker's message handler.
        """
        return MeshClient(
            handler=self.handle,
            remote_base_url=self._remote_base_url,
            api_key=self._api_key,
            timeout=self.timeout,
        )

    def start(self) -> None:
        """Start the worker connection in a fresh daemon thread."""
        if self.is_alive():
            raise RuntimeError("MeshWorker is already running")

        self.client = self._create_client()
        _thread = threading.Thread(target=self.run, daemon=True)
        _thread.start()
        self._thread = _thread

    def run(self) -> None:
        """Run the worker connection until it is stopped or disconnected."""
        asyncio.run(self._run_async())

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread to exit.

        :param timeout: Maximum time to wait in seconds.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        """Return whether the worker thread is currently running.

        :returns: ``True`` when the internal worker thread is alive.
        """
        return self._thread is not None and self._thread.is_alive()

    async def stop(self) -> None:
        """Stop the active worker connection.

        The active mesh client is stopped first, then the worker thread is
        joined in a background thread so callers do not block their event loop.
        Calling this method when the worker is already stopped is safe.
        """
        client = self.client
        if client is not None:
            await client.stop()

        thread = self._thread
        if thread is not None:
            await asyncio.to_thread(thread.join, 5.0)

    async def _run_async(self) -> None:
        """Register the worker and wait for the client connection to finish."""
        client = self.client
        if client is None:
            client = self._create_client()
            self.client = client

        client.start()

        # block until the client is ready and then send the hello
        try:
            await client.block_until_ready()
        except RuntimeError:
            return

        await client.send(
            MeshMessage(
                status="hello",
                data=MeshWorkerRegistration(
                    name=self.name,
                    timeout=self.timeout,
                ),
                type="manage",
            )
        )

        # join the client process
        client.join()

    async def handle(self, message: MeshMessage) -> None:
        """Handle one mesh message and send a response if needed.

        :param message: Mesh message received from the websocket client.
        """
        if message.status == "hello":
            response = await self._handle_hello(message)
        elif message.status == "ping":
            response = await self._handle_ping(message)
        elif message.status == "request" and message.type == "relay":
            response = await self._handle_relay_request(message)
        else:
            raise NotImplementedError(f"message status {message.status} is currently not handled in the client.")

        if response is not None:
            if self.client is None:
                raise RuntimeError("MeshWorker client is not running")
            await self.client.send(response)

    async def _handle_hello(self, message: MeshMessage) -> MeshMessage:
        """Store runtime config from the mesh API and return ready.

        :param message: Mesh hello message with worker configuration.
        :returns: Ready message for the mesh API.
        """
        # guards
        if message.status != "hello" or not isinstance(
                message.data, MeshWorkerConfig
        ):
            raise ValueError()

        self.remote_relay_address = message.data.remote_relay_address

        return MeshMessage(
            status="ready",
            data=None,
            type="manage"
        )

    async def _handle_ping(self, message: MeshMessage) -> MeshMessage:
        """Respond to mesh health checks.

        :param message: Mesh ping message.
        :returns: Ready response message.
        """
        # guards
        if not message.status == "ping":
            raise ValueError()

        if self.client and self.client.ready:
            return MeshMessage(status="ready", data=None, type="manage")
        else:
            return MeshMessage(status="pending", data=None, type="manage")

    async def _handle_relay_request(self, message: MeshMessage) -> MeshMessage | None:
        """Forward one relay request to the local target.

        :param message: Relay request message from the mesh API.
        :returns: Relay response message when the local target responds.
        """
        # guards
        if not message.status == "request" or not isinstance(
                message.data, MeshRelayRequest
        ):
            raise ValueError()
        if not self.relay_target:
            raise ValueError("This worker is not configured for relay work.")

        # build the request parts
        method = message.data.method.upper()
        headers = filter_hop_by_hop_headers(message.data.headers)
        body = decode_request_body(encoding=message.data.body_encoding, body=message.data.body)
        target_url = build_url(
            base_url=self.relay_target,
            path=message.data.path,
            query_string=message.data.query_string,
            query_params=message.data.query_params,
        )
        # request from the target
        async with httpx.AsyncClient(
                timeout=Timeout(timeout=self.timeout)
        ) as http_client:
            try:
                response = await http_client.request(
                    method=method,
                    url=target_url,
                    headers=headers,
                    content=body
                )
                return MeshMessage(
                    status="response",
                    type="relay",
                    data=MeshRelayResponse(
                        status=response.status_code,
                        headers=response.headers,
                        body=base64.b64encode(response.content).decode("utf-8"),
                        body_encoding="base64",
                    ),
                )
            except Exception as e:
                print(e)
