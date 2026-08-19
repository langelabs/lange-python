"""Composable mesh worker runtime."""

from __future__ import annotations

import asyncio
from logging import getLogger
import platform
import threading
import uuid
from collections import Counter
from collections.abc import Sequence

from .client import MeshClient
from .contracts import (
    PLATFORM_TYPE,
    MeshMessage,
    MeshRelayRequest,
    MeshRelayResponse,
    MeshWorkerConfig,
    MeshWorkerRegistration,
)
from .plugin import MeshPlugin
from .relay_plugin import MeshRelayPlugin

logger = getLogger("com.lange-labs.mesh")


class MeshWorker:
    """Coordinate a mesh connection and its installed plugins."""

    def __init__(
        self,
        project_id: uuid.UUID | str,
        plugins: Sequence[MeshPlugin] = (),
        timeout: float = 60.0,
        remote_base_url: str = "wss://worker.mesh.lange-labs.com",
        api_key: str | None = None,
    ) -> None:
        """Create a restartable mesh worker.

        :param project_id: Project whose worker pool receives this worker.
        :param plugins: Ordered plugins installed into the worker.
        :param timeout: Connection and protocol timeout in seconds.
        :param remote_base_url: Base WebSocket URL for the mesh service.
        :param api_key: Optional bearer token used for authentication.
        :raises TypeError: If an installed value is not a mesh plugin.
        :raises ValueError: If more than one relay plugin is installed.
        """
        if any(not isinstance(plugin, MeshPlugin) for plugin in plugins):
            raise TypeError("Every worker plugin must implement MeshPlugin.")
        if sum(isinstance(plugin, MeshRelayPlugin) for plugin in plugins) > 1:
            raise ValueError("A MeshWorker accepts at most one MeshRelayPlugin.")

        self.project_id = uuid.UUID(str(project_id))
        self.plugins = tuple(plugins)
        self.timeout = timeout
        self.remote_relay_address: str | None = None

        self._remote_base_url = remote_base_url
        self._api_key = api_key
        platform_name = platform.system()
        if platform_name == "Darwin":
            self.platform: PLATFORM_TYPE = "Darwin"
        elif platform_name == "Linux":
            self.platform = "Linux"
        elif platform_name == "Windows":
            self.platform = "Windows"
        else:
            self.platform = "_unknown"
        self._client: MeshClient | None = None
        self._thread: threading.Thread | None = None
        self._started_plugins: list[MeshPlugin] = []
        self._plugin_lock = threading.Lock()
        self._stop_requested = threading.Event()

    @property
    def client(self) -> MeshClient | None:
        """Return the current mesh client.

        :returns: Active client or ``None`` while stopped.
        """
        return self._client

    def _create_client(self) -> MeshClient:
        """Create a mesh client for one worker run.

        :returns: Client bound to this worker's message handler.
        """
        return MeshClient(
            handler=self.handle,
            remote_base_url=self._remote_base_url,
            project_id=self.project_id,
            api_key=self._api_key,
            timeout=self.timeout,
        )

    def _stop_plugins(self) -> None:
        """Stop all started plugins once in reverse order."""
        with self._plugin_lock:
            plugins = tuple(reversed(self._started_plugins))
            self._started_plugins.clear()
        first_error: Exception | None = None
        for plugin in plugins:
            try:
                plugin.stop()
            except Exception as error:  # pragma: no cover - defensive aggregation
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def start(self) -> None:
        """Start plugins and then open the mesh connection.

        :raises RuntimeError: If the worker is already running.
        """
        if self.is_alive():
            raise RuntimeError("MeshWorker is already running")
        self._stop_requested.clear()

        type_indexes: Counter[type[MeshPlugin]] = Counter()
        indexed_plugins: list[tuple[MeshPlugin, int]] = []
        claimed_resources: set[str] = set()
        for plugin in self.plugins:
            plugin_type = type(plugin)
            instance_index = type_indexes[plugin_type]
            type_indexes[plugin_type] += 1
            resources = plugin.startup_resources(instance_index=instance_index)
            duplicate_resources = claimed_resources.intersection(resources)
            if duplicate_resources:
                duplicate = min(duplicate_resources)
                raise ValueError(f"Duplicate plugin startup resource: {duplicate}.")
            claimed_resources.update(resources)
            indexed_plugins.append((plugin, instance_index))

        try:
            for plugin, instance_index in indexed_plugins:
                plugin.start(instance_index=instance_index)
                self._started_plugins.append(plugin)
            self._client = self._create_client()
            thread = threading.Thread(target=self.run, daemon=True)
            thread.start()
            self._thread = thread
        except Exception:
            self._client = None
            self._thread = None
            self._stop_plugins()
            raise

    def run(self) -> None:
        """Run the worker connection until stopped or disconnected."""
        try:
            asyncio.run(self._run_async())
        finally:
            self._stop_plugins()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the worker thread to exit.

        :param timeout: Maximum time to wait in seconds.
        """
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        """Return whether the worker connection thread is running.

        :returns: ``True`` while the internal thread is alive.
        """
        return self._thread is not None and self._thread.is_alive()

    async def stop(self) -> None:
        """Close the connection and stop plugins in reverse order."""
        self._stop_requested.set()
        client = self._client
        if client is not None:
            await client.stop()

        thread = self._thread
        if thread is not None:
            await asyncio.to_thread(thread.join, 5.0)
        self._thread = None
        self._client = None
        self._stop_plugins()

    async def _run_async(self) -> None:
        """Connect and re-register until an explicit stop is requested."""
        reconnect_delay = 1.0
        while not self._stop_requested.is_set():
            client = self._client
            if client is None:
                client = self._create_client()
                self._client = client

            try:
                client.start()
                await client.block_until_ready()
                if self._stop_requested.is_set():
                    break
                await client.send(
                    MeshMessage(
                        status="hello",
                        data=MeshWorkerRegistration(
                            timeout=self.timeout,
                            platform=self.platform,
                        ),
                        type="manage",
                    )
                )
                client.join()
            except Exception as error:
                if self._stop_requested.is_set():
                    break
                logger.warning(
                    "mesh_worker_connection_failed project_id=%s error=%s",
                    self.project_id,
                    type(error).__name__,
                )
            finally:
                if self._client is client:
                    self._client = None

            if not self._stop_requested.is_set():
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30.0)

    async def handle(self, message: MeshMessage) -> MeshMessage | None:
        """Handle a management message or delegate it to one plugin.

        :param message: Message received from the mesh service.
        :returns: Optional response produced for the message.
        :raises RuntimeError: If multiple plugins claim the message.
        :raises NotImplementedError: If no plugin supports a non-relay message.
        """
        if message.status == "hello" and message.type == "manage":
            response = self._handle_hello(message)
        elif message.status == "ping" and message.type == "manage":
            response = self._handle_ping(message)
        else:
            handlers = [plugin for plugin in self.plugins if plugin.supports(message)]
            if len(handlers) > 1:
                raise RuntimeError("Multiple mesh plugins support this message.")
            if handlers:
                response = await handlers[0].handle(message)
            elif (
                message.status == "request"
                and message.type == "relay"
                and isinstance(message.data, MeshRelayRequest)
            ):
                response = MeshMessage(
                    id=message.id,
                    status="response",
                    type="relay",
                    data=MeshRelayResponse(
                        status=502,
                        error="No mesh plugin supports this relay request.",
                    ),
                )
            else:
                raise NotImplementedError(
                    f"No mesh plugin supports {message.type}:{message.status}."
                )

        if response is not None and self._client is not None:
            await self._client.send(response)
        return response

    def _handle_hello(self, message: MeshMessage) -> MeshMessage:
        """Store server configuration and create the ready response.

        :param message: Server hello containing worker configuration.
        :returns: Ready response.
        :raises ValueError: If the hello payload is invalid.
        """
        if not isinstance(message.data, MeshWorkerConfig):
            raise ValueError("Mesh worker hello requires configuration data.")
        self.remote_relay_address = message.data.remote_relay_address
        return MeshMessage(status="ready", data=None, type="manage")

    def _handle_ping(self, message: MeshMessage) -> MeshMessage:
        """Create a health response for a management ping.

        :param message: Management ping.
        :returns: Ready or pending response based on connection state.
        """
        status = (
            "ready" if self._client is not None and self._client.ready else "pending"
        )
        return MeshMessage(status=status, data=None, type="manage")
