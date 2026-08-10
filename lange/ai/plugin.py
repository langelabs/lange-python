"""AI inference plugin for mesh workers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lange.mesh import MeshPlugin
from lange.mesh.contracts import MeshMessage

from .contracts import AiModelConfig
from .utils import get_platform

if TYPE_CHECKING:
    from .servers.__base import InferenceServer


def create_inference_server(
    model: AiModelConfig,
    *,
    host: str,
    port: int,
) -> InferenceServer:
    """Create the platform-appropriate inference server without starting it.

    :param model: Model served by the inference runtime.
    :param host: Server bind host.
    :param port: Server bind port.
    :returns: Configured inference server.
    :raises NotImplementedError: If the platform or model type is unsupported.
    """
    current_platform = get_platform()
    if current_platform == "Darwin":
        from .servers.mlx import MLXEmbeddingsServer, MlxImageServer, MlxLLMServer

        if model.model_type == "LLM":
            return MlxLLMServer(model, host=host, port=port)
        if model.model_type == "embedding":
            return MLXEmbeddingsServer(model, host=host, port=port)
        if model.model_type == "image":
            return MlxImageServer(model, host=host, port=port)
        raise NotImplementedError(f"Unsupported model type for MLX: {model.model_type}")
    if current_platform in {"Linux", "Windows"}:
        from .servers.llama_cpp import LlamaCppServer

        return LlamaCppServer(model, host=host, port=port)
    raise NotImplementedError(f"Unsupported inference platform: {current_platform}")


class MeshAiPlugin(MeshPlugin):
    """Manage one local AI inference server."""

    def __init__(
        self,
        model: AiModelConfig,
        *,
        host: str = "127.0.0.1",
        port: int | None = None,
    ) -> None:
        """Create an AI inference plugin.

        :param model: Model served by this plugin.
        :param host: Inference server bind host.
        :param port: Optional explicit inference server port.
        :raises ValueError: If the explicit port is outside the TCP port range.
        """
        if port is not None and not 1 <= port <= 65535:
            raise ValueError("AI inference server port must be between 1 and 65535.")
        self.model = model
        self.host = host
        self.port = port
        self.resolved_port: int | None = None
        self._server: InferenceServer | None = None

    def startup_resources(self, *, instance_index: int) -> frozenset[str]:
        """Claim the resolved inference server port.

        :param instance_index: Zero-based AI plugin index.
        :returns: Exclusive TCP bind resource.
        """
        resolved_port = self.port if self.port is not None else 8500 + instance_index
        if resolved_port > 65535:
            raise ValueError("Resolved AI inference port exceeds 65535.")
        self.resolved_port = resolved_port
        return frozenset({f"tcp:{self.host}:{resolved_port}"})

    def start(self, *, instance_index: int) -> None:
        """Create and start the model server.

        :param instance_index: Zero-based AI plugin index.
        :raises RuntimeError: If this plugin is already running.
        """
        if self._server is not None:
            raise RuntimeError("MeshAiPlugin is already running.")
        if self.resolved_port is None:
            self.startup_resources(instance_index=instance_index)
        if self.resolved_port is None:  # pragma: no cover - guarded above
            raise RuntimeError("AI inference server port was not resolved.")
        server = create_inference_server(
            self.model,
            host=self.host,
            port=self.resolved_port,
        )
        self._server = server
        try:
            server.start()
        except Exception:
            self._server = None
            server.stop()
            server.join(timeout=5.0)
            raise

    def stop(self) -> None:
        """Stop and join the active model server."""
        server = self._server
        self._server = None
        if server is not None:
            server.stop()
            server.join(timeout=5.0)

    def supports(self, message: MeshMessage) -> bool:
        """Return whether the current server protocol routes AI messages.

        :param message: Candidate mesh message.
        :returns: Always ``False`` until the server exposes an AI protocol.
        """
        return False

    async def handle(self, message: MeshMessage) -> MeshMessage | None:
        """Reject direct AI messages until a server protocol exists.

        :param message: Unsupported mesh message.
        :raises NotImplementedError: Always.
        """
        raise NotImplementedError("The mesh server does not expose AI messages yet.")
