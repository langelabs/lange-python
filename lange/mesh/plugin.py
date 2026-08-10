"""Plugin contract for extending a mesh worker."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .contracts import MeshMessage


class MeshPlugin(ABC):
    """Define lifecycle and message hooks supported by mesh plugins."""

    @abstractmethod
    def start(self, *, instance_index: int) -> None:
        """Start resources owned by the plugin.

        :param instance_index: Zero-based index among plugins of the same type.
        """

    def startup_resources(self, *, instance_index: int) -> frozenset[str]:
        """Return exclusive resources needed before plugin startup.

        :param instance_index: Zero-based index among plugins of the same type.
        :returns: Resource identifiers which must be unique on one worker.
        """
        return frozenset()

    @abstractmethod
    def stop(self) -> None:
        """Stop resources owned by the plugin."""

    @abstractmethod
    def supports(self, message: MeshMessage) -> bool:
        """Return whether the plugin can handle a mesh message.

        :param message: Candidate mesh message.
        :returns: Whether this plugin handles the message.
        """

    @abstractmethod
    async def handle(self, message: MeshMessage) -> MeshMessage | None:
        """Handle a supported mesh message.

        :param message: Supported mesh message.
        :returns: Optional response for the mesh service.
        """
