import uuid
from typing import Literal

from pydantic import BaseModel, Field

from .relay import MeshRelayRequest, MeshRelayResponse
from .worker import MeshWorkerConfig, MeshWorkerRegistration


class MeshMessage(BaseModel):
    """Envelope exchanged between mesh workers and the mesh service."""

    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    status: Literal[
        "hello",
        "bye",
        "ping",
        "ready",
        "pending",
        "request",
        "response",
    ]
    type: Literal["relay", "compute", "manage"]
    data: (
        MeshWorkerRegistration
        | MeshWorkerConfig
        | MeshRelayRequest
        | MeshRelayResponse
        | None
    )


__all__ = ["MeshMessage"]
