from pydantic import BaseModel, Field
import uuid
from typing import Literal

from .worker import MeshWorkerRegistration, MeshWorkerConfig
from .relay import MeshRelayRequest, MeshRelayResponse


class MeshMessage(BaseModel):
    id: uuid.UUID = Field(default_factory=uuid.uuid4)
    status: Literal["hello", "bye", "ping", "ready","pending", "request", "response"]
    type: Literal["relay", "compute", "manage"] | None = None

    # data type
    data: (
        MeshWorkerRegistration
        | MeshWorkerConfig
        | MeshRelayRequest
        | MeshRelayResponse
        | None
    )
