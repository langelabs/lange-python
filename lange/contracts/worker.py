from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MeshWorkerRegistration(BaseModel):
    """Registration payload sent by a mesh relay worker during hello."""

    model_config = ConfigDict(populate_by_name=True)

    name: str
    request_timeout_seconds: float = Field(
        validation_alias="requestTimeoutSeconds",
        serialization_alias="requestTimeoutSeconds",
    )


class MeshWorkerConfig(BaseModel):
    """Runtime configuration returned to a registered mesh relay worker."""

    remote_relay_address: str
    type: Literal["REST"]

__all__ = [
    "MeshWorkerConfig",
    "MeshWorkerRegistration",
]
