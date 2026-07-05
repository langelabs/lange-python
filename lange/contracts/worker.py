from typing import Literal

from pydantic import BaseModel


class MeshWorkerRegistration(BaseModel):
    """Registration payload sent by a mesh relay worker during hello."""

    name: str
    timeout: float


class MeshWorkerConfig(BaseModel):
    """Runtime configuration returned to a registered mesh relay worker."""

    remote_relay_address: str
    type: Literal["REST"]

__all__ = [
    "MeshWorkerConfig",
    "MeshWorkerRegistration",
]
