from typing import Literal

from pydantic import BaseModel, ConfigDict


PLATFORM_TYPE = Literal["Darwin", "Linux", "Windows", "_unknown"]


class MeshWorkerRegistration(BaseModel):
    """Registration payload sent by a mesh relay worker during hello."""

    model_config = ConfigDict(extra="forbid")

    timeout: float
    platform: PLATFORM_TYPE|None = None


class MeshWorkerConfig(BaseModel):
    """Runtime configuration returned to a registered mesh relay worker."""

    remote_relay_address: str
    type: Literal["REST"]

__all__ = [
    "MeshWorkerConfig",
    "MeshWorkerRegistration",
    "PLATFORM_TYPE"
]
