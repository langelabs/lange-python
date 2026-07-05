from typing import Literal

from pydantic import BaseModel, Field


class MeshRelayRequest(BaseModel):
    """REST request payload sent to a mesh relay worker."""

    method: str
    path: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = None
    body_encoding: Literal["base64"] | None = None
    query_params: dict[str, list[str]] = Field(default_factory=dict)
    query_string: str | None = None


class MeshRelayResponse(BaseModel):
    """REST response payload returned by a mesh relay worker."""

    status: int
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = None
    body_encoding: Literal["base64"] | None = None
    error: str | None = None

__all__ = [
    "MeshRelayRequest",
    "MeshRelayResponse",
]
