from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class MeshRelayRequest(BaseModel):
    """REST request payload sent to a mesh relay worker."""

    model_config = ConfigDict(populate_by_name=True)

    method: str
    path: str
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = None
    body_encoding: Literal["base64"] | None = Field(
        default=None,
        validation_alias="bodyEncoding",
        serialization_alias="bodyEncoding",
    )
    query_params: dict[str, list[str]] = Field(
        default_factory=dict,
        validation_alias="queryParams",
        serialization_alias="queryParams",
    )
    query_string: str | None = None


class MeshRelayResponse(BaseModel):
    """REST response payload returned by a mesh compute worker."""

    model_config = ConfigDict(populate_by_name=True)

    status: int
    headers: dict[str, str] = Field(default_factory=dict)
    body: str | None = None
    body_encoding: Literal["base64"] | None = Field(
        default=None,
        validation_alias="bodyEncoding",
        serialization_alias="bodyEncoding",
    )
    error: str | None = None
