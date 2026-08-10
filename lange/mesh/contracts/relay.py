from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator


class _RelayHeaders(BaseModel):
    """Provide lossless and legacy relay header representations."""

    headers: dict[str, str] = Field(
        default_factory=dict,
        description="Legacy collapsed header mapping for older relay clients.",
    )
    header_pairs: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Canonical ordered HTTP header pairs.",
    )

    @model_validator(mode="after")
    def normalize_headers(self) -> Self:
        """Synchronize canonical pairs with the legacy header mapping.

        :returns: The payload with both wire representations populated.
        """
        if self.header_pairs:
            self.headers = dict(self.header_pairs)
        else:
            self.header_pairs = list(self.headers.items())
        return self


class MeshRelayRequest(_RelayHeaders):
    """REST request payload sent to a mesh relay worker."""

    method: str
    path: str
    body: str | None = None
    body_encoding: Literal["base64"] | None = None
    query_params: dict[str, list[str]] = Field(default_factory=dict)
    query_string: str | None = None


class MeshRelayResponse(_RelayHeaders):
    """REST response payload returned by a mesh relay worker."""

    status: int
    body: str | None = None
    body_encoding: Literal["base64"] | None = None
    error: str | None = None


__all__ = [
    "MeshRelayRequest",
    "MeshRelayResponse",
]
