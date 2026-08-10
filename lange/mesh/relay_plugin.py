"""HTTP relay plugin for a mesh worker."""

import base64
from urllib.parse import urlencode, urlparse, urlunparse

import httpx

from .contracts import MeshMessage, MeshRelayRequest, MeshRelayResponse
from .plugin import MeshPlugin

HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "content-length",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class MeshRelayPlugin(MeshPlugin):
    """Relay mesh HTTP requests to one local target."""

    def __init__(self, relay_target: str, *, timeout: float = 60.0) -> None:
        """Create an HTTP relay plugin.

        :param relay_target: Local HTTP service receiving relayed requests.
        :param timeout: Local request timeout in seconds.
        """
        self.relay_target = relay_target
        self.timeout = timeout

    def start(self, *, instance_index: int) -> None:
        """Start the stateless relay plugin.

        :param instance_index: Zero-based relay plugin index.
        """

    def stop(self) -> None:
        """Stop the stateless relay plugin."""

    def supports(self, message: MeshMessage) -> bool:
        """Return whether the message is an HTTP relay request.

        :param message: Candidate mesh message.
        :returns: Whether the message is a relay request.
        """
        return (
            message.type == "relay"
            and message.status == "request"
            and isinstance(message.data, MeshRelayRequest)
        )

    async def handle(self, message: MeshMessage) -> MeshMessage:
        """Forward a relay request and return its correlated response.

        :param message: Relay request to process.
        :returns: Relay response message.
        """
        if not isinstance(message.data, MeshRelayRequest):
            raise ValueError("MeshRelayPlugin requires relay request data.")

        request = message.data
        parsed_url = urlparse(self.relay_target)
        forwarded_path = (
            f"{parsed_url.path.rstrip('/')}/{request.path.lstrip('/')}"
        )
        query = (
            urlencode(request.query_params, doseq=True)
            if request.query_params
            else request.query_string or ""
        )
        target_url = urlunparse(
            (
                parsed_url.scheme,
                parsed_url.netloc,
                forwarded_path,
                parsed_url.params,
                query,
                parsed_url.fragment,
            )
        )
        headers = {
            name: value
            for name, value in request.headers.items()
            if name.lower() not in HOP_BY_HOP_HEADERS
        }
        if request.body is None:
            body = None
        elif request.body_encoding == "base64":
            body = base64.b64decode(request.body, validate=True)
        else:
            body = request.body.encode("utf-8")

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout)) as client:
                response = await client.request(
                    method=request.method.upper(),
                    url=target_url,
                    headers=headers,
                    content=body,
                )
        except httpx.HTTPError as error:
            return MeshMessage(
                id=message.id,
                status="response",
                type="relay",
                data=MeshRelayResponse(status=502, error=str(error)[:512]),
            )
        return MeshMessage(
            id=message.id,
            status="response",
            type="relay",
            data=MeshRelayResponse(
                status=response.status_code,
                headers=dict(response.headers),
                body=base64.b64encode(response.content).decode("ascii"),
                body_encoding="base64",
            ),
        )
