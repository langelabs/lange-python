import base64
from typing import Literal


def decode_request_body(encoding: Literal["base64"] | None, body:str|None) -> bytes | None:
    """
    Decode the request body from a relay request.
    """
    if body is None:
        return None
    if encoding == "base64":
        return base64.b64decode(body)
    return body.encode("utf-8")
