from typing import Mapping


def filter_hop_by_hop_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """
    Remove hop-by-hop headers before forwarding a relay request.

    :param headers: Incoming relay request headers.
    :returns: Headers that are safe to forward to the local target.
    """
    hop_by_hop_headers = {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }

    return {
        key: value
        for key, value in headers.items()
        if key.lower() not in hop_by_hop_headers
        and key.lower() not in {"host", "content-length"}
    }
