from typing import Mapping
from urllib.parse import urlencode, urljoin, urlparse, urlunparse


def build_url(
    base_url: str,
    path: str,
    query_params: Mapping[str, list[str]] | None = None,
    query_string: str | None = None,
) -> str:
    joined_url = urlparse(urljoin(f"{base_url}/", path.lstrip("/")))
    resolved_query_string = (
        urlencode(query_params, doseq=True) if query_params else query_string or ""
    )
    return urlunparse(
        (
            joined_url.scheme,
            joined_url.netloc,
            joined_url.path,
            joined_url.params,
            resolved_query_string,
            joined_url.fragment,
        )
    )
