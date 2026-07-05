"""Tests for compute REST mesh payload types."""

from lange.contracts.mesh import MeshMessage
from lange.contracts.relay import (
    MeshRelayRequest,
    MeshRelayResponse,
)
from lange.contracts.worker import MeshWorkerConfig


def test_mesh_rest_request_serializes_http_payload() -> None:
    """Assert REST request payloads preserve headers, body, and encoding aliases."""
    request = MeshRelayRequest.model_validate(
        {
            "method": "POST",
            "path": "/api/widgets",
            "queryParams": {"page": ["1"], "tag": ["a", "b"]},
            "headers": {"content-type": "application/json"},
            "body": "eyJvayI6dHJ1ZX0=",
            "bodyEncoding": "base64",
        }
    )

    assert request.method == "POST"
    assert request.path == "/api/widgets"
    assert request.query_params == {"page": ["1"], "tag": ["a", "b"]}
    assert request.headers == {"content-type": "application/json"}
    assert request.body == "eyJvayI6dHJ1ZX0="
    assert request.body_encoding == "base64"
    assert request.model_dump(by_alias=True)["bodyEncoding"] == "base64"
    assert request.model_dump(by_alias=True)["queryParams"] == {
        "page": ["1"],
        "tag": ["a", "b"],
    }


def test_mesh_rest_response_serializes_http_payload() -> None:
    """Assert REST response payloads preserve status, headers, body, and errors."""
    response = MeshRelayResponse(
        status=502,
        headers={"content-type": "text/plain"},
        body="upstream failed",
        body_encoding=None,
        error="bad gateway",
    )

    assert response.status == 502
    assert response.headers == {"content-type": "text/plain"}
    assert response.body == "upstream failed"
    assert response.body_encoding is None
    assert response.error == "bad gateway"
    assert "bodyEncoding" in response.model_dump(by_alias=True)


def test_mesh_message_accepts_rest_request_payload() -> None:
    """Assert mesh messages parse REST request payloads into concrete types."""
    message = MeshMessage.model_validate(
        {
            "status": "request",
            "type": "relay",
            "data": {
                "method": "GET",
                "path": "/",
                "headers": {"accept": "text/html"},
                "body": None,
            },
        }
    )

    assert isinstance(message.data, MeshRelayRequest)
    assert message.data.method == "GET"
    assert message.data.path == "/"


def test_mesh_message_accepts_rest_response_payload() -> None:
    """Assert mesh messages parse REST response payloads into concrete types."""
    message = MeshMessage.model_validate(
        {
            "status": "response",
            "type": "relay",
            "data": {
                "status": 204,
                "headers": {},
                "body": None,
            },
        }
    )

    assert isinstance(message.data, MeshRelayResponse)
    assert message.data.status == 204


def test_mesh_message_accepts_relay_worker_config_payload() -> None:
    """Assert mesh hello messages parse relay worker config payloads."""
    message = MeshMessage.model_validate(
        {
            "status": "hello",
            "type": "manage",
            "data": {
                "relay_address": "http://localhost:8000/api/v1/mesh/relay/rest/default",
                "ai_models": [],
            },
        }
    )

    assert isinstance(message.data, MeshWorkerConfig)
    assert (
        message.data.relay_address
        == "http://localhost:8000/api/v1/mesh/relay/rest/default"
    )
    assert message.model_dump(mode="json")["data"] == {
        "relay_address": "http://localhost:8000/api/v1/mesh/relay/rest/default",
        "ai_models": [],
    }
