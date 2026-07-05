from __future__ import annotations

import importlib


def test_remote_manual_relay_defaults_match_production_contract() -> None:
    """Assert the remote manual relay script uses the production mesh contract."""
    script = importlib.import_module("manual_mesh_relay_test_remote")

    assert script.API_WEBSOCKET_HOST == "wss://api.lange-labs.com"
    assert script.RELAY_NAME == "testing"
    assert script.FORWARD_TARGET == "http://localhost:5173"
