from __future__ import annotations

import importlib


def test_remote_manual_relay_defaults_match_production_contract() -> None:
    """Assert the remote manual relay script uses the production mesh contract."""
    script = importlib.import_module("manual_mesh_relay_test_remote")

    assert script.MESH_WEBSOCKET_HOST == "wss://worker.mesh.lange-labs.com"
    assert script.PROJECT_ID == "00000000-0000-0000-0000-000000000001"
    assert script.FORWARD_TARGET == "http://localhost:5173"
