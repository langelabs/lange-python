"""Run an authenticated relay worker against the Lange mesh service."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from lange.mesh import MeshRelayPlugin, MeshWorker

DEFAULT_MESH_WEBSOCKET_HOST = "wss://mesh.lange-labs.com"
DEFAULT_RELAY_TARGET = "http://localhost:3000"


def main() -> None:
    """Load local configuration and run a relay worker until interrupted.

    :returns: ``None``.
    :raises FileNotFoundError: If the adjacent ``.env`` file is missing.
    :raises RuntimeError: If the API key is empty.
    """
    env_path = Path(__file__).with_name(".env")
    if not load_dotenv(env_path):
        raise FileNotFoundError(f"Manual relay configuration not found: {env_path}")

    api_key = os.getenv("LANGE_LABS_API_KEY")
    if not api_key:
        raise RuntimeError("LANGE_LABS_API_KEY must be set in tests/manual/.env")

    relay_target = os.getenv("MESH_RELAY_TARGET", DEFAULT_RELAY_TARGET)
    websocket_host = os.getenv(
        "MESH_WEBSOCKET_HOST",
        DEFAULT_MESH_WEBSOCKET_HOST,
    )
    worker = MeshWorker(
        plugins=[MeshRelayPlugin(relay_target)],
        remote_base_url=websocket_host,
        api_key=api_key,
    )

    print("Starting authenticated manual mesh relay worker")
    print(f"Mesh websocket host: {websocket_host}")
    print(f"Forward target: {relay_target}")

    worker.start()
    try:
        while worker.is_alive() and worker.remote_relay_address is None:
            time.sleep(0.25)
        if worker.remote_relay_address is not None:
            print(f"Remote relay address: {worker.remote_relay_address}")
        worker.join()
    except KeyboardInterrupt:
        print("Stopping authenticated manual mesh relay worker")
    finally:
        asyncio.run(worker.stop())
        worker.join(timeout=5.0)


if __name__ == "__main__":
    main()
