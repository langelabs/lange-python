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
    :raises RuntimeError: If a required environment variable is empty.
    """
    env_path = Path(__file__).with_name(".env")
    if not load_dotenv(env_path):
        raise FileNotFoundError(f"Manual relay configuration not found: {env_path}")

    project_id = os.getenv("MESH_PROJECT_ID")
    api_key = os.getenv("LANGE_LABS_API_KEY")
    if not project_id:
        raise RuntimeError("MESH_PROJECT_ID must be set in tests/manual/.env")
    if not api_key:
        raise RuntimeError("LANGE_LABS_API_KEY must be set in tests/manual/.env")

    relay_target = os.getenv("MESH_RELAY_TARGET", DEFAULT_RELAY_TARGET)
    websocket_host = os.getenv(
        "MESH_WEBSOCKET_HOST",
        DEFAULT_MESH_WEBSOCKET_HOST,
    )
    worker = MeshWorker(
        project_id=project_id,
        plugins=[MeshRelayPlugin(relay_target)],
        remote_base_url=websocket_host,
        api_key=api_key,
    )

    print("Starting authenticated manual mesh relay worker")
    print(f"Mesh websocket host: {websocket_host}")
    print(f"Project ID: {project_id}")
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
