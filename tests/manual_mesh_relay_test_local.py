from __future__ import annotations

import asyncio
import sys
import time


def _bootstrap_import_path() -> None:
    """Ensure local script execution cannot shadow standard-library modules."""
    tests_dir = __file__.rsplit("/", 1)[0]
    project_root = tests_dir.rsplit("/", 1)[0]
    package_dir = f"{project_root}/lange"
    normalized_package_dir = package_dir.rstrip("/")

    sanitized_paths = []
    for path in sys.path:
        normalized_path = path.rstrip("/")
        if path == "" or normalized_path == normalized_package_dir:
            continue
        sanitized_paths.append(path)

    sys.path[:] = [project_root] + [
        path for path in sanitized_paths if path.rstrip("/") != project_root
    ]


_bootstrap_import_path()


MESH_WEBSOCKET_HOST = "ws://localhost:8000"
PROJECT_ID = "00000000-0000-0000-0000-000000000001"
FORWARD_TARGET = "http://localhost:5173"


def main() -> None:
    """Run a local manual mesh relay client until interrupted.

    :returns: ``None``.
    """
    from lange.mesh.worker import MeshWorker

    relay = MeshWorker(
        project_id=PROJECT_ID,
        relay_target=FORWARD_TARGET,
        remote_base_url=MESH_WEBSOCKET_HOST,
    )

    print("Starting manual mesh relay client")
    print(f"Mesh websocket host: {MESH_WEBSOCKET_HOST}")
    print(f"Project ID: {PROJECT_ID}")
    print(f"Forward target: {FORWARD_TARGET}")

    relay.start()
    try:
        while relay.is_alive():
            if relay.remote_relay_address is not None:
                print(f"Remote relay address: {relay.remote_relay_address}")
                break
            time.sleep(0.25)
        relay.join()
    except KeyboardInterrupt:
        print("Stopping manual mesh relay client")
        asyncio.run(relay.stop())
        relay.join(timeout=5.0)


if __name__ == "__main__":
    main()
