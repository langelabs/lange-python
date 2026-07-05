from __future__ import annotations

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


API_WEBSOCKET_HOST = "ws://localhost:8000"
RELAY_NAME = "dev_manual_test"
FORWARD_TARGET = "http://localhost:5173"


def main() -> None:
    """Run a local manual mesh relay client until interrupted.

    :returns: ``None``.
    """
    from lange.mesh.worker import MeshWorker

    relay = MeshWorker(
        name=RELAY_NAME,
        relay_target=FORWARD_TARGET,
        remote_base_url=API_WEBSOCKET_HOST,
    )

    print("Starting manual mesh relay client")
    print(f"API websocket host: {API_WEBSOCKET_HOST}")
    print(f"Relay worker name: {RELAY_NAME}")
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
        relay.stop()
        relay.join(timeout=5.0)


if __name__ == "__main__":
    main()
