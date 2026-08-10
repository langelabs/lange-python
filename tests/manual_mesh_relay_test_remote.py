import asyncio
import time
from lange.mesh import MeshRelayPlugin, MeshWorker


MESH_WEBSOCKET_HOST = "wss://mesh.lange-labs.com"
PROJECT_ID = "00000000-0000-0000-0000-000000000001"
FORWARD_TARGET = "http://localhost:5173"


def main() -> None:
    """Run a production manual mesh relay client until interrupted.

    :returns: ``None``.
    """

    relay = MeshWorker(
        project_id=PROJECT_ID,
        plugins=[MeshRelayPlugin(FORWARD_TARGET)],
        remote_base_url=MESH_WEBSOCKET_HOST,
    )

    print("Starting remote manual mesh relay client")
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
        print("Stopping remote manual mesh relay client")
        asyncio.run(relay.stop())
        relay.join(timeout=5.0)


if __name__ == "__main__":
    main()
