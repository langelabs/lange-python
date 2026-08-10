# lange-python

Python helpers for Lange services.

## Mesh Relay Worker

`MeshWorker` connects a local HTTP service to the Lange mesh relay and forwards
public relay requests to your local target.

```bash
pip install lange-python
```

```python
import asyncio
import os
import time

from lange.mesh.worker import MeshWorker

relay = MeshWorker(
    project_id=os.environ["MESH_PROJECT_ID"],
    relay_target="http://localhost:3000",
)

relay.start()

try:
    while relay.remote_relay_address is None and relay.is_alive():
        time.sleep(0.25)
    print(relay.remote_relay_address)
finally:
    asyncio.run(relay.stop())
```

The worker connects to `wss://mesh.lange-labs.com` by default and receives a
public relay address for the project, such as
`https://my-project.mesh.lange-labs.com/`.

If a mesh deployment requires bearer authentication, pass the token as
`api_key`. Keep API keys in the environment or a local secret store instead of
hardcoding them in application code.

```python
import os

from lange.mesh.worker import MeshWorker

relay = MeshWorker(
    project_id=os.environ["MESH_PROJECT_ID"],
    relay_target="http://localhost:3000",
    api_key=os.environ["LANGE_LABS_API_KEY"],
)
```

The relay exposes lifecycle state for integrations:

- `remote_relay_address`: public REST relay address returned by the mesh service

Stop the worker from an async context:

```python
await relay.stop()
```
