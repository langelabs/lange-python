# lange-python

Python helpers for Lange services.

The package is organized into two public domains:

- `lange.mesh` contains mesh workers and plugins; its wire models live in
  `lange.mesh.contracts`.
- `lange.ai` contains AI inference plugins, models, and servers; its model
  configuration types live in `lange.ai.contracts`.

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

from lange.mesh import MeshRelayPlugin, MeshWorker

relay = MeshWorker(
    project_id=os.environ["MESH_PROJECT_ID"],
    plugins=[MeshRelayPlugin("http://localhost:3000")],
)

relay.start()

try:
    while relay.remote_relay_address is None and relay.is_alive():
        time.sleep(0.25)
    print(relay.remote_relay_address)
finally:
    asyncio.run(relay.stop())
```

The worker connects to `wss://mesh.lange-labs.com/worker/proxy` by default and receives a
public relay address for the project, such as
`https://my-project.mesh.lange-labs.com/`.

If a mesh deployment requires bearer authentication, pass the token as
`api_key`. Keep API keys in the environment or a local secret store instead of
hardcoding them in application code.

```python
import os

from lange.mesh import MeshRelayPlugin, MeshWorker

relay = MeshWorker(
    project_id=os.environ["MESH_PROJECT_ID"],
    plugins=[MeshRelayPlugin("http://localhost:3000")],
    api_key=os.environ["LANGE_LABS_API_KEY"],
)
```

The relay exposes lifecycle state for integrations:

- `remote_relay_address`: public REST relay address returned by the mesh service

Stop the worker from an async context:

```python
await relay.stop()
```

## AI Inference Plugins

Install the backend required by the host platform, then attach one plugin per
model. Default ports are assigned from `8500` in plugin order and can be
overridden explicitly.

```bash
pip install "lange-python[ai-mlx]"
```

```python
from lange.ai import MeshAiPlugin
from lange.ai.contracts import AiModelConfig
from lange.mesh import MeshWorker

model = AiModelConfig(
    model_name="example/model",
    model_alias="example",
    model_type="LLM",
    registration=None,
)
worker = MeshWorker(
    project_id="00000000-0000-0000-0000-000000000001",
    plugins=[MeshAiPlugin(model)],
)
```
