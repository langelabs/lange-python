# lange-python

Python helpers for Lange services.

## Mesh Relay Worker

`MeshRelay` connects a local HTTP service to the Lange mesh relay and forwards
public relay requests to your local target.

```bash
pip install lange-python
```

```python
from lange.mesh.relay import MeshRelay

relay = MeshRelay(
    host="wss://api.lange-labs.com",
    key="default",
    target="http://localhost:3000",
)

relay.start()

try:
    print(relay.status)
    print(relay.remote_relay_address)
finally:
    relay.stop()
```

By default, `MeshRelay` reads authentication from the `LANGE_LABS_API_KEY`
environment variable. Keep API keys in the environment or a local secret store
instead of hardcoding them in application code.

```bash
export LANGE_LABS_API_KEY="your-api-key"
```

The relay exposes lifecycle state for integrations:

- `status`: one of `unauthenticated`, `off`, `pending`, `connected`, or `failed`
- `connected`: whether the relay websocket is currently connected
- `remote_relay_address`: public REST relay address returned by the API
- `worker_config`: full worker configuration returned by the API

Reload authentication or reconnect the worker without rebuilding it:

```python
relay.reload()
relay.reload(api_key=None)
relay.reconnect()
```
