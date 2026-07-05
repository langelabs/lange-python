import asyncio
import threading
import platform

from lange.contracts import MeshRelayRequest, MeshWorkerRegistration
from lange.mesh.client import MeshClient
from lange.contracts import MeshMessage, MeshWorkerConfig, AiModelConfig
import base64
import httpx
from httpx import Timeout
from lange.contracts import (
    MeshRelayResponse,
)
from .utils.body import decode_request_body
from .utils.url import build_url
from .utils.headers import filter_hop_by_hop_headers
from .ai_clients import start_ai_models


class MeshWorker(threading.Thread):
    def __init__(
        self,
            name:str,
        relay_target: str | None = None,
        timeout: float = 60,
        remote_base_url="wss://api.lange-labs.com",
        is_ai_worker: bool = False,
    ):
        super().__init__(daemon=True)
        self.client = MeshClient(handler=self.handle,
                                 remote_base_url=remote_base_url,
                                 timeout=timeout)

        # worker config
        self.name = name
        self.platform = platform.system()

        # relay config
        self.timeout = timeout
        self.relay_target = relay_target
        self.remote_relay_address: str | None = None

        # ai config
        self.ai_models: list[AiModelConfig] = []
        self.ai_worker: list = []
        self.is_ai_worker = is_ai_worker
        
        

    def run(self):
        asyncio.run(self._run_async())


    async def _run_async(self):
        self.client.start()

        # block until the client is ready and then send the hello
        await self.client.block_until_ready()
        await self.client.send(
            MeshMessage(
                status="hello",
                type="manage",
                data=MeshWorkerRegistration(
                    name=self.name,
                    timeout=self.timeout,
                    platform=self.platform,
                    is_ai_worker=self.is_ai_worker,
                    is_relay_worker=self.relay_target is not None,
                ),
            )
        )

        # join the client process
        self.client.join()


    async def handle(self, message: MeshMessage) -> None:
        if message.status == "hello":
            response = await self._handle_hello(message)
        elif message.status == "request" and message.type == "relay":
            response = await self._handle_relay_request(message)
        else:
            raise NotImplementedError()

        if response is not None:
            await self.client.send(response)

    async def _handle_hello(self, message: MeshMessage) -> MeshMessage:
        # guards
        if not message.status == "hello" or not isinstance(
            message.data, MeshWorkerConfig
        ):
            raise ValueError()

        self.remote_relay_address = message.data.relay_address
        self.ai_models = message.data.ai_models

        # start the ai worker in case they are supplied
        if self.is_ai_worker and len(self.ai_models) > 0:
            self.ai_worker = start_ai_models(self.ai_models, self.platform)

        return MeshMessage(
            status="ready",
            type="manage",
            data=None
        )



    async def _handle_relay_request(self, message: MeshMessage) -> MeshMessage | None:
        # guards
        if not message.status == "request" or not isinstance(
            message.data, MeshRelayRequest
        ):
            raise ValueError()
        if not self.relay_target:
            raise ValueError("This worker is not configured for relay work.")

        # build the request parts
        method = message.data.method.upper()
        headers = filter_hop_by_hop_headers(message.data.headers)
        body = decode_request_body(encoding=message.data.body_encoding,body=message.data.body)
        target_url = build_url(
            base_url=self.relay_target,
            path=message.data.path,
            query_string=message.data.query_string,
            query_params=message.data.query_params,
        )
        # request from the target
        async with httpx.AsyncClient(
            timeout=Timeout(timeout=self.timeout)
        ) as http_client:
            try:
                response = await http_client.request(
                    method=method,
                    url=target_url,
                    headers=headers,
                    content=body
                )
                return MeshMessage(
                    status="response",
                    type="relay",
                    data=MeshRelayResponse(
                        status=response.status_code,
                        headers=response.headers,
                        body=base64.b64encode(response.content).decode("utf-8"),
                        body_encoding="base64",
                    ),
                )
            except Exception as e:
                print(e)
        
