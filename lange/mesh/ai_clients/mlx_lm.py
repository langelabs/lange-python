import threading

from ..utils.download_model import download_model
from lange.contracts import AiModelConfig
import argparse
from mlx_lm.server import ModelProvider, run as run_mlx_server


class MlxLmServer(threading.Thread):
    def __init__(
        self,
        model: AiModelConfig,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.host = host
        self.port = port

    def run(self):
        # download the model
        download_model(self.model)
        
        # init the model
        model_provider = ModelProvider(argparse.Namespace(
            model=self.model.registration.model_specs[0].model_id,
            adapter_path=None,
            host=self.host,
            port=self.port,
            allowed_origins="*",
            draft_model=None,
            num_draft_tokens=3,
            log_level="info",
            chat_template=self.model.registration.chat_template if self.model.registration else None or "",
            trust_remote_code=False,
            prompt_cache_size=10,
            use_default_chat_template=False,
            max_tokens=512,
            temp=0.0,
            top_p=1.0,

            # required by current mlx_lm.server internals
            top_k=0,
            min_p=0.0,
            chat_template_args={},
            decode_concurrency=32,
            prompt_concurrency=8,
            prefill_step_size=2048,
            prompt_cache_bytes=None,
            pipeline=False,
        )
        )
        # start the server
        run_mlx_server(
            host=self.host,
            port=self.port,
            model_provider=model_provider
        )
