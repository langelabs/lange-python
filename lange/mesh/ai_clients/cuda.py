import threading
from pathlib import Path

import uvicorn
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ServerSettings

from ..utils.download_model import download_model
from lange.contracts import AiModelConfig


class LlamaCppServer(threading.Thread):
    def __init__(
        self,
        model: AiModelConfig,
        host: str = "127.0.0.1",
        port: int = 8080,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        n_threads_batch: int | None = None,
        chat_format: str | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_threads_batch = n_threads_batch
        self.chat_format = chat_format

    def run(self):
        # download the model
        model_path = download_model(self.model)

        # start the server
        app = create_app(
            server_settings=ServerSettings(
                host=self.host,
                port=self.port,
            ),
            model_settings=[
                ModelSettings(
                    model=self._resolve_model_path(model_path),
                    model_alias=self.model.model_alias,
                    n_ctx=self.model.registration.context_length,
                    n_gpu_layers=self.n_gpu_layers,
                    n_threads=self.n_threads,
                    n_threads_batch=self.n_threads_batch,
                    chat_format=self.chat_format,
                    verbose=True,
                )
            ],
        )

        uvicorn.run(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )

    @staticmethod
    def _resolve_model_path(model_path) -> str:
        """
        Resolve the downloaded GGUF model path.

        download_model(...) may return either:
        - a direct .gguf file path
        - a snapshot directory containing one .gguf file
        """

        path = Path(model_path).expanduser()

        if path.is_file():
            return str(path)

        if path.is_dir():
            gguf_files = sorted(path.glob("*.gguf"))

            if len(gguf_files) == 1:
                return str(gguf_files[0])

            if len(gguf_files) > 1:
                raise ValueError(
                    f"Multiple GGUF files found in {path}. "
                    f"Set model_uri to the exact .gguf file or adapt selection logic."
                )

        raise ValueError(f"No .gguf model file found at: {path}")