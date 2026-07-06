import threading
from pathlib import Path
from typing import Any

import uvicorn
from llama_cpp import llama_cpp
from llama_cpp.server.app import create_app
from llama_cpp.server.settings import ModelSettings, ServerSettings

from ..utils.download_model import download_model
from lange.contracts import AiModelConfig


class LlamaCppServer(threading.Thread):
    """Run a llama.cpp OpenAI-compatible server in a daemon thread."""

    def __init__(
        self,
        model: AiModelConfig,
        host: str = "127.0.0.1",
        port: int = 8080,
    ) -> None:
        """Create a llama.cpp server thread.

        :param model: AI model configuration to serve.
        :param host: Host address for the server.
        :param port: TCP port for the server.
        """
        super().__init__(daemon=True)
        self.model = model
        self.host = host
        self.port = port

    def run(self) -> None:
        """Download the configured model and run the llama.cpp server."""
        model_path = download_model(self.model)
        runtime_config = self.model.runtime_config

        model_setting_values: dict[str, Any] = {
            "model": self._resolve_model_path(model_path),
            "model_alias": self.model.model_alias,
            "verbose": True,
        }

        if self.model.context_window is not None:
            model_setting_values["n_ctx"] = self.model.context_window

        if self.model.registration is not None:
            if self.model.registration.chat_template:
                model_setting_values["chat_format"] = self.model.registration.chat_template

            if self.model.registration.model_specs:
                model_id = self.model.registration.model_specs[0].model_id
                if model_id:
                    model_setting_values["hf_model_repo_id"] = model_id

        if runtime_config is not None:
            runtime_mappings = {
                "gpu_layers": "n_gpu_layers",
                "main_gpu": "main_gpu",
                "tensor_split": "tensor_split",
                "batch_size": "n_batch",
                "physical_batch_size": "n_ubatch",
                "cpu_threads": "n_threads",
                "cpu_batch_threads": "n_threads_batch",
                "use_mmap": "use_mmap",
                "use_mlock": "use_mlock",
                "flash_attention": "flash_attn",
                "offload_kqv": "offload_kqv",
                "cache_enabled": "cache",
                "cache_type": "cache_type",
                "cache_size": "cache_size",
                "seed": "seed",
            }
            runtime_values = runtime_config.model_dump()

            for config_key, model_settings_key in runtime_mappings.items():
                config_value = runtime_values[config_key]
                if config_value is not None:
                    model_setting_values[model_settings_key] = config_value

        if self.model.kv_cache_config is not None:
            kv_bits = self.model.kv_cache_config.kv_bits

            if kv_bits == 8:
                model_setting_values["type_k"] = llama_cpp.GGML_TYPE_Q8_0
                model_setting_values["type_v"] = llama_cpp.GGML_TYPE_Q8_0
            elif kv_bits is not None:
                raise ValueError(
                    "Unsupported llama.cpp KV-cache bit depth: "
                    f"{kv_bits!r}. Only 8-bit KV cache is currently supported."
                )

        app = create_app(
            server_settings=ServerSettings(
                host=self.host,
                port=self.port,
            ),
            model_settings=[ModelSettings(**model_setting_values)],
        )

        uvicorn.run(
            app,
            host=self.host,
            port=self.port,
            log_level="info",
        )

    @staticmethod
    def _resolve_model_path(model_path: str | Path) -> str:
        """Resolve the downloaded GGUF model path.

        download_model(...) may return either:
        - a direct .gguf file path
        - a snapshot directory containing one .gguf file

        :param model_path: Downloaded file or snapshot directory path.
        :return: Resolved GGUF file path.
        :raises ValueError: If no single GGUF file can be resolved.
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
