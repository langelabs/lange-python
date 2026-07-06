"""Tests for llama.cpp server configuration mapping."""

from pathlib import Path
from typing import Any

import pytest

from lange.contracts.ai_model import (
    AIModelSpecs,
    AiModelConfig,
    AiModelKVCacheConfig,
    AiModelRegistration,
    AiModelRuntimeConfig,
    AiModelVirtualEnvironment,
)
from lange.mesh.ai.servers import llama_cpp


def _model_config(
    *,
    context_window: int | None = None,
    runtime_config: AiModelRuntimeConfig | None = None,
    kv_cache_config: AiModelKVCacheConfig | None = None,
    chat_template: str | None = None,
) -> AiModelConfig:
    """Build an AI model config for llama.cpp mapping tests.

    :param context_window: Optional context window setting.
    :param runtime_config: Optional universal runtime settings.
    :param kv_cache_config: Optional KV-cache settings.
    :param chat_template: Optional registration chat template.
    :return: AI model config suitable for tests.
    """
    spec = AIModelSpecs(
        model_format="gguf",
        model_size_in_billions=7,
        quantization="q8_0",
        model_id="org/model",
        model_hub="huggingface",
        model_revision=None,
        model_uri=None,
        activated_size_in_billions=None,
        model_filename="model.gguf",
    )
    registration = AiModelRegistration(
        version=1,
        context_length=4096,
        model_name="test-model",
        model_lang=["en"],
        model_ability=["chat"],
        model_description="Test model",
        model_family="test",
        model_specs=[spec],
        chat_template=chat_template,
        stop_token_ids=None,
        stop=None,
        cache_config=None,
        virtualenv=AiModelVirtualEnvironment(
            packages=[],
            inherit_pip_config=False,
            index_url=None,
            extra_index_url=None,
            find_links=None,
            trusted_host=None,
            no_build_isolation=None,
        ),
        is_builtin=False,
        reasoning_start_tag=None,
        reasoning_end_tag=None,
    )

    return AiModelConfig(
        model_name="test-model",
        model_alias="test-alias",
        model_type="LLM",
        model_engine="llama.cpp",
        model_format="gguf",
        context_window=context_window,
        registration=registration,
        runtime_config=runtime_config,
        kv_cache_config=kv_cache_config,
    )


def _capture_model_settings(
    monkeypatch: pytest.MonkeyPatch,
    model: AiModelConfig,
) -> dict[str, Any]:
    """Run the server and capture the generated llama.cpp settings.

    :param monkeypatch: Pytest monkeypatch fixture.
    :param model: AI model config passed to the server.
    :return: Captured settings and uvicorn arguments.
    """
    captured: dict[str, Any] = {}

    def fake_create_app(*, server_settings: Any, model_settings: list[Any]) -> str:
        """Capture create_app settings.

        :param server_settings: Generated llama.cpp server settings.
        :param model_settings: Generated llama.cpp model settings.
        :return: Fake ASGI app marker.
        """
        captured["server_settings"] = server_settings
        captured["model_settings"] = model_settings
        return "fake-app"

    def fake_uvicorn_run(app: Any, *, host: str, port: int, log_level: str) -> None:
        """Capture uvicorn launch arguments.

        :param app: ASGI app passed to uvicorn.
        :param host: Server host.
        :param port: Server port.
        :param log_level: Uvicorn log level.
        """
        captured["uvicorn"] = {
            "app": app,
            "host": host,
            "port": port,
            "log_level": log_level,
        }

    monkeypatch.setattr(llama_cpp, "download_model", lambda _: Path("/tmp/model.gguf"))
    monkeypatch.setattr(
        llama_cpp.LlamaCppServer,
        "_resolve_model_path",
        staticmethod(lambda _: "/tmp/model.gguf"),
    )
    monkeypatch.setattr(llama_cpp, "create_app", fake_create_app)
    monkeypatch.setattr(llama_cpp.uvicorn, "run", fake_uvicorn_run)

    server = llama_cpp.LlamaCppServer(model, host="127.0.0.2", port=8501)
    server.run()

    return captured


def test_llama_cpp_minimal_config_builds_model_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert minimal config still creates valid llama.cpp model settings."""
    captured = _capture_model_settings(monkeypatch, _model_config())

    model_settings = captured["model_settings"][0]
    assert model_settings.model == "/tmp/model.gguf"
    assert model_settings.model_alias == "test-alias"
    assert model_settings.n_ctx == 2048
    assert model_settings.verbose is True
    assert captured["uvicorn"] == {
        "app": "fake-app",
        "host": "127.0.0.2",
        "port": 8501,
        "log_level": "info",
    }


def test_llama_cpp_maps_context_and_registration_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert portable context and registration fields map into settings."""
    captured = _capture_model_settings(
        monkeypatch,
        _model_config(context_window=8192, chat_template="chatml"),
    )

    model_settings = captured["model_settings"][0]
    assert model_settings.n_ctx == 8192
    assert model_settings.chat_format == "chatml"
    assert model_settings.hf_model_repo_id == "org/model"


def test_llama_cpp_maps_runtime_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert universal runtime settings map into llama.cpp model settings."""
    runtime_config = AiModelRuntimeConfig(
        gpu_layers=-1,
        main_gpu=1,
        tensor_split=[0.25, 0.75],
        batch_size=1024,
        physical_batch_size=256,
        cpu_threads=4,
        cpu_batch_threads=8,
        use_mmap=False,
        use_mlock=False,
        flash_attention=True,
        offload_kqv=False,
        cache_enabled=True,
        cache_type="disk",
        cache_size=1024,
        seed=123,
    )
    captured = _capture_model_settings(
        monkeypatch,
        _model_config(runtime_config=runtime_config),
    )

    model_settings = captured["model_settings"][0]
    assert model_settings.n_gpu_layers == -1
    assert model_settings.main_gpu == 1
    assert model_settings.tensor_split == [0.25, 0.75]
    assert model_settings.n_batch == 1024
    assert model_settings.n_ubatch == 256
    assert model_settings.n_threads == 4
    assert model_settings.n_threads_batch == 8
    assert model_settings.use_mmap is False
    assert model_settings.use_mlock is False
    assert model_settings.flash_attn is True
    assert model_settings.offload_kqv is False
    assert model_settings.cache is True
    assert model_settings.cache_type == "disk"
    assert model_settings.cache_size == 1024
    assert model_settings.seed == 123


def test_llama_cpp_maps_eight_bit_kv_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """Assert 8-bit KV cache maps to llama.cpp quantization constants."""
    captured = _capture_model_settings(
        monkeypatch,
        _model_config(kv_cache_config=AiModelKVCacheConfig(kv_bits=8)),
    )

    model_settings = captured["model_settings"][0]
    assert model_settings.type_k == llama_cpp.llama_cpp.GGML_TYPE_Q8_0
    assert model_settings.type_v == llama_cpp.llama_cpp.GGML_TYPE_Q8_0


def test_llama_cpp_rejects_unsupported_kv_cache_bits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert unsupported KV-cache bit depths are rejected clearly."""
    with pytest.raises(ValueError, match="Unsupported llama.cpp KV-cache bit depth"):
        _capture_model_settings(
            monkeypatch,
            _model_config(kv_cache_config=AiModelKVCacheConfig(kv_bits=4)),
        )
