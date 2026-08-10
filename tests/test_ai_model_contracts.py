"""Tests for AI model contract payloads."""

import importlib.util
import runpy
import sys
import types
from pathlib import Path

import pytest

from lange.ai.contracts import (
    AIModelSpecs,
    AiModelConfig,
    AiModelKVCacheConfig,
    AiModelRegistration,
    AiModelRuntimeConfig,
    AiModelVirtualEnvironment,
)


def test_kv_cache_defaults_are_scalar_values() -> None:
    """Assert KV-cache defaults are scalar values, not accidental tuples."""
    config = AiModelKVCacheConfig()

    assert config.kv_bits == 8
    assert config.kv_quant_scheme == "uniform"
    assert config.kv_group_size == 64
    assert config.kv_max_size is None


def test_runtime_config_defaults_leave_engine_defaults_unset() -> None:
    """Assert runtime settings are optional so engines can keep their defaults."""
    config = AiModelRuntimeConfig()

    assert config.gpu_layers is None
    assert config.main_gpu is None
    assert config.tensor_split is None
    assert config.batch_size is None
    assert config.physical_batch_size is None
    assert config.cpu_threads is None
    assert config.cpu_batch_threads is None
    assert config.use_mmap is None
    assert config.use_mlock is None
    assert config.flash_attention is None
    assert config.offload_kqv is None
    assert config.cache_enabled is None
    assert config.cache_type is None
    assert config.cache_size is None
    assert config.seed is None


def test_embedding_model_registration_accepts_fractional_size_and_embed_ability() -> (
    None
):
    """Assert embedding model metadata supports sub-billion parameter models."""
    spec = AIModelSpecs(
        model_format="mlx",
        model_size_in_billions=0.3,
        quantization="6bit",
        model_id="mlx-community/embeddinggemma-300m-6bit",
        model_hub="huggingface",
        model_revision=None,
        model_uri=None,
        activated_size_in_billions=None,
        model_filename=None,
    )
    registration = AiModelRegistration(
        version=1,
        context_length=2048,
        model_name="embeddinggemma-300m",
        model_lang=["en"],
        model_ability=["embed"],
        model_description="EmbeddingGemma 300M text embedding model in 6-bit MLX format.",
        model_family="embeddinggemma",
        model_specs=[spec],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
        cache_config=None,
        virtualenv=AiModelVirtualEnvironment(
            packages=[],
            inherit_pip_config=True,
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

    assert spec.model_size_in_billions == 0.3
    assert registration.model_ability == ["embed"]


def test_embeddinggemma_mlx_model_config_matches_huggingface_metadata() -> None:
    """Assert the EmbeddingGemma MLX config exposes the expected metadata."""
    from lange.ai.models.google.embeddinggemma.mlx import MODEL

    assert MODEL.model_name == "embeddinggemma-300m"
    assert MODEL.model_alias == "EMBEDDING_GEMMA_300M_MLX_6BIT"
    assert MODEL.model_type == "embedding"
    assert MODEL.size == 0.3
    assert MODEL.quantization == "6bit"
    assert MODEL.context_window == 2048
    assert MODEL.model_format == "mlx"
    assert MODEL.model_engine == "MLX"
    assert MODEL.registration is not None
    assert MODEL.registration.model_ability == ["embed"]
    assert MODEL.registration.context_length == 2048
    assert MODEL.registration.model_family == "embeddinggemma"
    assert (
        MODEL.registration.model_specs[0].model_id
        == "mlx-community/embeddinggemma-300m-6bit"
    )
    assert MODEL.registration.model_specs[0].model_size_in_billions == 0.3


def test_embeddinggemma_mlx_direct_run_does_not_shadow_mlx_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert direct script execution keeps third-party ``mlx`` importable.

    :param monkeypatch: Pytest monkeypatch fixture.
    :returns: ``None``.
    """
    module_path = (
        Path(__file__).resolve().parents[1]
        / "lange"
        / "ai"
        / "models"
        / "google"
        / "embeddinggemma"
        / "mlx.py"
    )
    script_dir = str(module_path.parent)
    fake_servers_module = types.ModuleType("lange.ai.servers")

    def fake_start_ai_models(models: list[AiModelConfig]) -> list[object]:
        """Inspect ``mlx`` resolution at the same boundary as server startup.

        :param models: Models passed by the direct-run entrypoint.
        :return: No workers, so the script exits without starting a server.
        """
        spec = importlib.util.find_spec("mlx")
        assert spec is not None
        assert spec.origin is None or not spec.origin.endswith("/embeddinggemma/mlx.py")
        assert spec.submodule_search_locations is not None
        assert models[0].model_type == "embedding"
        return []

    fake_servers_module.start_ai_models = fake_start_ai_models
    monkeypatch.setitem(sys.modules, "lange.ai.servers", fake_servers_module)
    monkeypatch.delitem(sys.modules, "mlx", raising=False)
    monkeypatch.delitem(sys.modules, "mlx.core", raising=False)
    monkeypatch.syspath_prepend(script_dir)

    runpy.run_path(str(module_path), run_name="__main__")
