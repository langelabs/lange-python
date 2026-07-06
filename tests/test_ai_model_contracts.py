"""Tests for AI model contract payloads."""

from lange.contracts.ai_model import AiModelKVCacheConfig, AiModelRuntimeConfig


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
