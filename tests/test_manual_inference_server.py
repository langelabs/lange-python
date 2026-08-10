"""Tests for manual inference server configuration."""

import importlib.util
from pathlib import Path


def test_llama_cpp_manual_config_uses_base_gguf_model() -> None:
    """Assert the manual llama.cpp server does not load an MTP companion model."""
    module_path = Path(__file__).parent / "manual" / "run_inference_server.py"
    spec = importlib.util.spec_from_file_location(
        "manual_run_inference_server",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model = module.LLAMA_CPP_CHAT_MODEL
    model_spec = model.registration.model_specs[0]

    assert model.quantization == "Q8_0"
    assert model_spec.quantization == "Q8_0"
    assert model_spec.model_filename == "gemma-4-12b-it-Q8_0.gguf"
