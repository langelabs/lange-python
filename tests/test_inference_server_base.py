"""Tests for base inference server behavior."""

from pathlib import Path
from typing import Any

import pytest

from lange.contracts.ai_model import (
    AIModelSpecs,
    AiModelConfig,
    AiModelRegistration,
    AiModelVirtualEnvironment,
)
from lange.mesh.ai.servers import __base


class FakeInferenceServer(__base.InferenceServer):
    """Concrete inference server for base-class tests."""

    def run(self) -> None:
        """Satisfy the abstract run method."""

    def stop(self) -> None:
        """Satisfy the abstract stop method."""


def _mlx_model_config() -> AiModelConfig:
    """Build a minimal MLX model config for download tests.

    :return: MLX model config.
    """
    spec = AIModelSpecs(
        model_format="mlx",
        model_size_in_billions=0.3,
        quantization="6bit",
        model_id="org/model",
        model_hub="huggingface",
        model_revision=None,
        model_uri=None,
        activated_size_in_billions=None,
        model_filename=None,
    )
    registration = AiModelRegistration(
        version=1,
        context_length=2048,
        model_name="test-model",
        model_lang=["en"],
        model_ability=["embed"],
        model_description="Test embedding model",
        model_family="test",
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
    return AiModelConfig(
        model_name="test-model",
        model_alias="test-model",
        model_type="embedding",
        model_engine="MLX",
        model_format="mlx",
        registration=registration,
    )


def test_download_uses_callable_tqdm_progress_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert download passes Hugging Face a callable tqdm progress class.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    captured: dict[str, Any] = {}

    def fake_snapshot_download(**kwargs: Any) -> str:
        """Capture snapshot download arguments.

        :param kwargs: Snapshot download arguments.
        :return: Fake downloaded snapshot path.
        """
        captured.update(kwargs)
        return "/tmp/model-snapshot"

    monkeypatch.setattr(__base, "snapshot_download", fake_snapshot_download)

    result = FakeInferenceServer(_mlx_model_config()).download(show_progress_bar=True)

    assert result == Path("/tmp/model-snapshot").resolve()
    assert callable(captured["tqdm_class"])
