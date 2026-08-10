"""Tests for the MLX embeddings OpenAI-compatible server."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from lange.ai.contracts import AiModelConfig
from lange.ai import plugin as ai_plugin
from lange.ai.servers.mlx import mlx_embeddings as mlx_embeddings_server


class _FakeEmbeddingArray:
    """Represent generated embeddings with the array API used by the server."""

    def __init__(self, values: list[list[float]]) -> None:
        """Initialize the fake embedding array.

        :param values: Embedding vectors returned from ``tolist``.
        """
        self._values = values

    def tolist(self) -> list[list[float]]:
        """Return embedding vectors as nested Python lists.

        :return: Embedding vectors.
        """
        return self._values


class _FakeEmbeddingOutput:
    """Represent model output that exposes normalized text embeddings."""

    def __init__(self, values: list[list[float]]) -> None:
        """Initialize the fake model output.

        :param values: Embedding vectors exposed as ``text_embeds``.
        """
        self.text_embeds = _FakeEmbeddingArray(values)


class _FakeTokenIds:
    """Represent token IDs with the array API used for usage accounting."""

    def __init__(self, values: list[list[int]]) -> None:
        """Initialize fake token IDs.

        :param values: Token IDs returned from ``tolist``.
        """
        self._values = values

    def tolist(self) -> list[list[int]]:
        """Return token IDs as nested Python lists.

        :return: Token IDs.
        """
        return self._values


def _model_config() -> AiModelConfig:
    """Build a minimal embedding model config for server tests.

    :return: Embedding model config.
    """
    return AiModelConfig(
        model_name="embeddinggemma-300m",
        model_alias="EMBEDDING_GEMMA_300M_MLX_6BIT",
        model_type="embedding",
        context_window=2048,
        model_engine="MLX",
        model_format="mlx",
        registration=None,
    )


def _capture_app(monkeypatch: pytest.MonkeyPatch) -> tuple[TestClient, dict[str, Any]]:
    """Run the server and capture the generated FastAPI app.

    :param monkeypatch: Pytest monkeypatch fixture.
    :return: Test client and captured server state.
    """
    captured: dict[str, Any] = {"model_calls": [], "tokenizer_calls": []}

    def fake_download(_: Any) -> Path:
        """Return a fake downloaded model path.

        :param _: Server instance.
        :return: Fake model path.
        """
        return Path("/tmp/embedding-model")

    class FakeModel:
        """Fake EmbeddingGemma model used to capture positional calls."""

        def __call__(
            self, *args: _FakeTokenIds, **kwargs: _FakeTokenIds
        ) -> _FakeEmbeddingOutput:
            """Capture the positional embedding model call.

            :param args: Positional model inputs.
            :param kwargs: Keyword model inputs.
            :return: Fake embedding output.
            """
            captured["model_calls"].append(
                {
                    "args": args,
                    "kwargs": kwargs,
                }
            )
            input_ids = args[0] if args else kwargs["input_ids"]
            return _FakeEmbeddingOutput(
                [
                    [float(index), float(index) + 0.25]
                    for index, _ in enumerate(input_ids.tolist())
                ]
            )

    class FakeTokenizer:
        """Fake tokenizer used to capture tokenization arguments."""

        def __call__(
            self,
            texts: list[str],
            *,
            return_tensors: str,
            padding: bool,
            truncation: bool,
            max_length: int,
        ) -> dict[str, _FakeTokenIds]:
            """Capture tokenization and return fake model inputs.

            :param texts: Text inputs.
            :param return_tensors: Tensor backend name.
            :param padding: Whether padding is enabled.
            :param truncation: Whether truncation is enabled.
            :param max_length: Maximum token length.
            :return: Fake tokenized model inputs.
            """
            captured["tokenizer_calls"].append(
                {
                    "texts": texts,
                    "return_tensors": return_tensors,
                    "padding": padding,
                    "truncation": truncation,
                    "max_length": max_length,
                }
            )
            return {
                "input_ids": _FakeTokenIds(
                    [[index, index + 10] for index, _ in enumerate(texts)]
                ),
                "attention_mask": _FakeTokenIds([[1, 1] for _ in texts]),
            }

    def fake_load(path_or_hf_repo: str) -> tuple[FakeModel, FakeTokenizer]:
        """Capture the model path passed to mlx-embeddings.

        :param path_or_hf_repo: Local model path or Hugging Face repo.
        :return: Fake model and tokenizer.
        """
        captured["loaded_path"] = path_or_hf_repo
        return FakeModel(), FakeTokenizer()

    def fake_uvicorn_config(app: Any, *, host: str, port: int) -> Any:
        """Capture uvicorn configuration.

        :param app: FastAPI app passed to uvicorn.
        :param host: Server host.
        :param port: Server port.
        :returns: Captured uvicorn configuration.
        """
        captured["app"] = app
        captured["host"] = host
        captured["port"] = port
        return SimpleNamespace(app=app, host=host, port=port)

    class FakeUvicornServer:
        """Avoid binding a network socket in API tests."""

        def __init__(self, config: Any) -> None:
            """Store the fake uvicorn config.

            :param config: Uvicorn server configuration.
            """
            self.config = config
            self.should_exit = False

        def run(self) -> None:
            """Return without binding a network socket."""

    monkeypatch.setattr(
        mlx_embeddings_server.MLXEmbeddingsServer, "download", fake_download
    )
    monkeypatch.setattr(mlx_embeddings_server, "load", fake_load)
    monkeypatch.setattr(mlx_embeddings_server.uvicorn, "Config", fake_uvicorn_config)
    monkeypatch.setattr(mlx_embeddings_server.uvicorn, "Server", FakeUvicornServer)

    server = mlx_embeddings_server.MLXEmbeddingsServer(
        _model_config(),
        host="127.0.0.2",
        port=8503,
    )
    server.run()

    return TestClient(captured["app"]), captured


def test_mlx_embeddings_server_loads_downloaded_model_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert the server loads the downloaded MLX model snapshot."""
    _, captured = _capture_app(monkeypatch)

    assert captured["loaded_path"] == "/tmp/embedding-model"
    assert captured["host"] == "127.0.0.2"
    assert captured["port"] == 8503


def test_mlx_embeddings_server_returns_embedding_for_single_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert a string input returns an OpenAI-compatible embedding response."""
    client, captured = _capture_app(monkeypatch)

    response = client.post(
        "/v1/embeddings",
        json={"model": "ignored-client-model", "input": "hello"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "object": "list",
        "model": "EMBEDDING_GEMMA_300M_MLX_6BIT",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 0.25]}],
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }
    assert captured["tokenizer_calls"] == [
        {
            "texts": ["hello"],
            "return_tensors": "mlx",
            "max_length": 2048,
            "padding": True,
            "truncation": True,
        }
    ]
    model_call = captured["model_calls"][0]
    assert model_call["kwargs"] == {}
    assert len(model_call["args"]) == 2
    assert model_call["args"][0].tolist() == [[0, 10]]
    assert model_call["args"][1].tolist() == [[1, 1]]


def test_mlx_embeddings_server_returns_ordered_embeddings_for_list_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert list input returns one indexed embedding per input item."""
    client, _ = _capture_app(monkeypatch)

    response = client.post(
        "/v1/embeddings",
        json={"input": ["alpha", "beta"]},
    )

    assert response.status_code == 200
    assert response.json()["data"] == [
        {"object": "embedding", "index": 0, "embedding": [0.0, 0.25]},
        {"object": "embedding", "index": 1, "embedding": [1.0, 1.25]},
    ]


def test_mlx_embeddings_server_rejects_base64_encoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert unsupported base64 embeddings fail with a clear client error."""
    client, _ = _capture_app(monkeypatch)

    response = client.post(
        "/v1/embeddings",
        json={"input": "hello", "encoding_format": "base64"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Only float embedding encoding is supported."


def test_mlx_embeddings_server_rejects_empty_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert empty embedding batches fail with a clear client error."""
    client, _ = _capture_app(monkeypatch)

    response = client.post("/v1/embeddings", json={"input": []})

    assert response.status_code == 400
    assert response.json()["detail"] == "Embedding input must not be empty."


def test_mlx_embeddings_server_rejects_dimensions_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert dimensions overrides fail instead of truncating vectors."""
    client, _ = _capture_app(monkeypatch)

    response = client.post(
        "/v1/embeddings",
        json={"input": "hello", "dimensions": 128},
    )

    assert response.status_code == 400
    assert (
        response.json()["detail"] == "Embedding dimensions override is not supported."
    )


def test_mlx_embeddings_server_rejects_tokenized_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert token-array inputs fail validation instead of being embedded."""
    client, _ = _capture_app(monkeypatch)

    response = client.post("/v1/embeddings", json={"input": [1, 2, 3]})

    assert response.status_code == 422


def test_server_factory_uses_embedding_server_for_embedding_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert Darwin embedding models start with the MLX embeddings server.

    :param monkeypatch: Pytest monkeypatch fixture.
    """
    started_models: list[AiModelConfig] = []

    class FakeEmbeddingServer:
        """Fake MLX embeddings server used to capture startup."""

        def __init__(
            self,
            model: AiModelConfig,
            *,
            host: str,
            port: int,
        ) -> None:
            """Capture model and port.

            :param model: Model config passed to the worker.
            :param host: Worker bind host.
            :param port: Worker port.
            """
            self.model = model
            self.host = host
            self.port = port

        def start(self) -> None:
            """Record that the fake worker was started."""
            started_models.append(self.model)

    monkeypatch.setattr(ai_plugin, "get_platform", lambda: "Darwin")
    monkeypatch.setattr(
        "lange.ai.servers.mlx.MLXEmbeddingsServer",
        FakeEmbeddingServer,
    )

    worker = ai_plugin.create_inference_server(
        _model_config(),
        host="127.0.0.1",
        port=8500,
    )
    worker.start()

    assert isinstance(worker, FakeEmbeddingServer)
    assert worker.host == "127.0.0.1"
    assert worker.port == 8500
    assert started_models == [_model_config()]


def test_mlx_embeddings_stop_signals_uvicorn_server() -> None:
    """Request graceful shutdown from the active uvicorn server."""
    server = mlx_embeddings_server.MLXEmbeddingsServer(_model_config())
    uvicorn_server = SimpleNamespace(should_exit=False)
    server.uvicorn_server = uvicorn_server

    server.stop()

    assert uvicorn_server.should_exit


def test_mlx_embeddings_run_retains_uvicorn_server(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retain the uvicorn server so plugin shutdown can signal it."""
    created: list[Any] = []

    class FakeUvicornServer:
        """Record uvicorn server construction and execution."""

        def __init__(self, config: Any) -> None:
            """Capture the uvicorn config.

            :param config: Uvicorn server configuration.
            """
            self.config = config
            self.ran = False
            self.should_exit = False
            created.append(self)

        def run(self) -> None:
            """Record server execution."""
            self.ran = True

    monkeypatch.setattr(
        mlx_embeddings_server.MLXEmbeddingsServer,
        "download",
        lambda _: Path("/tmp/embedding-model"),
    )
    monkeypatch.setattr(
        mlx_embeddings_server,
        "load",
        lambda _: (object(), object()),
    )
    monkeypatch.setattr(
        mlx_embeddings_server.uvicorn,
        "Config",
        lambda app, **kwargs: SimpleNamespace(app=app, **kwargs),
    )
    monkeypatch.setattr(mlx_embeddings_server.uvicorn, "Server", FakeUvicornServer)
    server = mlx_embeddings_server.MLXEmbeddingsServer(_model_config())

    server.run()

    assert server.uvicorn_server is created[0]
    assert created[0].ran
