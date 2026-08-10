import uvicorn
from fastapi import FastAPI, HTTPException
from mlx_embeddings import load
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage
from pydantic import BaseModel

from ..__base import InferenceServer


class EmbeddingRequest(BaseModel):
    """Represent an OpenAI-compatible embeddings request body."""

    input: str | list[str]
    model: str | None = None
    encoding_format: str | None = None
    dimensions: int | None = None
    user: str | None = None


class MLXEmbeddingsServer(InferenceServer):
    """Run an MLX embeddings model behind an OpenAI-compatible endpoint."""

    def run(self) -> None:
        """Download, load, and serve the configured MLX embeddings model."""
        model_path = self.download()
        model, tokenizer = load(str(model_path))
        fastapi_app = FastAPI()

        @fastapi_app.post("/v1/embeddings")
        async def create_embeddings(
            request: EmbeddingRequest,
        ) -> CreateEmbeddingResponse:
            """Create embeddings for string input.

            :param request: OpenAI-compatible embedding request body.
            :return: OpenAI-compatible embedding response.
            :raises HTTPException: If the request asks for unsupported behavior.
            """
            if request.encoding_format not in (None, "float"):
                raise HTTPException(
                    status_code=400,
                    detail="Only float embedding encoding is supported.",
                )

            if request.dimensions is not None:
                raise HTTPException(
                    status_code=400,
                    detail="Embedding dimensions override is not supported.",
                )

            if isinstance(request.input, str):
                texts = [request.input]
            else:
                texts = request.input

            if not texts:
                raise HTTPException(
                    status_code=400,
                    detail="Embedding input must not be empty.",
                )

            if not all(isinstance(text, str) for text in texts):
                raise HTTPException(
                    status_code=400,
                    detail="Embedding input must be a string or list of strings.",
                )

            max_length = self.model.context_window or 512
            tokenized = tokenizer(
                texts,
                return_tensors="mlx",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]
            output = model(input_ids, attention_mask)
            embeddings_output = output.text_embeds

            if hasattr(embeddings_output, "tolist"):
                raw_embeddings = embeddings_output.tolist()
            else:
                raw_embeddings = embeddings_output

            if (
                len(texts) == 1
                and raw_embeddings
                and isinstance(raw_embeddings[0], float)
            ):
                raw_embeddings = [raw_embeddings]

            data = [
                Embedding(
                    embedding=[float(value) for value in embedding],
                    index=index,
                    object="embedding",
                )
                for index, embedding in enumerate(raw_embeddings)
            ]

            token_values = (
                input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
            )
            prompt_tokens = sum(len(tokens) for tokens in token_values)

            return CreateEmbeddingResponse(
                data=data,
                model=self.model.model_alias,
                object="list",
                usage=Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
            )

        uvicorn_server = uvicorn.Server(
            uvicorn.Config(fastapi_app, host=self.host, port=self.port)
        )
        self.uvicorn_server = uvicorn_server
        uvicorn_server.run()

    def stop(self) -> None:
        """Request graceful shutdown from the active embeddings server."""
        if self.uvicorn_server is not None:
            self.uvicorn_server.should_exit = True
