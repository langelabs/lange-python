from lange.contracts import AiModelConfig
from ..utils.get_platform import get_platform
from .__base import InferenceServer


def start_ai_models(models: list[AiModelConfig], blocking: bool = False) -> list[InferenceServer]:
    workers = []
    platform = get_platform()

    if len(models) == 0:
        return workers

    # MLX servers
    if platform == "Darwin":
        from .mlx import MlxLLMServer, MLXEmbeddingsServer, MlxImageServer

        for index, model in enumerate(models):
            if model.model_type == "LLM":
                worker = MlxLLMServer(model, port=8500 + index)
            elif model.model_type == "embedding":
                worker = MLXEmbeddingsServer(model, port=8500 + index)
            elif model.model_type == "image":
                worker = MlxImageServer(model, port=8500 + index)
            else:
                raise NotImplementedError(f"Unsupported model type for MLX: {model.model_type}")

            worker.start()
            workers.append(worker)

    # LLAMA CPP Servers
    elif platform == "Windows" or platform == "Linux":
        from .llama_cpp import LlamaCppServer
        for index, ai_model in enumerate(models):
            ai_worker = LlamaCppServer(ai_model, port=8500 + index)
            ai_worker.start()
            workers.append(ai_worker)

    if blocking:  # block until all finished
        for worker in workers:
            worker.join()

    return workers
