from lange.contracts import AiModelConfig
from ..utils.get_platform import get_platform

def start_ai_models(ai_models:list[AiModelConfig]):
    workers = []
    platform = get_platform()

    if len(ai_models) == 0:
        return workers

    # MLX servers
    if platform == "Darwin":
        from .mlx_vlm import MlxVlmServer
        for index, ai_model in enumerate(ai_models):
            ai_worker = MlxVlmServer(ai_model, port=8500+index)
            ai_worker.start()
            workers.append(ai_worker)

    # LLAMA CPP Servers
    elif platform == "Windows" or platform == "Linux":
        from .llama_cpp import LlamaCppServer
        for index, ai_model in enumerate(ai_models):
            ai_worker = LlamaCppServer(ai_model, port=8500+index)
            ai_worker.start()
            workers.append(ai_worker)

    return workers