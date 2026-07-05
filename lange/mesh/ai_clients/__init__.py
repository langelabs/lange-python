from lange.contracts import AiModelConfig

def start_ai_models(ai_models:list[AiModelConfig], platform:str):
    worker = []

    if platform == "Darwin" and len(ai_models) > 0:
        from .mlx_lm import MlxLmServer
        for index, ai_model in enumerate(ai_models):
            ai_worker = MlxLmServer(ai_model, port=8500+index)
            ai_worker.start()
            worker.append(ai_worker)

    elif platform == "Windows" or platform == "Linux":
        from .cuda import LlamaCppServer
        for index, ai_model in enumerate(ai_models):
            ai_worker = LlamaCppServer(ai_model, port=8500+index)
            ai_worker.start()
            worker.append(ai_worker)

    return worker