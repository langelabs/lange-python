if __name__ == '__main__':
    from lange.mesh.ai.models import MLX_MODELS
    from lange.mesh.ai.servers import start_ai_models
    start_ai_models(MLX_MODELS, blocking=True)