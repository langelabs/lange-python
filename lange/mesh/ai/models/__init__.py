from .google.gemma4.mlx import MODEL as GEMMA4_MLX
from .google.embeddinggemma.mlx import MODEL as EMBEDDINGGEMMA_MLX

MLX_MODELS = [
    GEMMA4_MLX,
    EMBEDDINGGEMMA_MLX
]

__all__ = [
    MLX_MODELS
]
