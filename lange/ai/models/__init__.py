from .google.embeddinggemma.mlx import MODEL as EMBEDDINGGEMMA_MLX
from .google.gemma4.mlx import MODEL as GEMMA4_MLX

MLX_MODELS = [
    GEMMA4_MLX,
    EMBEDDINGGEMMA_MLX,
]

__all__ = [
    "EMBEDDINGGEMMA_MLX",
    "GEMMA4_MLX",
    "MLX_MODELS",
]
