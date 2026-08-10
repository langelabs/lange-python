from lange.ai.contracts import (
    AIModelSpecs,
    AiModelConfig,
    AiModelRegistration,
    AiModelVirtualEnvironment,
)

MODEL = AiModelConfig(
    model_name="embeddinggemma-300m",
    model_alias="EMBEDDING_GEMMA_300M_MLX_6BIT",
    model_type="embedding",
    size=0.3,
    quantization="6bit",
    context_window=2048,
    enable_thinking=None,
    model_format="mlx",
    model_engine="MLX",
    kv_cache_config=None,
    runtime_config=None,
    registration=AiModelRegistration(
        version=1,
        context_length=2048,
        model_name="embeddinggemma-300m",
        model_lang=["en"],
        model_ability=["embed"],
        model_description="EmbeddingGemma 300M text embedding model in 6-bit MLX format.",
        model_family="embeddinggemma",
        model_specs=[
            AIModelSpecs(
                model_format="mlx",
                model_size_in_billions=0.3,
                quantization="6bit",
                model_id="mlx-community/embeddinggemma-300m-6bit",
                model_hub="huggingface",
                model_uri=None,
                model_revision=None,
                activated_size_in_billions=None,
                model_filename=None,
            )
        ],
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
    ),
)

__all__ = ["MODEL"]
