from typing import Literal

from pydantic import BaseModel


class AIModelSpecs(BaseModel):
    """Describe a concrete model artifact.

    The spec identifies the downloadable or local artifact used by an inference
    engine.
    """

    model_format: Literal["mlx", "vLLM", "gguf"]
    model_size_in_billions: int
    quantization: str
    model_id: str
    model_hub: Literal["huggingface"]
    model_revision: None
    model_uri: str | None
    activated_size_in_billions: int | None
    model_filename: str | None


class AiModelVirtualEnvironment(BaseModel):
    """Describe Python environment requirements for a model.

    These fields mirror model registration metadata and are not interpreted by
    every runtime.
    """

    packages: list[str]
    inherit_pip_config: bool
    index_url: str | None
    extra_index_url: str | None
    find_links: bool | None
    trusted_host: None
    no_build_isolation: None


class AiModelRegistration(BaseModel):
    """Describe portable model registration metadata.

    Registration data captures model capabilities and artifact variants that
    can be consumed by multiple inference engines.
    """

    version: int
    context_length: int
    model_name: str
    model_lang: list[str]
    model_ability: list[Literal["generate", "chat"]]
    model_description: str
    model_family: str
    model_specs: list[AIModelSpecs]
    chat_template: str | None
    stop_token_ids: None
    stop: None
    cache_config: None
    virtualenv: AiModelVirtualEnvironment
    is_builtin: bool

    reasoning_start_tag: str | None
    reasoning_end_tag: str | None


class AiModelKVCacheConfig(BaseModel):
    """Configure KV-cache quantization and sizing.

    The fields are engine-neutral. Individual engines may support only a subset
    of bit depths or quantization schemes.
    """

    kv_bits: float | None = 8
    kv_quant_scheme: Literal["uniform", "turboquant"] = "uniform"
    kv_group_size: int = 64
    kv_max_size: int | None = None


class AiModelRuntimeConfig(BaseModel):
    """Configure universal model runtime options.

    Every field is optional so inference engines can keep their native defaults
    unless a model config explicitly overrides them.
    """

    gpu_layers: int | None = None
    main_gpu: int | None = None
    tensor_split: list[float] | None = None
    batch_size: int | None = None
    physical_batch_size: int | None = None
    cpu_threads: int | None = None
    cpu_batch_threads: int | None = None
    use_mmap: bool | None = None
    use_mlock: bool | None = None
    flash_attention: bool | None = None
    offload_kqv: bool | None = None
    cache_enabled: bool | None = None
    cache_type: Literal["ram", "disk"] | None = None
    cache_size: int | None = None
    seed: int | None = None


class AiModelConfig(BaseModel):
    """Configure a model for AI worker startup.

    The config intentionally keeps model selection, registration metadata, and
    runtime overrides portable across supported inference engines.
    """

    # The name of the model to launch
    model_name: str

    # An easy alias for the model
    model_alias: str

    # The type of the model.
    model_type: Literal["LLM", "embedding", "image", "audio", "video"]

    # The size of the model in case multiple are available. e.G. gemma4 with 12b vs 31b
    size: str | int | None = None

    # The quantization of the model to use. Keep None for default.
    quantization: str | None = None

    # the context window of the model to use. if None it chooses default
    context_window: int | None = None

    # thinking
    enable_thinking: bool | None = None

    # engine configuration
    model_engine: Literal["MLX", "vLLM", "llama.cpp"] | None = None
    model_format: Literal["mlx", "vLLM", "gguf"] | None = None

    registration: AiModelRegistration | None

    # kv cache config
    kv_cache_config: AiModelKVCacheConfig | None = None

    # runtime configuration
    runtime_config: AiModelRuntimeConfig | None = None
