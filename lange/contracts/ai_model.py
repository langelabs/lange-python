from typing import Literal

from pydantic import BaseModel


class AIModelSpecs(BaseModel):
    model_format: Literal["mlx", "vLLM", "gguf"]
    model_size_in_billions: int
    quantization: str
    model_id: str
    model_hub: Literal["huggingface"]
    model_revision: None
    model_uri:str|None
    activated_size_in_billions: int|None
    model_filename: str|None


class AiModelVirtualEnvironment(BaseModel):
    packages: list[str]
    inherit_pip_config: bool
    index_url: str | None
    extra_index_url: str | None
    find_links: bool | None
    trusted_host: None
    no_build_isolation: None


class AiModelRegistration(BaseModel):
    version: int
    context_length: int
    model_name: str
    model_lang: list[str]
    model_ability: list[Literal["generate", "chat"]]
    model_description: str
    model_family: str
    model_specs: list[AIModelSpecs]
    chat_template: str|None
    stop_token_ids: None
    stop: None
    cache_config: None
    virtualenv: AiModelVirtualEnvironment
    is_builtin: bool

    reasoning_start_tag: str|None
    reasoning_end_tag: str|None

class AiModelConfig(BaseModel):
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
