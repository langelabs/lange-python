from typing import Literal
from pydantic import BaseModel
from .ai_model import AiModelConfig

class MeshWorkerRegistration(BaseModel):
    """
    Registration payload sent by a mesh worker during hello.
    """

    name: str
    timeout: float
    platform: Literal["Darwin", "Linux", "Windows"]

    is_ai_worker: bool
    is_relay_worker: bool

class MeshWorkerConfig(BaseModel):
    relay_address: str | None
    ai_models: list[AiModelConfig]