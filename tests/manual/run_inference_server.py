from lange.mesh.ai_clients.mlx_vlm import MlxVlmServer
from lange.contracts.ai_model import (
    AiModelConfig,
    AiModelRegistration,
    AiModelVirtualEnvironment,
    AIModelSpecs,
)
import asyncio
from dotenv import load_dotenv

load_dotenv()

MODEL = AiModelConfig(
    model_name="gemma-4-12b-it",
    model_alias="LL_C_0",
    model_type="LLM",
    size=12,
    quantization="8bit",
    enable_thinking=True,
    model_format="mlx",
    model_engine="MLX",
    registration=AiModelRegistration(
        version=2,
        context_length=262144,
        model_name="gemma-4-12b-it",
        model_lang=["en"],
        model_ability=["generate", "chat"],
        model_description="Gemma 4 12B instruction model in MLX format.",
        model_family="gemma-4",
        model_specs=[
            AIModelSpecs(
                model_format="mlx",
                model_size_in_billions=12,
                quantization="8bit",
                model_id="mlx-community/gemma-4-12B-it-8bit",
                model_hub="huggingface",
                model_uri=None,
                model_revision=None,
                activated_size_in_billions=None,
                model_filename=None
            )
        ],
        chat_template=None,
        stop_token_ids=None,
        stop=None,
        reasoning_start_tag=None,
        reasoning_end_tag=None,
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
    ),
)


async def test():
    inference = MlxVlmServer(MODEL)
    inference.start()
    inference.join()


if __name__ == "__main__":
    asyncio.run(test())
