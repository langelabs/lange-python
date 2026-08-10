"""AI inference server startup utilities."""

from lange.ai.contracts import AiModelConfig

from .__base import InferenceServer


def start_ai_models(
    models: list[AiModelConfig],
    blocking: bool = False,
    *,
    host: str = "127.0.0.1",
    start_port: int = 8500,
) -> list[InferenceServer]:
    """Create and start inference servers for a list of models.

    :param models: Ordered model configurations to serve.
    :param blocking: Whether to join every server after startup.
    :param host: Bind host shared by all servers.
    :param start_port: Port assigned to the first model.
    :returns: Started inference server threads.
    """
    from lange.ai.plugin import create_inference_server

    workers: list[InferenceServer] = []
    for index, model in enumerate(models):
        worker = create_inference_server(
            model,
            host=host,
            port=start_port + index,
        )
        worker.start()
        workers.append(worker)

    if blocking:
        for worker in workers:
            worker.join()
    return workers


__all__ = ["InferenceServer", "start_ai_models"]
