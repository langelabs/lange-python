import os
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

from ....contracts import AiModelConfig


def get_primary_model_spec(model: AiModelConfig):
    """
    Return the primary concrete model artifact spec.

    :param model: AI model configuration.
    :return: First AIModelSpecs entry.
    :raises ValueError: If no model specs are configured.
    """

    if not model.registration.model_specs:
        raise ValueError("model.registration.model_specs is empty.")

    return model.registration.model_specs[0]


def resolve_local_model_uri(model: AiModelConfig) -> Path | None:
    """
    Return a local model path if model_uri points to an existing file or directory.

    :param model: AI model configuration.
    :return: Resolved local path, or None.
    """

    spec = get_primary_model_spec(model)

    if not spec.model_uri:
        return None

    path = Path(spec.model_uri).expanduser()

    if path.exists():
        return path.resolve()

    return None


def configure_huggingface_downloads(
    *,
    hf_token: str | None = None,
    enable_hf_transfer: bool = True,
) -> str | None:
    """
    Configure Hugging Face download environment.

    :param hf_token: Optional Hugging Face token.
    :param enable_hf_transfer: Enables hf_transfer if installed.
    :return: Resolved Hugging Face token, if available.
    """

    if enable_hf_transfer:
        # Requires: uv add "huggingface_hub[hf_transfer]"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")

    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token

    return token


def download_model(
    model: AiModelConfig,
    *,
    cache_dir: str | Path | None = None,
    hf_token: str | None = None,
    enable_hf_transfer: bool = True,
    force_download: bool = False,
    revision: str | None = None,
) -> Path:
    """
    Download a model from Hugging Face or return a local model path.

    For MLX models, this downloads the full repository snapshot.
    For GGUF models with model_filename set, this downloads only that file.

    :param model: AI model configuration.
    :param cache_dir: Optional Hugging Face cache directory.
    :param hf_token: Optional Hugging Face token. If omitted, HF_TOKEN from env is used.
    :param enable_hf_transfer: Enables faster Hugging Face downloads if hf_transfer is installed.
    :param force_download: If True, re-download files even if cached.
    :param revision: Optional model revision. If omitted, the config revision is used.
    :return: Local path to the downloaded model directory or file.
    """

    local_uri = resolve_local_model_uri(model)
    if local_uri is not None:
        return local_uri

    spec = get_primary_model_spec(model)

    if not spec.model_id:
        raise ValueError("No model_id found in model.registration.model_specs[0].")

    token = configure_huggingface_downloads(
        hf_token=hf_token,
        enable_hf_transfer=enable_hf_transfer,
    )

    resolved_revision = revision or spec.model_revision
    resolved_cache_dir = str(cache_dir) if cache_dir is not None else None

    if model.model_format == "gguf":
        if not spec.model_filename:
            raise ValueError(
                "GGUF models require spec.model_filename so only the target .gguf file is downloaded."
            )

        downloaded_file = hf_hub_download(
            repo_id=spec.model_id,
            filename=spec.model_filename,
            revision=resolved_revision,
            cache_dir=resolved_cache_dir,
            token=token,
            force_download=force_download,
            local_files_only=False,
        )

        return Path(downloaded_file).resolve()

    if model.model_format == "mlx":
        downloaded_snapshot = snapshot_download(
            repo_id=spec.model_id,
            revision=resolved_revision,
            cache_dir=resolved_cache_dir,
            token=token,
            force_download=force_download,
            local_files_only=False,
        )

        return Path(downloaded_snapshot).resolve()

    raise ValueError(f"Unsupported model_format: {model.model_format!r}") 