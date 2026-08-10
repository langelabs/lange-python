import abc
import subprocess
import threading
from pathlib import Path

from tqdm.asyncio import tqdm
from huggingface_hub import hf_hub_download, snapshot_download

from lange.contracts.ai_model import AiModelConfig


class InferenceServer(threading.Thread, abc.ABC):

    def __init__(
            self,
            model: AiModelConfig,
            host: str = "127.0.0.1",
            port: int = 8080,
            model_dir: str | None = None,
            huggingface_token: str | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self.model = model
        self.host = host
        self.port = port
        self.model_dir = model_dir
        self.process: subprocess.Popen | None = None
        self.huggingface_token = huggingface_token

    def download(self,
                 hf_token: str | None = None,
                 force_download: bool = False,
                 revision: str | None = None,
                 show_progress_bar: bool = True,
                 ) -> Path:
        """
        Download a model from Hugging Face or return a local model path.

        For MLX models, this downloads the full repository snapshot.
        For GGUF models with model_filename set, this downloads only that file.
        :param hf_token: Optional Hugging Face token. If omitted, HF_TOKEN from env is used.
        :param force_download: If True, re-download files even if cached.
        :param revision: Optional model revision. If omitted, the config revision is used.
        :param show_progress_bar: If True, show tqdm progress bars in the terminal.
        :return: Local path to the downloaded model directory or file.
        """

        # resolve the model spec
        if not self.model.registration or not self.model.registration.model_specs:
            raise ValueError("model.registration.model_specs is empty.")
        model_spec = self.model.registration.model_specs[0]

        if not model_spec.model_id:
            raise ValueError("No model_id found in model.registration.model_specs[0].")

        resolved_revision = revision or model_spec.model_revision
        tqdm_class = tqdm if show_progress_bar else None

        if self.model.model_format == "gguf":
            if not model_spec.model_filename:
                raise ValueError(
                    "GGUF models require spec.model_filename so only the target .gguf file is downloaded."
                )

            downloaded_file = hf_hub_download(
                repo_id=model_spec.model_id,
                filename=model_spec.model_filename,
                revision=resolved_revision,
                cache_dir=self.model_dir,
                token=self.huggingface_token,
                local_files_only=False,
                tqdm_class=tqdm_class,
            )
            return Path(downloaded_file).resolve()

        if self.model.model_format == "mlx":
            downloaded_snapshot = snapshot_download(
                repo_id=model_spec.model_id,
                revision=resolved_revision,
                cache_dir=self.model_dir,
                token=self.huggingface_token,
                local_files_only=False,
                tqdm_class=tqdm_class)
            return Path(downloaded_snapshot).resolve()

        raise ValueError(f"Unsupported model_format: {self.model.model_format!r}")

    @abc.abstractmethod
    def run(self) -> None:
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        pass
