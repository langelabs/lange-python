import os
import subprocess
import threading
from pathlib import Path

from ..utils.download_model import download_model
from lange.contracts import AiModelConfig


class MlxVlmServer(threading.Thread):
    def __init__(
        self,
        model: AiModelConfig,
        host: str = "127.0.0.1",
        port: int = 8080,
        *,
        adapter_path: str | None = None,

        # Lower than mlx-vlm default 20. Good for memory.
        # Increase only if you repeatedly ask about the same images.
        vision_cache_size: int = 6,

        # KV cache tuning. 8 is a safe default.
        # For very long context / memory pressure try 4 or 3.5 + turboquant.
        kv_bits: float | None = 8,
        kv_quant_scheme: str = "uniform",  # "uniform" or "turboquant"
        kv_group_size: int = 64,
        max_kv_size: int | None = None,

        # Thinking costs tokens and latency. Keep off for serving.
        enable_thinking: bool = False,
        thinking_budget: int | None = None,

        # Only enable if the model really requires it.
        trust_remote_code: bool = False,

        log_level: str = "info",

        # APC can help when many requests share the same long prefix,
        # but mlx-vlm skips APC when KV quantization is enabled.
        apc_enabled: bool = False,
        apc_num_blocks: int = 2048,
        apc_disk_path: str | Path | None = None,
        apc_disk_max_gb: int = 0,
    ) -> None:
        super().__init__(daemon=True)

        self.model = model
        self.host = host
        self.port = port
        self.adapter_path = adapter_path

        self.vision_cache_size = vision_cache_size

        self.kv_bits = kv_bits
        self.kv_quant_scheme = kv_quant_scheme
        self.kv_group_size = kv_group_size
        self.max_kv_size = max_kv_size

        self.enable_thinking = enable_thinking
        self.thinking_budget = thinking_budget

        self.trust_remote_code = trust_remote_code
        self.log_level = log_level

        self.apc_enabled = apc_enabled
        self.apc_num_blocks = apc_num_blocks
        self.apc_disk_path = Path(apc_disk_path).expanduser() if apc_disk_path else None
        self.apc_disk_max_gb = apc_disk_max_gb

        self.process: subprocess.Popen | None = None

    def run(self) -> None:
        download_model(self.model)

        env = os.environ.copy()

        if self.trust_remote_code:
            env["MLX_TRUST_REMOTE_CODE"] = "true"

        # APC is useful for repeated long prefixes, but mlx-vlm docs say APC is skipped
        # when KV-cache quantization is enabled.
        if self.apc_enabled and self.kv_bits is None:
            env["APC_ENABLED"] = "1"
            env["APC_NUM_BLOCKS"] = str(self.apc_num_blocks)

            if self.apc_disk_path:
                env["APC_DISK_PATH"] = str(self.apc_disk_path)
                env["APC_DISK_MAX_GB"] = str(self.apc_disk_max_gb)
        else:
            env["APC_ENABLED"] = "0"

        cmd = [
            "python",
            "-m",
            "mlx_vlm.server",
            "--model",
            self.model.registration.model_specs[0].model_id,
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--vision-cache-size",
            str(self.vision_cache_size),
            "--log-level",
            self.log_level.upper(),
        ]

        if self.adapter_path:
            cmd.extend(["--adapter-path", self.adapter_path])

        if self.trust_remote_code:
            cmd.append("--trust-remote-code")

        if self.enable_thinking:
            cmd.append("--enable-thinking")

            if self.thinking_budget is not None:
                cmd.extend(["--thinking-budget", str(self.thinking_budget)])

        if self.kv_bits is not None:
            cmd.extend(["--kv-bits", str(self.kv_bits)])
            cmd.extend(["--kv-quant-scheme", self.kv_quant_scheme])
            cmd.extend(["--kv-group-size", str(self.kv_group_size)])

        if self.max_kv_size is not None:
            cmd.extend(["--max-kv-size", str(self.max_kv_size)])

        self.process = subprocess.Popen(cmd, env=env)
        self.process.wait()

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()