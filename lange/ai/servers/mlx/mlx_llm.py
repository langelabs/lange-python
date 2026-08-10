import os
import subprocess
from ..__base import InferenceServer


class MlxLLMServer(InferenceServer):
    def run(self) -> None:
        self.download()

        if self.model.registration is None:
            raise ValueError("MLX LLM models require registration metadata.")

        env = os.environ.copy()

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
            str(8),
            "--log-level",
            "INFO",
        ]

        if self.model.enable_thinking:
            cmd.append("--enable-thinking")

        if self.model.kv_cache_config:
            if self.model.kv_cache_config.kv_bits is not None:
                cmd.extend(["--kv-bits", str(self.model.kv_cache_config.kv_bits)])
                cmd.extend(
                    ["--kv-quant-scheme", self.model.kv_cache_config.kv_quant_scheme]
                )
                cmd.extend(
                    ["--kv-group-size", str(self.model.kv_cache_config.kv_group_size)]
                )

            if self.model.kv_cache_config.kv_max_size is not None:
                cmd.extend(
                    ["--max-kv-size", str(self.model.kv_cache_config.kv_max_size)]
                )

        _process = subprocess.Popen(cmd, env=env)
        self.process = _process
        _process.wait()

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
