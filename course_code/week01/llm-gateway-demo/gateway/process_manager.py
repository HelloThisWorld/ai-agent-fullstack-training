from __future__ import annotations

import subprocess
from pathlib import Path

from gateway.config import Settings


class LlamaProcessManager:
    """Optionally manage a local llama-server process.

    Disabled by default. The process receives only model/runtime arguments; no
    provider credential is ever passed as an argument or environment value.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.process: subprocess.Popen[bytes] | None = None

    async def start(self) -> None:
        if not self.settings.manage_local_process:
            return
        if self.process and self.process.poll() is None:
            return

        binary = Path(self.settings.llama_binary)
        model = Path(self.settings.llama_model)
        if not binary.is_file():
            raise FileNotFoundError(f"llama-server binary was not found: {binary}")
        if not model.is_file():
            raise FileNotFoundError(f"GGUF model was not found: {model}")

        command = [
            str(binary),
            "-m",
            str(model),
            "--host",
            self.settings.llama_host,
            "--port",
            str(self.settings.llama_port),
            "--ctx-size",
            str(self.settings.llama_context_size),
            "--parallel",
            str(self.settings.llama_parallel),
            "--batch-size",
            str(self.settings.llama_batch_size),
        ]
        if self.settings.llama_gpu_layers is not None:
            command.extend(["--n-gpu-layers", str(self.settings.llama_gpu_layers)])
        if self.settings.llama_threads is not None:
            command.extend(["--threads", str(self.settings.llama_threads)])

        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

    async def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=5)
        finally:
            self.process = None
