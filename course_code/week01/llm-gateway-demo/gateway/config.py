from __future__ import annotations

import os
from dataclasses import dataclass, field

from gateway.storage import default_data_path


@dataclass(frozen=True, slots=True)
class Settings:
    """Non-sensitive runtime settings.

    This class intentionally does not load dotenv files or secret files. Provider
    credentials are resolved by SecretStore implementations at request time.
    """

    host: str = "127.0.0.1"
    port: int = 9000
    local_base_url: str = "http://127.0.0.1:8080"
    local_model: str = "local/qwen3.8-27b-q5ks"
    request_timeout_seconds: float = 120.0
    manage_local_process: bool = False
    llama_binary: str = r"D:\titan-llama\llama-server.exe"
    llama_model: str = r"D:\titan-models\qwen\Qwen3.8-27B-Q5_K_S.gguf"
    llama_host: str = "127.0.0.1"
    llama_port: int = 8080
    llama_context_size: int = 4096
    llama_parallel: int = 1
    llama_gpu_layers: int | None = None
    llama_threads: int | None = None
    llama_batch_size: int = 256
    data_path: str = field(default_factory=default_data_path)
    rate_limit_requests: int = 30
    rate_limit_window_seconds: float = 60.0
    max_retries: int = 3
    retry_base_delay_seconds: float = 0.25
    retry_max_delay_seconds: float = 4.0
    openai_base_url: str = "https://api.openai.com/v1"
    anthropic_base_url: str = "https://api.anthropic.com"
    deepseek_base_url: str = "https://api.deepseek.com"
    openai_secret_ref: str = "windows-credential://llm-gateway/openai"
    anthropic_secret_ref: str = "windows-credential://llm-gateway/anthropic"
    deepseek_secret_ref: str = "windows-credential://llm-gateway/deepseek"

    @classmethod
    def from_environment(cls) -> "Settings":
        """Read only non-secret settings from the process environment.

        In particular, this method never reads `.env` or any file under the workspace.
        """

        defaults = cls()
        return cls(
            host=os.getenv("LLM_GATEWAY_HOST", defaults.host),
            port=int(os.getenv("LLM_GATEWAY_PORT", str(defaults.port))),
            local_base_url=os.getenv("LLAMA_CPP_BASE_URL", defaults.local_base_url),
            local_model=os.getenv("LLM_GATEWAY_LOCAL_MODEL", defaults.local_model),
            request_timeout_seconds=float(
                os.getenv("LLM_GATEWAY_REQUEST_TIMEOUT", str(defaults.request_timeout_seconds))
            ),
            manage_local_process=os.getenv("LLM_GATEWAY_MANAGE_LLAMA", "false").lower()
            in {"1", "true", "yes", "on"},
            llama_binary=os.getenv("LLAMA_CPP_BINARY", defaults.llama_binary),
            llama_model=os.getenv("LLAMA_CPP_MODEL", defaults.llama_model),
            llama_host=os.getenv("LLAMA_CPP_HOST", defaults.llama_host),
            llama_port=int(os.getenv("LLAMA_CPP_PORT", str(defaults.llama_port))),
            llama_context_size=int(
                os.getenv("LLAMA_CPP_CONTEXT_SIZE", str(defaults.llama_context_size))
            ),
            llama_parallel=int(os.getenv("LLAMA_CPP_PARALLEL", str(defaults.llama_parallel))),
            llama_gpu_layers=(
                int(os.environ["LLAMA_CPP_GPU_LAYERS"])
                if os.getenv("LLAMA_CPP_GPU_LAYERS")
                else defaults.llama_gpu_layers
            ),
            llama_threads=(
                int(os.environ["LLAMA_CPP_THREADS"])
                if os.getenv("LLAMA_CPP_THREADS")
                else defaults.llama_threads
            ),
            llama_batch_size=int(
                os.getenv("LLAMA_CPP_BATCH_SIZE", str(defaults.llama_batch_size))
            ),
            data_path=os.getenv("LLM_GATEWAY_DATA_PATH", default_data_path()),
            rate_limit_requests=int(
                os.getenv("LLM_GATEWAY_RATE_LIMIT_REQUESTS", str(defaults.rate_limit_requests))
            ),
            rate_limit_window_seconds=float(
                os.getenv(
                    "LLM_GATEWAY_RATE_LIMIT_WINDOW_SECONDS",
                    str(defaults.rate_limit_window_seconds),
                )
            ),
            max_retries=int(os.getenv("LLM_GATEWAY_MAX_RETRIES", str(defaults.max_retries))),
            retry_base_delay_seconds=float(
                os.getenv(
                    "LLM_GATEWAY_RETRY_BASE_DELAY_SECONDS",
                    str(defaults.retry_base_delay_seconds),
                )
            ),
            retry_max_delay_seconds=float(
                os.getenv(
                    "LLM_GATEWAY_RETRY_MAX_DELAY_SECONDS",
                    str(defaults.retry_max_delay_seconds),
                )
            ),
            openai_base_url=os.getenv("LLM_GATEWAY_OPENAI_BASE_URL", defaults.openai_base_url),
            anthropic_base_url=os.getenv(
                "LLM_GATEWAY_ANTHROPIC_BASE_URL", defaults.anthropic_base_url
            ),
            deepseek_base_url=os.getenv("LLM_GATEWAY_DEEPSEEK_BASE_URL", defaults.deepseek_base_url),
            openai_secret_ref=os.getenv(
                "LLM_GATEWAY_OPENAI_SECRET_REF", defaults.openai_secret_ref
            ),
            anthropic_secret_ref=os.getenv(
                "LLM_GATEWAY_ANTHROPIC_SECRET_REF", defaults.anthropic_secret_ref
            ),
            deepseek_secret_ref=os.getenv(
                "LLM_GATEWAY_DEEPSEEK_SECRET_REF", defaults.deepseek_secret_ref
            ),
        )
