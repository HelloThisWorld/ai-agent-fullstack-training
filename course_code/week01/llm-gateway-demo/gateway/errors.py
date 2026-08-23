class GatewayError(Exception):
    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 500,
        *,
        retryable: bool = False,
        details: dict[str, object] | None = None,
    ):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.retryable = retryable
        self.details = details or {}


class ProviderNotConfiguredError(GatewayError):
    def __init__(self, provider: str):
        super().__init__(
            code="provider_not_configured",
            message=f"Provider '{provider}' has no credential configured outside the workspace.",
            status_code=503,
        )


class UnsupportedModelError(GatewayError):
    def __init__(self, model: str):
        super().__init__(
            code="model_not_found",
            message=f"No adapter is registered for model '{model}'.",
            status_code=404,
        )


class ProviderRequestError(GatewayError):
    def __init__(self, provider: str, message: str, status_code: int = 502):
        if status_code == 429:
            code = "provider_rate_limited"
            retryable = True
        elif status_code in {408, 504}:
            code = "provider_timeout"
            retryable = True
        elif status_code >= 500:
            code = "provider_unavailable"
            retryable = True
        else:
            code = "provider_bad_request"
            retryable = False
        super().__init__(
            code=code,
            message=f"{provider} request failed: {message}",
            status_code=status_code,
            retryable=retryable,
        )


class RateLimitExceededError(GatewayError):
    def __init__(self, model: str, retry_after_seconds: float):
        super().__init__(
            code="rate_limit_exceeded",
            message=f"Model '{model}' exceeded its request rate limit.",
            status_code=492,
            details={"model": model, "retry_after_seconds": round(retry_after_seconds, 3)},
        )


class StructuredOutputError(GatewayError):
    def __init__(self, message: str):
        super().__init__("invalid_json_output", message, status_code=502)


class TemplateError(GatewayError):
    def __init__(self, code: str, message: str, status_code: int = 422):
        super().__init__(code, message, status_code=status_code)
