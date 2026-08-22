class GatewayError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 500):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


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
        super().__init__(
            code="provider_request_failed",
            message=f"{provider} request failed: {message}",
            status_code=status_code,
        )
