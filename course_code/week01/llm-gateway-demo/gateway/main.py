from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from gateway.api.routes import gateway_error_handler, router
from gateway.config import Settings
from gateway.errors import GatewayError
from gateway.runtime import GatewayRuntime


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.runtime = GatewayRuntime(app_settings)
        await app.state.runtime.start()
        yield
        await app.state.runtime.close()

    app = FastAPI(
        title="LLM Gateway Demo",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.add_exception_handler(GatewayError, gateway_error_handler)
    app.include_router(router)
    web_root = Path(__file__).resolve().parent.parent / "web"
    if web_root.exists():
        app.mount("/", StaticFiles(directory=web_root, html=True), name="web")
    return app


app = create_app()


def run() -> None:
    import uvicorn

    settings = Settings.from_environment()
    uvicorn.run("gateway.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    run()
