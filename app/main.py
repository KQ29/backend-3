from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes.health import router as health_router
from app.api.routes.interviews import internal_router
from app.api.routes.interviews import router as interviews_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger
from app.interview.engine import InterviewEngine, InvalidInterviewAction
from app.providers.llm.base import LLMProvider
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.openai_compatible import (
    LLMProviderError,
    OpenAICompatibleLLMProvider,
)
from app.providers.stt.mock import MockSpeechToTextProvider
from app.providers.stt.speechmatics import (
    SpeechmaticsBatchProvider,
    SpeechToTextError,
)
from app.providers.transport.logging import LoggingTransport
from app.repositories.base import InterviewRepository, RecordNotFound
from app.repositories.memory import MemoryInterviewRepository
from app.repositories.supabase import (
    SupabaseInterviewRepository,
    SupabaseRepositoryError,
    SupabaseRestClient,
)
from app.services.interviews import InterviewService
from app.workers.inactivity import run_inactivity_pass

logger = get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    repository: InterviewRepository | None = None,
    llm_provider: LLMProvider | None = None,
) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings.log_level)
    if repository is not None:
        resolved_repository = repository
    elif resolved_settings.repository_backend == "supabase":
        resolved_repository = SupabaseInterviewRepository(
            SupabaseRestClient(
                project_url=resolved_settings.supabase_url or "",
                service_role_key=(
                    resolved_settings.supabase_service_role_key.get_secret_value()
                    if resolved_settings.supabase_service_role_key
                    else ""
                ),
            )
        )
    else:
        resolved_repository = MemoryInterviewRepository()

    if llm_provider is not None:
        resolved_llm_provider = llm_provider
    elif resolved_settings.llm_enabled:
        resolved_llm_provider = OpenAICompatibleLLMProvider(
            base_url=resolved_settings.llm_base_url or "",
            api_key=(
                resolved_settings.llm_api_key.get_secret_value()
                if resolved_settings.llm_api_key
                else ""
            ),
            model=resolved_settings.llm_model or "",
            timeout_seconds=resolved_settings.llm_timeout_seconds,
            max_output_tokens=resolved_settings.llm_max_output_tokens,
        )
    else:
        resolved_llm_provider = MockLLMProvider()

    if resolved_settings.stt_provider == "speechmatics":
        speech_to_text_provider = SpeechmaticsBatchProvider(
            api_key=(
                resolved_settings.stt_api_key.get_secret_value()
                if resolved_settings.stt_api_key
                else ""
            ),
            base_url=resolved_settings.stt_base_url or "",
            timeout_seconds=resolved_settings.stt_timeout_seconds,
            default_language=resolved_settings.stt_language,
        )
    else:
        speech_to_text_provider = MockSpeechToTextProvider()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        stop_event = asyncio.Event()
        transport = LoggingTransport()

        async def inactivity_loop() -> None:
            while not stop_event.is_set():
                try:
                    await asyncio.to_thread(
                        run_inactivity_pass,
                        application.state.repository,
                        transport,
                        resolved_settings,
                    )
                except Exception as exc:  # noqa: BLE001 - keep worker alive
                    logger.error(
                        "Inactivity pass failed; it will retry on schedule",
                        extra={
                            "context": {
                                "error_type": type(exc).__name__,
                                "repository": (resolved_settings.repository_backend),
                            }
                        },
                    )
                try:
                    await asyncio.wait_for(
                        stop_event.wait(),
                        timeout=resolved_settings.inactivity_check_seconds,
                    )
                except TimeoutError:
                    continue

        worker = asyncio.create_task(
            inactivity_loop(),
            name="interview-inactivity-worker",
        )
        try:
            yield
        finally:
            stop_event.set()
            await worker

    app = FastAPI(
        title=resolved_settings.app_name,
        version="0.3.1",
        description="Channel-independent adaptive research interview backend",
        lifespan=lifespan,
    )
    app.state.settings = resolved_settings
    app.state.repository = resolved_repository
    app.state.llm_provider = resolved_llm_provider
    app.state.speech_to_text_provider = speech_to_text_provider
    app.state.interview_service = InterviewService(
        app.state.repository,
        InterviewEngine(resolved_settings, resolved_llm_provider),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://127.0.0.1:{os.getenv('STREAMLIT_PORT', '8501')}",
            f"http://localhost:{os.getenv('STREAMLIT_PORT', '8501')}",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Admin-Token"],
    )
    app.include_router(health_router)
    app.include_router(interviews_router)
    app.include_router(internal_router)

    @app.exception_handler(RecordNotFound)
    async def record_not_found_handler(
        request: Request,
        exc: RecordNotFound,
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=404,
            content={"detail": f"Interview session not found: {exc.args[0]}"},
        )

    @app.exception_handler(InvalidInterviewAction)
    async def invalid_action_handler(
        request: Request,
        exc: InvalidInterviewAction,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.exception_handler(SpeechToTextError)
    async def speech_to_text_error_handler(
        request: Request,
        exc: SpeechToTextError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(LLMProviderError)
    async def llm_error_handler(
        request: Request,
        exc: LLMProviderError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(SupabaseRepositoryError)
    async def supabase_error_handler(
        request: Request,
        exc: SupabaseRepositoryError,
    ) -> JSONResponse:
        del request
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    return app


app = create_app()
