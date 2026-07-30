from typing import Any

from fastapi import APIRouter, Request

router = APIRouter(tags=["system"])


@router.get("/health")
def health(request: Request) -> dict[str, Any]:
    settings = request.app.state.settings
    return {
        "status": "ok",
        "service": settings.app_name,
        "environment": settings.app_env,
        "version": request.app.version,
        "repository": settings.repository_backend,
        "llm_enabled": settings.llm_enabled,
        "llm_model": settings.llm_model or "disabled",
        "llm_mode": settings.llm_mode,
        "llm_max_calls_per_session": settings.llm_max_calls_per_session,
        "max_probes_per_anchor": settings.max_probes_per_anchor,
        "stt_provider": settings.stt_provider,
    }
