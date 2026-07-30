from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Header, HTTPException, Request, status

from app.core.config import Settings
from app.providers.stt.base import SpeechToTextProvider
from app.services.interviews import InterviewService


def get_interview_service(request: Request) -> InterviewService:
    return request.app.state.interview_service


def get_speech_to_text_provider(request: Request) -> SpeechToTextProvider:
    return request.app.state.speech_to_text_provider


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings


def require_backend_token(
    request: Request,
    provided_token: Annotated[
        str | None,
        Header(alias="X-Backend-Token"),
    ] = None,
) -> None:
    configured_token = request.app.state.settings.backend_api_token
    if configured_token is None:
        return

    expected = configured_token.get_secret_value().encode("utf-8")
    provided = (provided_token or "").encode("utf-8")
    if not secrets.compare_digest(provided, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing backend token",
        )
