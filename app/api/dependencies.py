from fastapi import Request

from app.core.config import Settings
from app.providers.stt.base import SpeechToTextProvider
from app.services.interviews import InterviewService


def get_interview_service(request: Request) -> InterviewService:
    return request.app.state.interview_service


def get_speech_to_text_provider(request: Request) -> SpeechToTextProvider:
    return request.app.state.speech_to_text_provider


def get_request_settings(request: Request) -> Settings:
    return request.app.state.settings
