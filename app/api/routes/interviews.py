from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.dependencies import (
    get_interview_service,
    get_request_settings,
    get_speech_to_text_provider,
)
from app.core.config import Settings
from app.interview.engine import InvalidInterviewAction
from app.models.api import (
    BatchAuditResponse,
    ConsentRequest,
    InterviewResponse,
    InterviewStateResponse,
    StartInterviewRequest,
    TextAnswerRequest,
)
from app.models.domain import SessionStatus
from app.providers.stt.base import SpeechToTextProvider
from app.services.interviews import InterviewService

router = APIRouter(prefix="/api/v1/interviews", tags=["interviews"])
internal_router = APIRouter(
    prefix="/api/v1/internal/interviews",
    tags=["internal"],
)
Service = Annotated[InterviewService, Depends(get_interview_service)]
STTProvider = Annotated[
    SpeechToTextProvider,
    Depends(get_speech_to_text_provider),
]
RequestSettings = Annotated[Settings, Depends(get_request_settings)]
SUPPORTED_AUDIO_MIME_TYPES = {
    "audio/aac",
    "audio/flac",
    "audio/m4a",
    "audio/mp4",
    "audio/mpeg",
    "audio/ogg",
    "audio/wav",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
    "application/ogg",
    "video/mp4",
}


@router.post("/start", response_model=InterviewResponse, status_code=201)
def start_interview(
    payload: StartInterviewRequest,
    service: Service,
) -> InterviewResponse:
    del payload
    return service.start_interview()


@router.post("/{session_id}/consent", response_model=InterviewResponse)
def submit_consent(
    session_id: str,
    payload: ConsentRequest,
    service: Service,
) -> InterviewResponse:
    return service.submit_consent(session_id, payload.choice)


@router.post("/{session_id}/text", response_model=InterviewResponse)
def submit_text(
    session_id: str,
    payload: TextAnswerRequest,
    service: Service,
) -> InterviewResponse:
    return service.submit_text(session_id, payload.text)


@router.post("/{session_id}/voice", response_model=InterviewResponse)
async def submit_voice(
    session_id: str,
    service: Service,
    stt_provider: STTProvider,
    settings: RequestSettings,
    audio: Annotated[UploadFile, File()],
    language_hint: Annotated[str | None, Form()] = None,
) -> InterviewResponse:
    state = service.get_state(session_id)
    if not state.consent_given or state.status not in {
        SessionStatus.COLLECTING_DEMOGRAPHICS,
        SessionStatus.IN_PROGRESS,
    }:
        raise InvalidInterviewAction(
            "Voice transcription requires an active consented interview"
        )

    mime_type = (audio.content_type or "").lower()
    if mime_type not in SUPPORTED_AUDIO_MIME_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported audio type: {mime_type or 'unknown'}",
        )
    audio_bytes = await audio.read(settings.stt_max_audio_bytes + 1)
    await audio.close()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Audio file is empty")
    if len(audio_bytes) > settings.stt_max_audio_bytes:
        raise HTTPException(
            status_code=413,
            detail="Audio file exceeds the configured size limit",
        )

    transcription = stt_provider.transcribe(
        audio=audio_bytes,
        filename=audio.filename or "voice_note.ogg",
        mime_type=mime_type,
        language_hint=language_hint,
    )
    response = service.submit_voice(
        session_id,
        transcription.text,
        transcription.confidence,
    )
    response.transcript = transcription.text
    response.transcription_confidence = transcription.confidence
    return response


@router.get("/{session_id}/state", response_model=InterviewStateResponse)
def get_state(
    session_id: str,
    service: Service,
) -> InterviewStateResponse:
    return service.get_state(session_id)


@router.post("/{session_id}/stop", response_model=InterviewResponse)
def stop_interview(
    session_id: str,
    service: Service,
) -> InterviewResponse:
    return service.stop(session_id)


@router.get("/{session_id}/export")
def export_interview(
    session_id: str,
    service: Service,
) -> dict[str, Any]:
    return service.export_record(session_id)


@internal_router.post(
    "/{session_id}/batch-audit",
    response_model=BatchAuditResponse,
)
def batch_audit_status(
    session_id: str,
    service: Service,
) -> BatchAuditResponse:
    return service.batch_audit_status(session_id)
