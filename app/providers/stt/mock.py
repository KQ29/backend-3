from __future__ import annotations

from app.providers.stt.base import TranscriptionResult


class MockSpeechToTextProvider:
    def __init__(self, transcript: str = "Mock transcription") -> None:
        self.transcript = transcript

    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        mime_type: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        if not audio:
            raise ValueError("Audio cannot be empty")
        return TranscriptionResult(
            text=self.transcript,
            confidence=0.90,
            detected_language=language_hint or "en",
            provider_request_id="mock-request",
        )
