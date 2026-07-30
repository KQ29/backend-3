from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    confidence: float | None = None
    detected_language: str | None = None
    provider_request_id: str | None = None
    duration_seconds: float | None = None


class SpeechToTextProvider(Protocol):
    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        mime_type: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult: ...
