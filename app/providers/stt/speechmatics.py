from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from app.providers.stt.base import TranscriptionResult


class SpeechToTextError(RuntimeError):
    pass


class SpeechmaticsBatchProvider:
    """Speechmatics Batch API adapter for recorded interview answers."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        timeout_seconds: int,
        default_language: str = "auto",
        client: httpx.Client | None = None,
    ) -> None:
        normalized = base_url.rstrip("/")
        self.jobs_url = (
            normalized if normalized.endswith("/jobs") else f"{normalized}/jobs"
        )
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.default_language = default_language
        self._client = client

    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        mime_type: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        if not audio:
            raise SpeechToTextError("Audio cannot be empty")

        language = language_hint or self.default_language
        config = {
            "type": "transcription",
            "transcription_config": {
                "operating_point": "enhanced",
                "language": language,
            },
        }
        submit = self._request(
            "POST",
            f"{self.jobs_url}/",
            files={"data_file": (filename, audio, mime_type)},
            data={"config": json.dumps(config)},
        )
        try:
            job_id = submit.json()["id"]
        except (ValueError, KeyError, TypeError) as exc:
            raise SpeechToTextError(
                "Speechmatics did not return a transcription job ID"
            ) from exc

        deadline = time.monotonic() + self.timeout_seconds
        job_details: dict[str, Any] = {}
        while time.monotonic() < deadline:
            response = self._request("GET", f"{self.jobs_url}/{job_id}")
            try:
                job_details = response.json().get("job", {})
                status = job_details.get("status")
            except (ValueError, AttributeError) as exc:
                raise SpeechToTextError(
                    "Speechmatics returned an invalid job status"
                ) from exc

            if status == "done":
                break
            if status in {"rejected", "deleted"}:
                raise SpeechToTextError(
                    f"Speechmatics rejected the transcription job ({status})"
                )
            time.sleep(1)
        else:
            raise SpeechToTextError(
                "Speechmatics transcription exceeded the configured timeout"
            )

        transcript_response = self._request(
            "GET",
            f"{self.jobs_url}/{job_id}/transcript",
            params={"format": "json-v2"},
        )
        try:
            transcript = transcript_response.json()
        except ValueError as exc:
            raise SpeechToTextError(
                "Speechmatics transcript response was not JSON"
            ) from exc

        text, confidence = self._extract_transcript(transcript)
        if not text:
            raise SpeechToTextError("Speechmatics returned an empty transcript")

        duration = job_details.get("duration")
        return TranscriptionResult(
            text=text,
            confidence=confidence,
            detected_language=language,
            provider_request_id=str(job_id),
            duration_seconds=float(duration) if duration is not None else None,
        )

    def probe(self) -> dict[str, Any]:
        return {
            "provider": "speechmatics",
            "jobs_url": self.jobs_url,
            "enabled": True,
        }

    def _request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            if self._client is not None:
                response = self._client.request(
                    method,
                    url,
                    headers=headers,
                    **kwargs,
                )
            else:
                response = httpx.request(
                    method,
                    url,
                    headers=headers,
                    timeout=self.timeout_seconds,
                    **kwargs,
                )
        except httpx.HTTPError as exc:
            raise SpeechToTextError("Speechmatics request failed") from exc

        if response.is_error:
            raise SpeechToTextError(
                f"Speechmatics request failed with status {response.status_code}"
            )
        return response

    @staticmethod
    def _extract_transcript(
        transcript: dict[str, Any],
    ) -> tuple[str, float | None]:
        tokens: list[str] = []
        confidences: list[float] = []
        for result in transcript.get("results", []):
            alternatives = result.get("alternatives") or []
            if not alternatives:
                continue
            alternative = alternatives[0]
            content = alternative.get("content")
            if content:
                tokens.append(str(content))
            confidence = alternative.get("confidence")
            if isinstance(confidence, (int, float)):
                confidences.append(float(confidence))

        text = " ".join(tokens).strip()
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        average = round(sum(confidences) / len(confidences), 2) if confidences else None
        return text, average
