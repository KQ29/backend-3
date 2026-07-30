from __future__ import annotations

from threading import RLock
from typing import Any, Callable, TypeVar

import httpx

from app.core.config import Settings
from app.interview.engine import InterviewEngine, InvalidInterviewAction
from app.models.domain import QuestionId, SessionStatus
from app.providers.llm.mock import MockLLMProvider
from app.providers.llm.openai_compatible import (
    LLMProviderError,
    OpenAICompatibleLLMProvider,
)
from app.providers.stt.base import SpeechToTextProvider
from app.providers.stt.speechmatics import SpeechmaticsBatchProvider
from app.repositories.base import RecordNotFound
from app.repositories.memory import MemoryInterviewRepository
from app.services.interviews import InterviewService

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODELS_URL = f"{NVIDIA_BASE_URL}/models"
NVIDIA_MODEL = "meta/llama-3.1-70b-instruct"
SPEECHMATICS_BASE_URL = "https://asr.api.speechmatics.com/v2"
SPEECHMATICS_JOBS_URL = f"{SPEECHMATICS_BASE_URL}/jobs"
MAX_AUDIO_BYTES = 20 * 1024 * 1024
NVIDIA_CLASSIFICATION_VERIFICATION_ATTEMPTS = 2

SUPPORTED_AUDIO_MIME_TYPES = frozenset(
    {
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
)

_NVIDIA_VALIDATION_ANSWER = (
    "AI helps me draft weekly reports and saves time for reviewing client work."
)
_T = TypeVar("_T")


class StandaloneRuntimeError(RuntimeError):
    """A safe error that may be displayed by the standalone frontend."""


class CredentialVerificationError(StandaloneRuntimeError):
    """Raised when a provider credential cannot be verified safely."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "provider_verification_failed",
    ) -> None:
        super().__init__(message)
        self.code = code


def _normalize_required_key(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CredentialVerificationError(
            "Enter an NVIDIA API key.",
            code="missing_credentials",
        )
    normalized = value.strip()
    if (
        normalized.startswith(("NVIDIA_API_KEY=", "LLM_API_KEY="))
        or not normalized.startswith("nvapi-")
    ):
        raise CredentialVerificationError(
            "Paste only the NVIDIA API Catalog key value beginning with "
            "nvapi-; remove variable names and quotation marks.",
            code="invalid_key_format",
        )
    return normalized


def _normalize_optional_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _standalone_settings(
    nvidia_api_key: str,
    speechmatics_api_key: str | None,
) -> Settings:
    """Build settings without reading or changing process environment values."""

    return Settings(
        app_name="Otermans Kenya Interviewer",
        app_env="standalone",
        log_level="INFO",
        api_host="127.0.0.1",
        api_port=8000,
        nudge_after_hours=10,
        abandon_after_hours=24,
        inactivity_check_seconds=3600,
        max_substantive_turns=16,
        max_probes_per_anchor=2,
        llm_enabled=True,
        llm_base_url=NVIDIA_BASE_URL,
        llm_api_key=nvidia_api_key,
        llm_model=NVIDIA_MODEL,
        llm_mode="always",
        llm_timeout_seconds=30,
        llm_max_calls_per_session=16,
        llm_low_confidence_threshold=0.70,
        llm_max_output_tokens=1024,
        stt_provider="speechmatics" if speechmatics_api_key else "mock",
        stt_api_key=speechmatics_api_key,
        stt_base_url=(
            SPEECHMATICS_BASE_URL if speechmatics_api_key is not None else None
        ),
        stt_model=None,
        stt_language="auto",
        stt_timeout_seconds=60,
        stt_max_audio_bytes=MAX_AUDIO_BYTES,
        supabase_enabled=False,
        repository_backend="memory",
        supabase_url=None,
        supabase_service_role_key=None,
        database_url=None,
        admin_api_token=None,
        backend_api_token=None,
    )


def _cleared_settings() -> Settings:
    return Settings(
        app_name="Otermans Kenya Interviewer",
        app_env="standalone",
        llm_enabled=False,
        llm_base_url=None,
        llm_api_key=None,
        llm_model=None,
        stt_provider="mock",
        stt_api_key=None,
        stt_base_url=None,
        supabase_enabled=False,
        repository_backend="memory",
        supabase_url=None,
        supabase_service_role_key=None,
        database_url=None,
        admin_api_token=None,
        backend_api_token=None,
        max_probes_per_anchor=2,
    )


def _json_dict(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if isinstance(model, dict):
        return model
    raise TypeError("Standalone response could not be serialized")


class StandaloneInterviewRuntime:
    """Run one private, in-process interview without FastAPI or Supabase."""

    def __init__(
        self,
        nvidia_api_key: str,
        speechmatics_api_key: str | None = None,
        *,
        verify: bool = True,
        _transport: httpx.BaseTransport | None = None,
    ) -> None:
        nvidia_key = _normalize_required_key(nvidia_api_key)
        speechmatics_key = _normalize_optional_key(speechmatics_api_key)
        self._lock = RLock()
        self._cleared = False
        self._nvidia_preflight_verified = False
        self._credentials_verified = False
        self._speechmatics_verified = False
        self._clients: list[httpx.Client] = []
        self.settings = _standalone_settings(nvidia_key, speechmatics_key)

        self._llm_client = self._new_client(
            timeout_seconds=self.settings.llm_timeout_seconds,
            transport=_transport,
        )
        self._clients.append(self._llm_client)
        self._llm_provider = OpenAICompatibleLLMProvider(
            base_url=NVIDIA_BASE_URL,
            api_key=nvidia_key,
            model=NVIDIA_MODEL,
            timeout_seconds=self.settings.llm_timeout_seconds,
            max_output_tokens=self.settings.llm_max_output_tokens,
            client=self._llm_client,
        )

        self._stt_client: httpx.Client | None = None
        self._stt_provider: SpeechToTextProvider | None = None
        if speechmatics_key is not None:
            self._stt_client = self._new_client(
                timeout_seconds=self.settings.stt_timeout_seconds,
                transport=_transport,
            )
            self._clients.append(self._stt_client)
            self._stt_provider = SpeechmaticsBatchProvider(
                api_key=speechmatics_key,
                base_url=SPEECHMATICS_BASE_URL,
                timeout_seconds=self.settings.stt_timeout_seconds,
                default_language=self.settings.stt_language,
                client=self._stt_client,
            )

        self._replace_repository()
        if verify:
            try:
                self.verify_credentials()
            except Exception:
                self.clear_credentials()
                raise

    @staticmethod
    def _new_client(
        *,
        timeout_seconds: int,
        transport: httpx.BaseTransport | None,
    ) -> httpx.Client:
        options: dict[str, Any] = {
            "timeout": httpx.Timeout(timeout_seconds),
            "trust_env": False,
            "follow_redirects": False,
        }
        if transport is not None:
            options["transport"] = transport
        return httpx.Client(**options)

    def _replace_repository(self) -> None:
        self.repository = MemoryInterviewRepository()
        self._service = InterviewService(
            self.repository,
            InterviewEngine(self.settings, self._llm_provider),
        )

    def _ensure_configured(self) -> None:
        if self._cleared:
            raise StandaloneRuntimeError(
                "Provider credentials are not configured"
            )

    @staticmethod
    def _safe_service_call(operation: Callable[[], _T]) -> _T:
        try:
            return operation()
        except RecordNotFound:
            raise StandaloneRuntimeError(
                "Interview session was not found"
            ) from None
        except InvalidInterviewAction as exc:
            raise StandaloneRuntimeError(str(exc)) from None

    def verify_credentials(self) -> dict[str, Any]:
        """Validate keys against fixed provider endpoints without exposing them."""

        with self._lock:
            self._ensure_configured()
            self._verify_nvidia_access()
            for _ in range(NVIDIA_CLASSIFICATION_VERIFICATION_ATTEMPTS):
                try:
                    result = self._llm_provider.classify(
                        question_id=QuestionId.ANCHOR_1,
                        answer=_NVIDIA_VALIDATION_ANSWER,
                        rolling_summary="",
                        probes_remaining=0,
                    )
                except LLMProviderError as exc:
                    if self._is_structured_response_error(exc):
                        continue
                    raise self._safe_nvidia_model_error(exc) from None
                except Exception:
                    raise CredentialVerificationError(
                        "NVIDIA was reachable, but the app could not complete "
                        "its model compatibility check. Try again.",
                        code="compatibility_check_failed",
                    ) from None
                if result is None:
                    continue
                self._credentials_verified = True
                break
            else:
                raise CredentialVerificationError(
                    "NVIDIA accepted the model request, but Llama did not "
                    "return the required structured response. Click Verify "
                    "again; if this continues, create a fresh NVIDIA API "
                    "Catalog key.",
                    code="invalid_response",
                )

            self._verify_speechmatics_access()
            return {
                "nvidia": {
                    "configured": True,
                    "preflight_verified": self._nvidia_preflight_verified,
                    "verified": self._credentials_verified,
                    "model": NVIDIA_MODEL,
                },
                "speechmatics": {
                    "configured": self._stt_provider is not None,
                    "verified": self._speechmatics_verified,
                },
            }

    def _verify_nvidia_access(self) -> None:
        try:
            response = self._llm_client.get(
                NVIDIA_MODELS_URL,
                headers={
                    "Authorization": (
                        "Bearer "
                        f"{self.settings.llm_api_key.get_secret_value()}"
                    )
                },
            )
        except httpx.TimeoutException:
            raise CredentialVerificationError(
                "NVIDIA could not be reached before the verification timeout. "
                "Check the internet connection and try again.",
                code="network_unavailable",
            ) from None
        except httpx.HTTPError:
            raise CredentialVerificationError(
                "NVIDIA could not be reached. Check the internet connection "
                "and try again.",
                code="network_unavailable",
            ) from None
        if not 200 <= response.status_code < 300:
            raise self._nvidia_status_error(response.status_code)
        self._nvidia_preflight_verified = True

    def _verify_speechmatics_access(self) -> None:
        if self._stt_client is None:
            return
        try:
            response = self._stt_client.get(
                SPEECHMATICS_JOBS_URL,
                headers={
                    "Authorization": (
                        "Bearer "
                        f"{self.settings.stt_api_key.get_secret_value()}"
                    )
                },
                params={"limit": 1},
            )
            if not 200 <= response.status_code < 300:
                raise RuntimeError(
                    "Speechmatics validation returned an error"
                )
        except Exception:
            raise CredentialVerificationError(
                "Speechmatics credentials could not be verified",
                code="speechmatics_verification_failed",
            ) from None
        self._speechmatics_verified = True

    @staticmethod
    def _is_structured_response_error(exc: LLMProviderError) -> bool:
        return exc.kind == "invalid_response"

    @classmethod
    def _safe_nvidia_model_error(
        cls,
        exc: LLMProviderError,
    ) -> CredentialVerificationError:
        if exc.status_code is not None:
            return cls._nvidia_status_error(exc.status_code)
        if exc.kind == "network":
            return CredentialVerificationError(
                "NVIDIA was reachable during preflight, but the test "
                "classification could not connect. Check the connection and "
                "try again.",
                code="network_unavailable",
            )
        return CredentialVerificationError(
            "NVIDIA was reachable, but the test classification could not "
            "complete. Check NVIDIA service availability and try again.",
            code="model_request_failed",
        )

    @staticmethod
    def _nvidia_status_error(
        status_code: int,
    ) -> CredentialVerificationError:
        if status_code == 401:
            return CredentialVerificationError(
                "NVIDIA rejected this API key. Generate a fresh key from the "
                "NVIDIA API Catalog and try again.",
                code="invalid_credentials",
            )
        if status_code == 403:
            return CredentialVerificationError(
                "This NVIDIA key cannot access the selected Llama model. Sign "
                "in to NVIDIA, accept the model terms, then retry or create a "
                "new API Catalog key.",
                code="model_access_denied",
            )
        if status_code == 429:
            return CredentialVerificationError(
                "NVIDIA is rate-limiting this key. Wait briefly, check its "
                "quota, and try again.",
                code="rate_limited",
            )
        if status_code == 402:
            return CredentialVerificationError(
                "This NVIDIA account has no available quota for the configured "
                "Llama model. Check the account and key quota.",
                code="quota_unavailable",
            )
        if status_code == 404:
            return CredentialVerificationError(
                "The required NVIDIA Llama endpoint was not found. Try again "
                "later.",
                code="model_unavailable",
            )
        if status_code in {400, 422}:
            return CredentialVerificationError(
                "NVIDIA is reachable, but rejected the app's verification "
                "request. This is an app compatibility issue, not necessarily "
                "a bad key.",
                code="request_incompatible",
            )
        if status_code >= 500:
            return CredentialVerificationError(
                "NVIDIA is temporarily unavailable. Wait briefly and try "
                "verification again.",
                code="provider_unavailable",
            )
        return CredentialVerificationError(
            f"NVIDIA rejected the verification request (status {status_code}). "
            "Check the key and its API Catalog access.",
            code="request_rejected",
        )

    def health(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": "ok",
                "service": self.settings.app_name,
                "environment": "standalone",
                "repository": "memory",
                "persistence": "browser_session_only",
                "llm_enabled": not self._cleared,
                "llm_provider": "nvidia" if not self._cleared else "disabled",
                "llm_model": NVIDIA_MODEL if not self._cleared else "disabled",
                "llm_mode": "always",
                "llm_preflight_verified": self._nvidia_preflight_verified,
                "llm_verified": self._credentials_verified,
                "llm_max_calls_per_session": (
                    self.settings.llm_max_calls_per_session
                ),
                "max_probes_per_anchor": (
                    self.settings.max_probes_per_anchor
                ),
                "stt_provider": (
                    "speechmatics"
                    if self._stt_provider is not None
                    else "disabled"
                ),
                "stt_verified": self._speechmatics_verified,
            }

    def start(self) -> dict[str, Any]:
        with self._lock:
            self._ensure_configured()
            # A standalone browser session retains only its current interview.
            self._replace_repository()
            return _json_dict(self._service.start_interview())

    def consent(self, session_id: str, choice: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_configured()
            response = self._safe_service_call(
                lambda: self._service.submit_consent(session_id, choice)
            )
            return _json_dict(response)

    def text(self, session_id: str, text: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_configured()
            response = self._safe_service_call(
                lambda: self._service.submit_text(session_id, text)
            )
            return _json_dict(response)

    def voice(
        self,
        session_id: str,
        *,
        audio: bytes,
        filename: str,
        mime_type: str,
        language_hint: str | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            self._ensure_configured()
            state = self._safe_service_call(
                lambda: self._service.get_state(session_id)
            )
            if not state.consent_given or state.status not in {
                SessionStatus.COLLECTING_DEMOGRAPHICS,
                SessionStatus.IN_PROGRESS,
            }:
                raise StandaloneRuntimeError(
                    "Voice transcription requires an active consented interview"
                )
            if self._stt_provider is None:
                raise StandaloneRuntimeError(
                    "Voice transcription requires Speechmatics credentials"
                )

            normalized_mime = (mime_type or "").split(";", 1)[0].strip().lower()
            if normalized_mime not in SUPPORTED_AUDIO_MIME_TYPES:
                raise StandaloneRuntimeError(
                    f"Unsupported audio type: {normalized_mime or 'unknown'}"
                )
            if not isinstance(audio, bytes):
                raise StandaloneRuntimeError("Audio must be provided as bytes")
            if not audio:
                raise StandaloneRuntimeError("Audio file is empty")
            if len(audio) > MAX_AUDIO_BYTES:
                raise StandaloneRuntimeError(
                    "Audio file exceeds the 20 MB size limit"
                )

            try:
                transcription = self._stt_provider.transcribe(
                    audio=audio,
                    filename=filename or "voice_note.ogg",
                    mime_type=normalized_mime,
                    language_hint=language_hint,
                )
            except Exception:
                raise StandaloneRuntimeError("Voice transcription failed") from None

            response = self._safe_service_call(
                lambda: self._service.submit_voice(
                    session_id,
                    transcription.text,
                    transcription.confidence,
                )
            )
            response.transcript = transcription.text
            response.transcription_confidence = transcription.confidence
            return _json_dict(response)

    def state(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_configured()
            response = self._safe_service_call(
                lambda: self._service.get_state(session_id)
            )
            return _json_dict(response)

    def stop(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_configured()
            response = self._safe_service_call(
                lambda: self._service.stop(session_id)
            )
            return _json_dict(response)

    def export(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._ensure_configured()
            response = self._safe_service_call(
                lambda: self._service.export_record(session_id)
            )
            return _json_dict(response)

    def reset(self) -> dict[str, Any]:
        """Delete interview records while retaining verified providers."""

        with self._lock:
            self._ensure_configured()
            self._replace_repository()
            return self.health()

    def clear_credentials(self) -> dict[str, Any]:
        """Delete interviews and release every provider credential reference."""

        with self._lock:
            for client in self._clients:
                try:
                    client.close()
                except Exception:
                    pass
            self._clients.clear()
            self._llm_client = None
            self._stt_client = None
            self._llm_provider = MockLLMProvider()
            self._stt_provider = None
            self._nvidia_preflight_verified = False
            self._credentials_verified = False
            self._speechmatics_verified = False
            self._cleared = True
            self.settings = _cleared_settings()
            self._replace_repository()
            return self.health()


def create_standalone_runtime(
    nvidia_api_key: str,
    speechmatics_api_key: str | None = None,
    *,
    verify: bool = True,
) -> StandaloneInterviewRuntime:
    """Create an isolated runtime from credentials entered in the frontend."""

    return StandaloneInterviewRuntime(
        nvidia_api_key,
        speechmatics_api_key,
        verify=verify,
    )


def verify_provider_credentials(
    nvidia_api_key: str,
    speechmatics_api_key: str | None = None,
) -> dict[str, Any]:
    """Validate provider credentials without retaining them or interview state."""

    runtime = StandaloneInterviewRuntime(
        nvidia_api_key,
        speechmatics_api_key,
        verify=False,
    )
    try:
        return runtime.verify_credentials()
    finally:
        runtime.clear_credentials()
