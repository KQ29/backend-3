from __future__ import annotations

import json
import logging
import os
from typing import Any

import httpx
import pytest

from app.providers.stt.base import TranscriptionResult
from app.core.logging import JsonFormatter
from app.providers.llm import openai_compatible
from app.services import standalone
from app.services.standalone import (
    MAX_AUDIO_BYTES,
    NVIDIA_BASE_URL,
    NVIDIA_MODEL,
    NVIDIA_MODELS_URL,
    SPEECHMATICS_JOBS_URL,
    CredentialVerificationError,
    StandaloneInterviewRuntime,
    StandaloneRuntimeError,
    create_standalone_runtime,
)


def _valid_nvidia_validation_response() -> httpx.Response:
    classification = {
        "polarity": "positive",
        "mixed_evidence": False,
        "confidence": 0.93,
        "vague": False,
        "concrete": True,
        "barrier_named": False,
        "benefit_named": True,
        "affirmative": False,
        "on_topic": True,
        "apology": False,
        "correction_or_error": False,
        "skip_requested": False,
        "tough_or_complex": False,
        "positive_milestone": False,
        "economic_outcome": "improved_current_role_only",
        "bottleneck_types": [],
        "benefit_mechanism": "efficiency_in_current_role",
        "needs_probe": False,
        "probe_type": "none",
        "probe_reason": None,
        "suggested_probe": None,
        "reflection": (
            "AI helps draft weekly reports and saves review time."
        ),
        "grounding_quote": "draft weekly reports",
    }
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(classification)},
                }
            ]
        },
    )


def _valid_nvidia_models_response() -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"id": NVIDIA_MODEL}]},
    )


def _assert_json_safe_and_secret_free(
    value: dict[str, Any],
    *secrets: str,
) -> None:
    serialized = json.dumps(value)
    for secret in secrets:
        assert secret not in serialized


def test_credentials_are_verified_at_fixed_endpoints_without_env_trust(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nvidia_key = "nvapi-runtime-verification-secret"
    speechmatics_key = "speechmatics-runtime-verification-secret"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url == NVIDIA_MODELS_URL:
            return _valid_nvidia_models_response()
        if request.url == f"{NVIDIA_BASE_URL}/chat/completions":
            return _valid_nvidia_validation_response()
        if request.url.copy_with(query=None) == SPEECHMATICS_JOBS_URL:
            return httpx.Response(200, json={"jobs": []})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    original_client = httpx.Client
    client_options: list[dict[str, Any]] = []

    def recording_client(*args: Any, **kwargs: Any) -> httpx.Client:
        client_options.append(dict(kwargs))
        return original_client(*args, **kwargs)

    monkeypatch.setattr(standalone.httpx, "Client", recording_client)
    runtime = StandaloneInterviewRuntime(
        nvidia_key,
        speechmatics_key,
        _transport=transport,
    )

    assert len(requests) == 3
    assert requests[0].url == NVIDIA_MODELS_URL
    assert requests[1].url == f"{NVIDIA_BASE_URL}/chat/completions"
    assert requests[2].url.path == "/v2/jobs"
    request_body = json.loads(requests[1].content)
    assert request_body["model"] == NVIDIA_MODEL
    assert requests[0].headers["authorization"] == f"Bearer {nvidia_key}"
    assert requests[1].headers["authorization"] == f"Bearer {nvidia_key}"
    assert requests[2].headers["authorization"] == (
        f"Bearer {speechmatics_key}"
    )
    assert all(options["trust_env"] is False for options in client_options)
    assert all(options["follow_redirects"] is False for options in client_options)

    health = runtime.health()
    assert health["llm_preflight_verified"] is True
    assert health["llm_verified"] is True
    assert health["stt_verified"] is True
    assert health["repository"] == "memory"
    _assert_json_safe_and_secret_free(
        health,
        nvidia_key,
        speechmatics_key,
    )
    runtime.clear_credentials()


def test_rejected_nvidia_key_is_actionable_and_secret_free() -> None:
    nvidia_key = "nvapi-do-not-leak-this-value"
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            401,
            text=f"invalid credential {nvidia_key}",
        )

    with pytest.raises(CredentialVerificationError) as raised:
        StandaloneInterviewRuntime(
            nvidia_key,
            _transport=httpx.MockTransport(handler),
        )

    assert raised.value.code == "invalid_credentials"
    assert str(raised.value) == (
        "NVIDIA rejected this API key. Generate a fresh key from the NVIDIA "
        "API Catalog and try again."
    )
    assert len(requests) == 1
    assert nvidia_key not in str(raised.value)


def test_nvidia_model_access_failure_is_distinct_and_secret_free() -> None:
    nvidia_key = "nvapi-no-model-access-test"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            403,
            text=f"terms not accepted for {nvidia_key}",
        )

    with pytest.raises(CredentialVerificationError) as raised:
        StandaloneInterviewRuntime(
            nvidia_key,
            _transport=httpx.MockTransport(handler),
        )

    assert raised.value.code == "model_access_denied"
    assert "accept the model terms" in str(raised.value)
    assert nvidia_key not in str(raised.value)


@pytest.mark.parametrize(
    "value",
    [
        "NVIDIA_API_KEY=nvapi-example",
        "LLM_API_KEY=nvapi-example",
        '"nvapi-example"',
        "an-ngc-or-other-key",
    ],
)
def test_nvidia_key_format_error_explains_what_to_paste(value: str) -> None:
    with pytest.raises(CredentialVerificationError) as raised:
        StandaloneInterviewRuntime(value, verify=False)

    assert raised.value.code == "invalid_key_format"
    assert "beginning with nvapi-" in str(raised.value)
    assert value not in str(raised.value)


def test_speechmatics_failure_remains_generic_and_secret_free() -> None:
    nvidia_key = "nvapi-valid-test-value"
    speechmatics_key = "speechmatics-do-not-leak-this-value"

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url == NVIDIA_MODELS_URL:
            return _valid_nvidia_models_response()
        if request.url == f"{NVIDIA_BASE_URL}/chat/completions":
            return _valid_nvidia_validation_response()
        return httpx.Response(
            403,
            text=f"invalid credential {speechmatics_key}",
        )

    with pytest.raises(CredentialVerificationError) as raised:
        StandaloneInterviewRuntime(
            nvidia_key,
            speechmatics_key,
            _transport=httpx.MockTransport(handler),
        )

    assert raised.value.code == "speechmatics_verification_failed"
    assert str(raised.value) == (
        "Speechmatics credentials could not be verified"
    )
    assert nvidia_key not in str(raised.value)
    assert speechmatics_key not in str(raised.value)


def test_nvidia_structured_verification_retries_then_succeeds() -> None:
    nvidia_key = "nvapi-structured-retry-test"
    chat_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_attempts
        if request.url == NVIDIA_MODELS_URL:
            return _valid_nvidia_models_response()
        chat_attempts += 1
        if chat_attempts == 1:
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "not-json"},
                        }
                    ]
                },
            )
        return _valid_nvidia_validation_response()

    runtime = StandaloneInterviewRuntime(
        nvidia_key,
        _transport=httpx.MockTransport(handler),
    )

    assert chat_attempts == 2
    assert runtime.health()["llm_verified"] is True
    _assert_json_safe_and_secret_free(runtime.health(), nvidia_key)
    runtime.clear_credentials()


def test_valid_nvidia_key_with_repeated_invalid_model_output_is_distinct() -> None:
    nvidia_key = "nvapi-invalid-output-test"
    chat_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_attempts
        if request.url == NVIDIA_MODELS_URL:
            return _valid_nvidia_models_response()
        chat_attempts += 1
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "not-json"},
                    }
                ]
            },
        )

    with pytest.raises(CredentialVerificationError) as raised:
        StandaloneInterviewRuntime(
            nvidia_key,
            _transport=httpx.MockTransport(handler),
        )

    assert raised.value.code == "invalid_response"
    assert "NVIDIA accepted the model request" in str(raised.value)
    assert nvidia_key not in str(raised.value)
    assert chat_attempts == 2


def test_nvidia_rate_limit_is_distinct_and_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nvidia_key = "nvapi-rate-limit-test"
    chat_attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal chat_attempts
        if request.url == NVIDIA_MODELS_URL:
            return _valid_nvidia_models_response()
        chat_attempts += 1
        return httpx.Response(
            429,
            text=f"quota unavailable for {nvidia_key}",
        )

    monkeypatch.setattr(openai_compatible.time, "sleep", lambda _: None)
    with pytest.raises(CredentialVerificationError) as raised:
        StandaloneInterviewRuntime(
            nvidia_key,
            _transport=httpx.MockTransport(handler),
        )

    assert raised.value.code == "rate_limited"
    assert "rate-limiting" in str(raised.value)
    assert nvidia_key not in str(raised.value)
    assert chat_attempts == 3


def test_nvidia_network_failure_is_distinct_and_secret_free() -> None:
    nvidia_key = "nvapi-network-test"

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(
            f"connection failed with {nvidia_key}",
            request=request,
        )

    with pytest.raises(CredentialVerificationError) as raised:
        StandaloneInterviewRuntime(
            nvidia_key,
            _transport=httpx.MockTransport(handler),
        )

    assert raised.value.code == "network_unavailable"
    assert "could not be reached" in str(raised.value)
    assert nvidia_key not in str(raised.value)


def test_each_runtime_is_isolated_and_start_discards_the_previous_record() -> None:
    first_runtime = create_standalone_runtime(
        "nvapi-first-private-key",
        verify=False,
    )
    second_runtime = create_standalone_runtime(
        "nvapi-second-private-key",
        verify=False,
    )

    first_session = first_runtime.start()["session_id"]
    with pytest.raises(StandaloneRuntimeError, match="was not found"):
        second_runtime.state(first_session)

    replacement_session = first_runtime.start()["session_id"]
    assert replacement_session != first_session
    with pytest.raises(StandaloneRuntimeError, match="was not found"):
        first_runtime.state(first_session)
    assert first_runtime.state(replacement_session)["session_id"] == (
        replacement_session
    )

    first_runtime.clear_credentials()
    second_runtime.clear_credentials()


def test_interview_methods_return_json_safe_secret_free_dicts() -> None:
    nvidia_key = "nvapi-never-include-in-interview-output"
    runtime = create_standalone_runtime(nvidia_key, verify=False)
    results: list[dict[str, Any]] = [runtime.health(), runtime.start()]
    session_id = results[-1]["session_id"]
    results.append(runtime.consent(session_id, "consent_yes"))
    for answer in (
        "Kamau",
        "kamau@example.test",
        "31",
        "Male",
        "Nairobi",
        "Westlands",
        "Programmer",
    ):
        results.append(runtime.text(session_id, answer))
    results.append(runtime.state(session_id))
    results.append(runtime.stop(session_id))
    results.append(runtime.export(session_id))

    for result in results:
        assert isinstance(result, dict)
        _assert_json_safe_and_secret_free(result, nvidia_key)
    assert results[-1]["state"]["status"] == "stopped"

    cleared = runtime.clear_credentials()
    assert cleared["llm_enabled"] is False
    assert cleared["stt_provider"] == "disabled"
    _assert_json_safe_and_secret_free(cleared, nvidia_key)
    with pytest.raises(
        StandaloneRuntimeError,
        match="credentials are not configured",
    ) as raised:
        runtime.start()
    assert nvidia_key not in str(raised.value)


class _RecordingSpeechToText:
    def __init__(self, transcript: str = "My spoken answer") -> None:
        self.transcript = transcript
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        *,
        audio: bytes,
        filename: str,
        mime_type: str,
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        self.calls.append(
            {
                "audio": audio,
                "filename": filename,
                "mime_type": mime_type,
                "language_hint": language_hint,
            }
        )
        return TranscriptionResult(
            text=self.transcript,
            confidence=0.91,
            detected_language=language_hint,
        )


def test_voice_guards_and_successful_transcription_handoff() -> None:
    nvidia_key = "nvapi-voice-runtime-key"
    speechmatics_key = "speechmatics-voice-runtime-key"
    runtime = StandaloneInterviewRuntime(
        nvidia_key,
        speechmatics_key,
        verify=False,
    )
    provider = _RecordingSpeechToText()
    runtime._stt_provider = provider
    session_id = runtime.start()["session_id"]

    with pytest.raises(
        StandaloneRuntimeError,
        match="active consented interview",
    ):
        runtime.voice(
            session_id,
            audio=b"audio",
            filename="note.webm",
            mime_type="audio/webm",
        )

    runtime.consent(session_id, "consent_yes")
    with pytest.raises(StandaloneRuntimeError, match="Unsupported audio type"):
        runtime.voice(
            session_id,
            audio=b"audio",
            filename="note.txt",
            mime_type="text/plain",
        )
    with pytest.raises(StandaloneRuntimeError, match="Audio file is empty"):
        runtime.voice(
            session_id,
            audio=b"",
            filename="note.webm",
            mime_type="audio/webm",
        )
    with pytest.raises(StandaloneRuntimeError, match="20 MB"):
        runtime.voice(
            session_id,
            audio=b"x" * (MAX_AUDIO_BYTES + 1),
            filename="note.webm",
            mime_type="audio/webm",
        )

    response = runtime.voice(
        session_id,
        audio=b"recorded-audio",
        filename="note.webm",
        mime_type="audio/webm; codecs=opus",
        language_hint="en",
    )

    assert response["transcript"] == "My spoken answer"
    assert response["transcription_confidence"] == 0.91
    assert provider.calls == [
        {
            "audio": b"recorded-audio",
            "filename": "note.webm",
            "mime_type": "audio/webm",
            "language_hint": "en",
        }
    ]
    state = runtime.state(session_id)
    assert state["transcript"][-2]["content"] == "My spoken answer"
    assert state["transcript"][-2]["input_mode"] == "voice"
    _assert_json_safe_and_secret_free(
        response,
        nvidia_key,
        speechmatics_key,
    )

    runtime.stop(session_id)
    with pytest.raises(
        StandaloneRuntimeError,
        match="active consented interview",
    ):
        runtime.voice(
            session_id,
            audio=b"audio",
            filename="note.webm",
            mime_type="audio/webm",
        )
    runtime.clear_credentials()


def test_voice_requires_speechmatics_and_sanitizes_provider_errors() -> None:
    nvidia_key = "nvapi-voice-error-key"
    runtime = StandaloneInterviewRuntime(nvidia_key, verify=False)
    session_id = runtime.start()["session_id"]
    runtime.consent(session_id, "consent_yes")
    with pytest.raises(
        StandaloneRuntimeError,
        match="requires Speechmatics credentials",
    ):
        runtime.voice(
            session_id,
            audio=b"audio",
            filename="note.webm",
            mime_type="audio/webm",
        )
    runtime.clear_credentials()

    speechmatics_key = "speechmatics-leaky-provider-key"
    runtime = StandaloneInterviewRuntime(
        nvidia_key,
        speechmatics_key,
        verify=False,
    )

    class LeakyProvider:
        def transcribe(self, **kwargs: Any) -> TranscriptionResult:
            del kwargs
            raise RuntimeError(speechmatics_key)

    runtime._stt_provider = LeakyProvider()
    session_id = runtime.start()["session_id"]
    runtime.consent(session_id, "consent_yes")
    with pytest.raises(StandaloneRuntimeError) as raised:
        runtime.voice(
            session_id,
            audio=b"audio",
            filename="note.webm",
            mime_type="audio/webm",
        )
    assert str(raised.value) == "Voice transcription failed"
    assert speechmatics_key not in str(raised.value)
    runtime.clear_credentials()


def test_standalone_configuration_does_not_mutate_environment() -> None:
    before = dict(os.environ)
    runtime = create_standalone_runtime(
        "nvapi-env-isolation-key",
        verify=False,
    )

    assert dict(os.environ) == before
    assert runtime.settings.llm_base_url == NVIDIA_BASE_URL
    assert runtime.settings.llm_model == NVIDIA_MODEL
    assert runtime.settings.repository_backend == "memory"
    assert runtime.settings.supabase_enabled is False
    assert runtime.settings.supabase_url is None
    assert runtime.settings.supabase_service_role_key is None
    runtime.clear_credentials()


def test_structured_logging_redacts_all_provider_credential_names() -> None:
    credential = "provider-log-canary"
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Static safe message",
        args=(),
        exc_info=None,
    )
    record.context = {
        "llm_api_key": credential,
        "nvidia_api_key": credential,
        "stt_api_key": credential,
        "speechmatics_api_key": credential,
    }

    serialized = JsonFormatter().format(record)
    payload = json.loads(serialized)

    assert credential not in serialized
    assert set(payload["context"].values()) == {"[REDACTED]"}
