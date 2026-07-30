from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.models.domain import (
    AnswerAnalysis,
    Polarity,
    ProbeReason,
    ProbeStrategy,
)
from app.providers.stt.base import TranscriptionResult
from app.repositories.memory import MemoryInterviewRepository


def test_health_start_and_state_contract(client: TestClient) -> None:
    health = client.get("/health")
    started = client.post(
        "/api/v1/interviews/start",
        json={"channel": "streamlit_demo"},
    )
    session_id = started.json()["session_id"]
    state = client.get(f"/api/v1/interviews/{session_id}/state")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert health.json()["max_probes_per_anchor"] == 2
    assert started.status_code == 201
    assert started.json()["allowed_choices"] == ["consent_yes", "consent_no"]
    assert started.json()["max_probes_per_anchor"] == 2
    assert started.json()["probe_counts"] == {}
    assert state.status_code == 200
    assert state.json()["transcript"][0]["content"].startswith("Karibu.")
    assert state.json()["max_probes_per_anchor"] == 2
    assert state.json()["probe_counts"] == {}


def test_voice_endpoint_uses_mock_transcript_and_tracks_confidence(
    client: TestClient,
) -> None:
    started = client.post(
        "/api/v1/interviews/start",
        json={"channel": "streamlit_demo"},
    ).json()
    session_id = started["session_id"]
    client.post(
        f"/api/v1/interviews/{session_id}/consent",
        json={"choice": "consent_yes"},
    )
    for index in range(7):
        response = client.post(
            f"/api/v1/interviews/{session_id}/voice",
            files={
                "audio": (
                    f"answer-{index}.wav",
                    b"RIFF-mock-audio",
                    "audio/wav",
                )
            },
        )
        assert response.status_code == 200

    response = client.post(
        f"/api/v1/interviews/{session_id}/voice",
        files={
            "audio": (
                "substantive.wav",
                b"RIFF-mock-substantive-audio",
                "audio/wav",
            )
        },
    )
    exported = client.get(f"/api/v1/interviews/{session_id}/export").json()

    assert response.status_code == 200
    assert response.json()["mode_of_input"] == "voice"
    assert response.json()["transcript"] == "Mock transcription"
    assert exported["tags"][0]["transcription_confidence"] == 0.9
    assert exported["turns"][-2]["input_mode"] == "voice"


def test_declined_consent_and_closed_session_contract(client: TestClient) -> None:
    started = client.post(
        "/api/v1/interviews/start",
        json={"channel": "streamlit_demo"},
    ).json()
    session_id = started["session_id"]
    declined = client.post(
        f"/api/v1/interviews/{session_id}/consent",
        json={"choice": "consent_no"},
    )
    rejected = client.post(
        f"/api/v1/interviews/{session_id}/text",
        json={"text": "This must not be stored."},
    )
    state = client.get(f"/api/v1/interviews/{session_id}/state").json()

    assert declined.status_code == 200
    assert declined.json()["status"] == "declined"
    assert rejected.status_code == 409
    assert len(state["transcript"]) == 1
    assert state["tags"] == []


def test_unknown_session_and_validation_errors(client: TestClient) -> None:
    missing = client.get("/api/v1/interviews/not-a-session/state")
    started = client.post(
        "/api/v1/interviews/start",
        json={"channel": "streamlit_demo"},
    ).json()
    invalid_voice = client.post(
        f"/api/v1/interviews/{started['session_id']}/voice",
        json={"transcript": "", "transcription_confidence": 2},
    )

    assert missing.status_code == 404
    assert invalid_voice.status_code == 422


def test_voice_validation_happens_before_provider_call(
    client: TestClient,
) -> None:
    class CountingSTT:
        def __init__(self) -> None:
            self.calls = 0

        def transcribe(self, **kwargs) -> TranscriptionResult:
            del kwargs
            self.calls += 1
            return TranscriptionResult(text="Should not be called")

    provider = CountingSTT()
    client.app.state.speech_to_text_provider = provider
    session_id = client.post(
        "/api/v1/interviews/start",
        json={"channel": "streamlit_demo"},
    ).json()["session_id"]

    before_consent = client.post(
        f"/api/v1/interviews/{session_id}/voice",
        files={"audio": ("answer.wav", b"RIFF-audio", "audio/wav")},
    )
    assert before_consent.status_code == 409
    assert provider.calls == 0

    client.post(
        f"/api/v1/interviews/{session_id}/consent",
        json={"choice": "consent_yes"},
    )
    unsupported = client.post(
        f"/api/v1/interviews/{session_id}/voice",
        files={"audio": ("answer.txt", b"not-audio", "text/plain")},
    )
    assert unsupported.status_code == 415
    assert provider.calls == 0

    empty = client.post(
        f"/api/v1/interviews/{session_id}/voice",
        files={"audio": ("answer.wav", b"", "audio/wav")},
    )
    assert empty.status_code == 422
    assert provider.calls == 0


def test_batch_audit_reports_not_run_without_provider(
    client: TestClient,
) -> None:
    session_id = client.post(
        "/api/v1/interviews/start",
        json={"channel": "streamlit_demo"},
    ).json()["session_id"]
    response = client.post(f"/api/v1/internal/interviews/{session_id}/batch-audit")

    assert response.status_code == 200
    assert response.json()["status"] == "not_run"
    assert "has not been run" in response.json()["message"]


def test_fastapi_uses_llm_result_for_routing_and_exposes_source() -> None:
    class ApiLLMProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(self, **kwargs) -> AnswerAnalysis:
            del kwargs
            return AnswerAnalysis(
                polarity=Polarity.NEGATIVE,
                confidence=0.95,
                concrete=True,
                on_topic=True,
                economic_outcome="no_change",
                reflection="You described no measurable change after the training.",
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
        llm_max_calls_per_session=16,
        max_probes_per_anchor=2,
    )
    app = create_app(
        settings=settings,
        repository=MemoryInterviewRepository(),
        llm_provider=ApiLLMProvider(),
    )
    with TestClient(app) as live_client:
        session_id = live_client.post(
            "/api/v1/interviews/start",
            json={"channel": "streamlit_demo"},
        ).json()["session_id"]
        live_client.post(
            f"/api/v1/interviews/{session_id}/consent",
            json={"choice": "consent_yes"},
        )
        for answer in (
            "Kamau Otieno",
            "kamau.demo@example.com",
            "31",
            "Male",
            "Nairobi",
            "Westlands",
            "Digital marketing officer",
        ):
            live_client.post(
                f"/api/v1/interviews/{session_id}/text",
                json={"text": answer},
            )

        response = live_client.post(
            f"/api/v1/interviews/{session_id}/text",
            json={"text": ("I got a promotion in March 2026 after using AI at work.")},
        )

    assert response.status_code == 200
    assert response.json()["question_id"] == "anchor_2"
    assert response.json()["analysis_source"] == "llm"
    assert response.json()["analysis_confidence"] == 0.95
    assert response.json()["llm_calls_used"] == 1
    assert response.json()["needs_probe"] is False
    assert response.json()["probe_type"] == "none"
    assert response.json()["probe_reason"] is None
    assert response.json()["probe_asked"] is False


def test_fastapi_exposes_the_probe_decision_and_whether_it_was_asked() -> None:
    class ClarifyingApiLLMProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(self, **kwargs) -> AnswerAnalysis:
            del kwargs
            return AnswerAnalysis(
                polarity=Polarity.NEUTRAL,
                confidence=0.92,
                vague=True,
                on_topic=True,
                probe_strategy=ProbeStrategy.CLARITY,
                probe_reason=ProbeReason.VAGUE_OR_UNCLEAR,
                reflection="You use AI for emails.",
                suggested_probe=(
                    "Can you describe the last email you used AI for?"
                ),
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
        llm_max_calls_per_session=16,
    )
    app = create_app(
        settings=settings,
        repository=MemoryInterviewRepository(),
        llm_provider=ClarifyingApiLLMProvider(),
    )
    with TestClient(app) as live_client:
        session_id = live_client.post(
            "/api/v1/interviews/start",
            json={"channel": "streamlit_demo"},
        ).json()["session_id"]
        live_client.post(
            f"/api/v1/interviews/{session_id}/consent",
            json={"choice": "consent_yes"},
        )
        for answer in (
            "Kamau Otieno",
            "kamau.demo@example.com",
            "31",
            "Male",
            "Nairobi",
            "Westlands",
            "Digital marketing officer",
        ):
            live_client.post(
                f"/api/v1/interviews/{session_id}/text",
                json={"text": answer},
            )

        first_probe = live_client.post(
            f"/api/v1/interviews/{session_id}/text",
            json={"text": "I use AI for emails."},
        )
        second_probe = live_client.post(
            f"/api/v1/interviews/{session_id}/text",
            json={"text": "It helps with work."},
        )

    first_payload = first_probe.json()
    assert first_probe.status_code == 200
    assert first_payload["question_id"] == "anchor_1_probe"
    assert first_payload["needs_probe"] is True
    assert first_payload["probe_type"] == "clarity"
    assert first_payload["probe_reason"] == "vague_or_unclear"
    assert first_payload["probe_asked"] is True
    assert first_payload["probe_number"] == 1
    assert first_payload["probe_strategy"] == "clarity"
    assert first_payload["probe_counts"] == {"anchor_1": 1}
    assert first_payload["max_probes_per_anchor"] == 2

    second_payload = second_probe.json()
    assert second_probe.status_code == 200
    assert second_payload["question_id"] == "anchor_1_probe"
    assert second_payload["probe_asked"] is True
    assert second_payload["probe_number"] == 2
    assert second_payload["probe_counts"] == {"anchor_1": 2}
    assert second_payload["probes_used"] == [
        "anchor_1_probe",
        "anchor_1_probe",
    ]
