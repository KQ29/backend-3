from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import streamlit_app
from app.services.standalone import (
    StandaloneInterviewRuntime,
    StandaloneRuntimeError,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeSessionState(dict):
    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value) -> None:
        self[name] = value


@pytest.fixture
def session_state(monkeypatch: pytest.MonkeyPatch) -> FakeSessionState:
    state = FakeSessionState()
    monkeypatch.setattr(streamlit_app.st, "session_state", state)
    streamlit_app.initialize_session_state()
    return state


@pytest.fixture
def runtime() -> StandaloneInterviewRuntime:
    configured = StandaloneInterviewRuntime(
        "nvidia-test-canary",
        verify=False,
    )
    yield configured
    configured.clear_credentials()


def test_streamlit_uses_one_runtime_from_session_state(
    session_state: FakeSessionState,
    runtime: StandaloneInterviewRuntime,
) -> None:
    session_state[streamlit_app.RUNTIME_SESSION_KEY] = runtime

    assert streamlit_app.current_runtime() is runtime
    assert streamlit_app.current_runtime() is runtime


def test_clear_credentials_discards_runtime_interview_and_widget_values(
    session_state: FakeSessionState,
    runtime: StandaloneInterviewRuntime,
) -> None:
    started = runtime.start()
    session_state[streamlit_app.RUNTIME_SESSION_KEY] = runtime
    session_state.session_id = started["session_id"]
    session_state[streamlit_app.NVIDIA_WIDGET_KEY] = "nvidia-test-canary"
    session_state[streamlit_app.SPEECHMATICS_WIDGET_KEY] = "stt-test-canary"
    session_state["recorded_audio_0"] = b"recorded-audio"
    session_state["uploaded_audio_0"] = b"uploaded-audio"

    streamlit_app.clear_credentials_and_interview_data()

    assert streamlit_app.RUNTIME_SESSION_KEY not in session_state
    assert session_state.session_id is None
    assert streamlit_app.NVIDIA_WIDGET_KEY not in session_state
    assert streamlit_app.SPEECHMATICS_WIDGET_KEY not in session_state
    assert "recorded_audio_0" not in session_state
    assert "uploaded_audio_0" not in session_state
    assert runtime.health()["llm_enabled"] is False


def test_unexpected_runtime_error_does_not_expose_credential() -> None:
    credential = "nvidia-test-canary"

    def fail() -> None:
        raise RuntimeError(credential)

    with pytest.raises(streamlit_app.DemoRuntimeError) as captured:
        streamlit_app.runtime_action(fail)

    assert credential not in str(captured.value)


def test_safe_runtime_error_is_preserved() -> None:
    with pytest.raises(
        streamlit_app.DemoRuntimeError,
        match="Voice transcription failed",
    ):
        streamlit_app.runtime_action(
            lambda: (_ for _ in ()).throw(
                StandaloneRuntimeError("Voice transcription failed")
            )
        )


def test_fallback_analysis_is_explicitly_labeled() -> None:
    warning = streamlit_app.analysis_source_warning(
        "local_provider_fallback"
    )

    assert warning is not None
    assert "NVIDIA request failed" in warning
    assert streamlit_app.analysis_source_warning("llm") is None


def test_streamlit_has_no_remote_backend_or_stored_provider_configuration() -> None:
    source = (PROJECT_ROOT / "streamlit_app.py").read_text(encoding="utf-8")

    for forbidden_name in (
        "STREAMLIT_API_URL",
        "BACKEND_API_TOKEN",
        "LLM_API_KEY",
        "SPEECHMATICS_API_KEY",
        "st.cache_data",
        "st.cache_resource",
        "api_call(",
        "api_upload(",
    ):
        assert forbidden_name not in source


def test_streamlit_initial_view_is_a_masked_credential_gate() -> None:
    app = AppTest.from_file(
        str(PROJECT_ROOT / "streamlit_app.py"),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert [item.label for item in app.text_input] == [
        "NVIDIA API key",
        "Speechmatics API key (optional)",
    ]
    assert all(
        item.proto.type == item.proto.PASSWORD
        for item in app.text_input
    )
    assert all(
        item.autocomplete == "new-password" for item in app.text_input
    )
    assert "Verify and continue" in [item.label for item in app.button]
    assert "Start demo interview" not in [
        item.label for item in app.button
    ]
    assert any(
        "identifying demographics and verbatim answers" in str(item.value)
        for item in app.markdown
    )
