from __future__ import annotations

import httpx
import pytest

import streamlit_app


def test_streamlit_waits_longer_than_the_backend_llm_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            captured["timeout"] = float(kwargs["timeout"])

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def request(self, *args, **kwargs) -> httpx.Response:
            del args, kwargs
            return httpx.Response(200, json={"status": "ok"})

    monkeypatch.setattr(streamlit_app.httpx, "Client", FakeClient)

    response = streamlit_app.api_call("GET", "/health")

    assert response == {"status": "ok"}
    assert captured["timeout"] >= 120
    assert captured["timeout"] > streamlit_app.LLM_TIMEOUT_SECONDS


def test_streamlit_timeout_does_not_incorrectly_call_fastapi_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TimeoutClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def request(self, *args, **kwargs) -> httpx.Response:
            del args, kwargs
            raise httpx.ReadTimeout("model is still responding")

    monkeypatch.setattr(streamlit_app.httpx, "Client", TimeoutClient)

    with pytest.raises(
        streamlit_app.DemoApiError,
        match="backend may still finish",
    ):
        streamlit_app.api_call("POST", "/api/v1/interviews/example/text")


def test_json_requests_send_configured_backend_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def request(self, *args, **kwargs) -> httpx.Response:
            del args
            captured.update(kwargs)
            return httpx.Response(200, json={"status": "ok"})

    monkeypatch.setattr(streamlit_app, "BACKEND_API_TOKEN", "shared-secret")
    monkeypatch.setattr(streamlit_app.httpx, "Client", FakeClient)

    streamlit_app.api_call("GET", "/health")

    assert captured["headers"] == {"X-Backend-Token": "shared-secret"}


def test_upload_requests_send_configured_backend_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def post(self, *args, **kwargs) -> httpx.Response:
            del args
            captured.update(kwargs)
            return httpx.Response(200, json={"transcript": "hello"})

    monkeypatch.setattr(streamlit_app, "BACKEND_API_TOKEN", "shared-secret")
    monkeypatch.setattr(streamlit_app.httpx, "Client", FakeClient)

    streamlit_app.api_upload(
        "/api/v1/interviews/example/voice",
        filename="answer.wav",
        content=b"audio",
        mime_type="audio/wav",
    )

    assert captured["headers"] == {"X-Backend-Token": "shared-secret"}


def test_backend_token_is_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def request(self, *args, **kwargs) -> httpx.Response:
            del args
            captured.update(kwargs)
            return httpx.Response(200, json={"status": "ok"})

    monkeypatch.setattr(streamlit_app, "BACKEND_API_TOKEN", "")
    monkeypatch.setattr(streamlit_app.httpx, "Client", FakeClient)

    streamlit_app.api_call("GET", "/health")

    assert captured["headers"] == {}


def test_connection_error_does_not_expose_backend_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingClient:
        def __init__(self, **kwargs) -> None:
            del kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args) -> None:
            del args

        def request(self, *args, **kwargs) -> httpx.Response:
            del args, kwargs
            raise httpx.ConnectError("connection failed")

    token = "do-not-expose-this-token"
    monkeypatch.setattr(streamlit_app, "BACKEND_API_TOKEN", token)
    monkeypatch.setattr(streamlit_app.httpx, "Client", FailingClient)

    with pytest.raises(streamlit_app.DemoApiError) as exc_info:
        streamlit_app.api_call("GET", "/health")

    assert "configured FastAPI backend" in str(exc_info.value)
    assert token not in str(exc_info.value)
