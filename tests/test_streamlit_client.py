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
