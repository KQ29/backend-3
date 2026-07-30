from __future__ import annotations

from scripts.run_demo import build_child_environment


def test_launcher_overrides_stale_streamlit_api_url(
    monkeypatch,
) -> None:
    monkeypatch.setenv("STREAMLIT_API_URL", "http://127.0.0.1:8000")

    environment = build_child_environment("127.0.0.1", 8001)

    assert environment["STREAMLIT_API_URL"] == "http://127.0.0.1:8001"


def test_launcher_uses_loopback_for_wildcard_api_host(
    monkeypatch,
) -> None:
    monkeypatch.delenv("STREAMLIT_API_URL", raising=False)

    environment = build_child_environment("0.0.0.0", 9000)

    assert environment["STREAMLIT_API_URL"] == "http://127.0.0.1:9000"
