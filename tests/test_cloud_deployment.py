from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from dotenv import dotenv_values

from scripts import run_api

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cloud_port_takes_precedence_over_local_api_port() -> None:
    assert run_api.resolve_port({"PORT": "10000", "API_PORT": "8001"}) == 10000
    assert run_api.resolve_port({"API_PORT": "8001"}) == 8001
    assert run_api.resolve_port({}) == 8000


@pytest.mark.parametrize("value", ["not-a-port", "0", "65536"])
def test_cloud_port_must_be_valid(value: str) -> None:
    with pytest.raises(ValueError, match="PORT or API_PORT"):
        run_api.resolve_port({"PORT": value})


def test_cloud_launcher_binds_publicly_with_one_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_run(app: str, **kwargs: Any) -> None:
        captured["app"] = app
        captured.update(kwargs)

    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setattr(run_api.uvicorn, "run", fake_run)

    run_api.main()

    assert captured == {
        "app": "app.main:app",
        "host": "0.0.0.0",
        "port": 10000,
        "workers": 1,
        "app_dir": str(PROJECT_ROOT),
        "proxy_headers": True,
    }


def test_render_manifest_defines_memory_fastapi_service_and_secret_prompts() -> None:
    manifest = (PROJECT_ROOT / "render.yaml").read_text(encoding="utf-8")

    for required_line in (
        "type: web",
        "runtime: python",
        "buildCommand: python -m pip install -r requirements.txt",
        "startCommand: python scripts/run_api.py",
        "healthCheckPath: /health",
        "value: memory",
        'value: "false"',
        "value: speechmatics",
        "value: https://integrate.api.nvidia.com/v1",
    ):
        assert required_line in manifest

    for secret_name in (
        "BACKEND_API_TOKEN",
        "LLM_API_KEY",
        "SPEECHMATICS_API_KEY",
    ):
        assert f"- key: {secret_name}\n        sync: false" in manifest

    assert "- key: SUPABASE_URL" not in manifest
    assert "- key: SUPABASE_SECRET_KEY" not in manifest


def test_cloud_runtime_files_are_explicitly_pinned() -> None:
    assert (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8") == "3.13.3\n"

    requirements = (
        PROJECT_ROOT / "requirements.txt"
    ).read_text(encoding="utf-8").splitlines()
    assert requirements
    assert all("==" in requirement for requirement in requirements)


def test_streamlit_secret_template_contains_only_frontend_configuration() -> None:
    template = tomllib.loads(
        (
            PROJECT_ROOT / ".streamlit" / "secrets.example.toml"
        ).read_text(encoding="utf-8")
    )

    assert set(template) == {
        "STREAMLIT_API_URL",
        "BACKEND_API_TOKEN",
        "LLM_TIMEOUT_SECONDS",
        "STT_TIMEOUT_SECONDS",
        "STREAMLIT_API_TIMEOUT_SECONDS",
        "STREAMLIT_VOICE_TIMEOUT_SECONDS",
    }
    assert template["STREAMLIT_API_URL"].startswith("https://")
    assert "127.0.0.1" not in template["STREAMLIT_API_URL"]
    assert "localhost" not in template["STREAMLIT_API_URL"]


def test_environment_example_is_unique_and_contains_no_credentials() -> None:
    path = PROJECT_ROOT / ".env.example"
    content = path.read_text(encoding="utf-8")
    keys = [
        line.split("=", maxsplit=1)[0]
        for line in content.splitlines()
        if line and not line.startswith("#") and "=" in line
    ]
    values = dotenv_values(path)

    assert len(keys) == len(set(keys))
    for secret_name in (
        "BACKEND_API_TOKEN",
        "SUPABASE_SECRET_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
        "LLM_API_KEY",
        "SPEECHMATICS_API_KEY",
        "ADMIN_API_TOKEN",
    ):
        assert values[secret_name] == ""
    assert "nvapi-" not in content
    assert "eyJhbGciOi" not in content


def test_real_local_secret_files_are_ignored() -> None:
    ignore_rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")

    assert "\n.env\n" in f"\n{ignore_rules}"
    assert ".streamlit/secrets.toml" in ignore_rules
    assert "!.streamlit/secrets.example.toml" in ignore_rules
