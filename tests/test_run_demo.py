from __future__ import annotations

from scripts.run_demo import SENSITIVE_PROVIDER_ENV_KEYS, build_child_environment


def test_launcher_does_not_pass_backend_or_provider_credentials() -> None:
    environment = {
        "PATH": "/example/bin",
        "STREAMLIT_API_URL": "http://127.0.0.1:8001",
        **{key: f"secret-{key}" for key in SENSITIVE_PROVIDER_ENV_KEYS},
    }

    child_environment = build_child_environment(environment)

    assert child_environment == {"PATH": "/example/bin"}


def test_launcher_preserves_non_sensitive_environment_values() -> None:
    environment = {
        "PATH": "/example/bin",
        "LANG": "en_US.UTF-8",
        "STREAMLIT_PORT": "8502",
    }

    child_environment = build_child_environment(environment)

    assert child_environment == environment
