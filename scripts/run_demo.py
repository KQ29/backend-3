from __future__ import annotations

import os
import socket
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENSITIVE_PROVIDER_ENV_KEYS = {
    "BACKEND_API_TOKEN",
    "LLM_API_KEY",
    "NVIDIA_API_KEY",
    "SPEECHMATICS_API_KEY",
    "STT_API_KEY",
    "SUPABASE_SECRET_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
}


def build_child_environment(
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build a Streamlit environment without inherited provider credentials."""

    child_environment = dict(os.environ if environment is None else environment)
    for key in SENSITIVE_PROVIDER_ENV_KEYS:
        child_environment.pop(key, None)
    child_environment.pop("STREAMLIT_API_URL", None)
    return child_environment


def require_free_port(host: str, port: int, label: str) -> None:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((bind_host, port))
        except OSError as exc:
            raise RuntimeError(
                f"{label} port {port} is already in use. Stop the existing "
                "process or choose another STREAMLIT_PORT."
            ) from exc


def stop_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    streamlit_host = os.getenv("STREAMLIT_HOST", "127.0.0.1")
    streamlit_port = int(os.getenv("STREAMLIT_PORT", "8502"))
    child_environment = build_child_environment()

    interface: subprocess.Popen[bytes] | None = None
    try:
        require_free_port(streamlit_host, streamlit_port, "Streamlit")
        interface = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_app.py",
                "--server.address",
                streamlit_host,
                "--server.port",
                str(streamlit_port),
                "--server.headless",
                "true",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=PROJECT_ROOT,
            env=child_environment,
            start_new_session=True,
        )
        print(
            f"\nDemo ready: http://{streamlit_host}:{streamlit_port}\n"
            "Enter provider credentials in the browser. Press Ctrl+C to stop.",
            flush=True,
        )
        return_code = interface.wait()
        if return_code != 0:
            raise RuntimeError(f"Streamlit stopped with status {return_code}")
        return 0
    except KeyboardInterrupt:
        print("\nStopping demo...", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Unable to run demo: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_process(interface)


if __name__ == "__main__":
    raise SystemExit(main())
