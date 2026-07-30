from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_child_environment(api_host: str, api_port: int) -> dict[str, str]:
    child_environment = os.environ.copy()
    client_host = "127.0.0.1" if api_host in {"0.0.0.0", "::"} else api_host
    child_environment["STREAMLIT_API_URL"] = (
        f"http://{client_host}:{api_port}"
    )
    return child_environment


def wait_for_api(
    process: subprocess.Popen[bytes],
    host: str,
    port: int,
    *,
    timeout_seconds: float = 15,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"FastAPI exited early with status {process.returncode}")
        connection_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        try:
            connection = HTTPConnection(connection_host, port, timeout=0.5)
            connection.request("GET", "/health")
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except (OSError, TimeoutError):
            time.sleep(0.2)
    raise RuntimeError(f"FastAPI did not become ready on {host}:{port}")


def require_free_port(host: str, port: int, label: str) -> None:
    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((bind_host, port))
        except OSError as exc:
            raise RuntimeError(
                f"{label} port {port} is already in use. Stop the existing "
                "process or choose another port in .env."
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
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    api_host = os.getenv("API_HOST", "127.0.0.1")
    api_port = int(os.getenv("API_PORT", "8000"))
    streamlit_port = int(os.getenv("STREAMLIT_PORT", "8501"))
    child_environment = build_child_environment(api_host, api_port)

    api: subprocess.Popen[bytes] | None = None
    interface: subprocess.Popen[bytes] | None = None
    try:
        if api_port == streamlit_port:
            raise ValueError("API_PORT and STREAMLIT_PORT must be different")
        require_free_port(api_host, api_port, "FastAPI")
        require_free_port(api_host, streamlit_port, "Streamlit")

        api = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--host",
                api_host,
                "--port",
                str(api_port),
            ],
            cwd=PROJECT_ROOT,
            env=child_environment,
            start_new_session=True,
        )
        wait_for_api(api, api_host, api_port)
        interface = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                "streamlit_app.py",
                "--server.address",
                api_host,
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
            f"\nDemo ready: http://{api_host}:{streamlit_port}\n"
            f"API docs:  http://{api_host}:{api_port}/docs\n"
            "Press Ctrl+C to stop both services.",
            flush=True,
        )

        while api.poll() is None and interface.poll() is None:
            time.sleep(0.5)
        if api.poll() is not None:
            raise RuntimeError(f"FastAPI stopped with status {api.returncode}")
        raise RuntimeError(f"Streamlit stopped with status {interface.returncode}")
    except KeyboardInterrupt:
        print("\nStopping demo...", flush=True)
        return 0
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Unable to run demo: {exc}", file=sys.stderr)
        return 1
    finally:
        stop_process(interface)
        stop_process(api)


if __name__ == "__main__":
    raise SystemExit(main())
