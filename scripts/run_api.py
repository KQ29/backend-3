from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLOUD_HOST = "0.0.0.0"
DEFAULT_API_PORT = 8000


def resolve_port(environment: Mapping[str, str] | None = None) -> int:
    """Resolve the managed-host port, preferring PORT over local API_PORT."""

    resolved_environment = os.environ if environment is None else environment
    raw_port = (
        resolved_environment.get("PORT")
        or resolved_environment.get("API_PORT")
        or str(DEFAULT_API_PORT)
    )
    try:
        port = int(raw_port)
    except ValueError as exc:
        raise ValueError("PORT or API_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError("PORT or API_PORT must be between 1 and 65535")
    return port


def main() -> None:
    uvicorn.run(
        "app.main:app",
        host=CLOUD_HOST,
        port=resolve_port(),
        workers=1,
        app_dir=str(PROJECT_ROOT),
        proxy_headers=True,
    )


if __name__ == "__main__":
    main()
