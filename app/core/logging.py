from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

SENSITIVE_KEYS = {
    "admin_api_token",
    "authorization",
    "api_key",
    "apikey",
    "access_token",
    "backend_api_token",
    "llm_api_key",
    "nvidia_api_key",
    "speechmatics_api_key",
    "stt_api_key",
    "phone",
    "email",
    "raw_audio",
    "raw_response",
    "response_text",
    "supabase_service_role_key",
    "supabase_secret_key",
    "x_backend_token",
}


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            normalized = key.lower().replace("-", "_")
            result[key] = (
                "[REDACTED]" if normalized in SENSITIVE_KEYS else _sanitize(item)
            )
        return result
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        context = getattr(record, "context", None)
        if isinstance(context, dict):
            payload["context"] = _sanitize(context)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
