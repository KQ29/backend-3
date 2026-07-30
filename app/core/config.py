from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr, field_validator, model_validator

load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class Settings(BaseModel):
    """Validated runtime settings.

    Secrets are accepted only from environment variables and are represented
    with ``SecretStr`` so accidental model serialization does not reveal them.
    """

    app_name: str = "Otermans Kenya Interviewer"
    app_env: str = "development"
    log_level: str = "INFO"
    api_host: str = "127.0.0.1"
    api_port: int = Field(default=8000, ge=1, le=65535)

    nudge_after_hours: int = Field(default=10, ge=1)
    abandon_after_hours: int = Field(default=24, ge=2)
    inactivity_check_seconds: int = Field(default=3600, ge=60, le=86400)
    max_substantive_turns: int = Field(default=16, ge=8, le=40)
    max_probes_per_anchor: int = Field(default=2, ge=0, le=2)

    llm_enabled: bool = False
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    llm_model: str | None = None
    llm_mode: str = "always"
    llm_timeout_seconds: int = Field(default=30, ge=5, le=180)
    llm_max_calls_per_session: int = Field(default=16, ge=0, le=40)
    llm_low_confidence_threshold: float = Field(default=0.70, ge=0, le=1)
    llm_max_output_tokens: int = Field(default=512, ge=128, le=2048)

    stt_provider: str = "mock"
    stt_api_key: SecretStr | None = None
    stt_base_url: str | None = None
    stt_model: str | None = None
    stt_language: str = "auto"
    stt_timeout_seconds: int = Field(default=60, ge=5, le=300)
    stt_max_audio_bytes: int = Field(
        default=20 * 1024 * 1024,
        ge=1024,
        le=100 * 1024 * 1024,
    )

    supabase_enabled: bool = False
    repository_backend: str = "memory"
    supabase_url: str | None = None
    supabase_service_role_key: SecretStr | None = None
    database_url: SecretStr | None = None
    admin_api_token: SecretStr | None = None
    backend_api_token: SecretStr | None = None

    @field_validator("app_env")
    @classmethod
    def normalize_app_environment(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("APP_ENV cannot be empty")
        return normalized

    @field_validator("backend_api_token")
    @classmethod
    def normalize_backend_api_token(
        cls,
        value: SecretStr | None,
    ) -> SecretStr | None:
        if value is None:
            return None
        normalized = value.get_secret_value().strip()
        if not normalized:
            return None
        return SecretStr(normalized)

    @field_validator("log_level")
    @classmethod
    def normalize_log_level(cls, value: str) -> str:
        normalized = value.upper()
        if normalized not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
            raise ValueError(
                "LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR or CRITICAL"
            )
        return normalized

    @field_validator("abandon_after_hours")
    @classmethod
    def validate_abandonment_window(cls, value: int, info):
        nudge_after = info.data.get("nudge_after_hours", 10)
        if value <= nudge_after:
            raise ValueError(
                "ABANDON_AFTER_HOURS must be greater than NUDGE_AFTER_HOURS"
            )
        return value

    @field_validator("stt_provider")
    @classmethod
    def normalize_stt_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("STT_PROVIDER cannot be empty")
        return normalized

    @field_validator("repository_backend")
    @classmethod
    def normalize_repository_backend(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"memory", "supabase"}:
            raise ValueError("INTERVIEW_REPOSITORY must be memory or supabase")
        return normalized

    @field_validator("llm_mode")
    @classmethod
    def normalize_llm_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"always", "fallback"}:
            raise ValueError("LLM_MODE must be always or fallback")
        return normalized

    @model_validator(mode="after")
    def validate_provider_configuration(self) -> Settings:
        if self.app_env == "production" and (
            self.backend_api_token is None
            or not self.backend_api_token.get_secret_value()
        ):
            raise ValueError(
                "BACKEND_API_TOKEN is required when APP_ENV=production"
            )
        if self.llm_enabled and (
            self.llm_api_key is None or not self.llm_base_url or not self.llm_model
        ):
            raise ValueError(
                "LLM_BASE_URL, LLM_API_KEY and LLM_MODEL are required when "
                "LLM_ENABLED=true"
            )
        if self.stt_provider == "speechmatics" and (
            self.stt_api_key is None or not self.stt_base_url
        ):
            raise ValueError(
                "SPEECHMATICS_API_KEY (or STT_API_KEY) and STT_BASE_URL are "
                "required for Speechmatics"
            )
        if self.stt_provider not in {"mock", "speechmatics"}:
            raise ValueError("STT_PROVIDER must be mock or speechmatics")
        if (self.supabase_enabled or self.repository_backend == "supabase") and (
            self.supabase_url is None or self.supabase_service_role_key is None
        ):
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SECRET_KEY (or the legacy "
                "SUPABASE_SERVICE_ROLE_KEY) are required for Supabase persistence"
            )
        return self

    @classmethod
    def from_environment(cls) -> Settings:
        llm_base_url = os.getenv("LLM_BASE_URL") or None
        llm_api_key = os.getenv("LLM_API_KEY") or os.getenv("NVIDIA_API_KEY") or None
        llm_model = os.getenv("LLM_MODEL") or os.getenv("NVIDIA_MODEL") or None
        speechmatics_key = (
            os.getenv("SPEECHMATICS_API_KEY") or os.getenv("STT_API_KEY") or None
        )
        supabase_url = os.getenv("SUPABASE_URL") or None
        supabase_key = (
            os.getenv("SUPABASE_SERVICE_ROLE_KEY")
            or os.getenv("SUPABASE_SECRET_KEY")
            or None
        )
        inferred_llm_enabled = bool(llm_base_url and llm_api_key and llm_model)
        inferred_stt_provider = "speechmatics" if speechmatics_key else "mock"
        inferred_supabase_enabled = bool(supabase_url and supabase_key)

        return cls(
            app_env=os.getenv("APP_ENV", "development"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            api_host=os.getenv("API_HOST", "127.0.0.1"),
            api_port=int(os.getenv("API_PORT", "8000")),
            nudge_after_hours=int(os.getenv("NUDGE_AFTER_HOURS", "10")),
            abandon_after_hours=int(os.getenv("ABANDON_AFTER_HOURS", "24")),
            inactivity_check_seconds=int(os.getenv("INACTIVITY_CHECK_SECONDS", "3600")),
            max_substantive_turns=int(os.getenv("MAX_SUBSTANTIVE_TURNS", "16")),
            max_probes_per_anchor=int(
                os.getenv("MAX_PROBES_PER_ANCHOR", "2")
            ),
            llm_enabled=_env_bool("LLM_ENABLED", inferred_llm_enabled),
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=llm_model,
            llm_mode=os.getenv("LLM_MODE", "always"),
            llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
            llm_max_calls_per_session=int(
                os.getenv(
                    "LLM_MAX_CALLS_PER_SESSION",
                    os.getenv("MAX_GEMINI_CALLS_PER_SESSION", "16"),
                )
            ),
            llm_low_confidence_threshold=float(
                os.getenv(
                    "LLM_LOW_CONFIDENCE_THRESHOLD",
                    os.getenv("MIN_CONFIDENCE_FOR_GEMINI", "0.70"),
                )
            ),
            llm_max_output_tokens=int(os.getenv("LLM_MAX_OUTPUT_TOKENS", "512")),
            stt_provider=os.getenv("STT_PROVIDER", inferred_stt_provider),
            stt_api_key=speechmatics_key,
            stt_base_url=(
                os.getenv("STT_BASE_URL")
                or ("https://asr.api.speechmatics.com/v2" if speechmatics_key else None)
            ),
            stt_model=os.getenv("STT_MODEL") or None,
            stt_language=os.getenv("STT_LANGUAGE", "auto"),
            stt_timeout_seconds=int(os.getenv("STT_TIMEOUT_SECONDS", "60")),
            stt_max_audio_bytes=int(
                os.getenv("STT_MAX_AUDIO_BYTES", str(20 * 1024 * 1024))
            ),
            supabase_enabled=_env_bool(
                "SUPABASE_ENABLED",
                inferred_supabase_enabled,
            ),
            repository_backend=os.getenv(
                "INTERVIEW_REPOSITORY",
                "supabase" if inferred_supabase_enabled else "memory",
            ),
            supabase_url=supabase_url,
            supabase_service_role_key=supabase_key,
            database_url=os.getenv("DATABASE_URL") or None,
            admin_api_token=os.getenv("ADMIN_API_TOKEN") or None,
            backend_api_token=os.getenv("BACKEND_API_TOKEN") or None,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_environment()
