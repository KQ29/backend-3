from __future__ import annotations

import argparse
import json
import mimetypes
import sys
from collections.abc import Sequence
from pathlib import Path

import httpx

from app.core.config import Settings
from app.models.domain import QuestionId
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.stt.speechmatics import SpeechmaticsBatchProvider
from app.repositories.supabase import SupabaseRestClient


DEFAULT_LLM_SMOKE_ANSWER = "My role has not changed since the training."
MAX_LLM_SMOKE_ANSWER_CHARS = 8000


def _parse_llm_answer(value: str) -> str:
    answer = value.strip()
    if not answer:
        raise argparse.ArgumentTypeError("LLM answer cannot be empty")
    if len(answer) > MAX_LLM_SMOKE_ANSWER_CHARS:
        raise argparse.ArgumentTypeError(
            f"LLM answer cannot exceed {MAX_LLM_SMOKE_ANSWER_CHARS} characters"
        )
    return answer


def check_supabase(settings: Settings) -> None:
    if not settings.supabase_enabled:
        print("Supabase: disabled")
        return
    client = SupabaseRestClient(
        project_url=settings.supabase_url or "",
        service_role_key=(
            settings.supabase_service_role_key.get_secret_value()
            if settings.supabase_service_role_key
            else ""
        ),
    )
    rows = client.request(
        "GET",
        "protocols",
        params={
            "select": "id,version,is_active",
            "is_active": "eq.true",
            "limit": "2",
        },
    )
    print(f"Supabase: connected; active protocols visible={len(rows or [])}")


def check_llm(
    settings: Settings,
    *,
    generate: bool,
    answer: str = DEFAULT_LLM_SMOKE_ANSWER,
) -> None:
    if not settings.llm_enabled:
        print("LLM: disabled")
        return
    key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else ""
    if not generate:
        response = httpx.get(
            f"{(settings.llm_base_url or '').rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=settings.llm_timeout_seconds,
        )
        response.raise_for_status()
        print(f"LLM: connected; model configured={settings.llm_model}")
        return

    provider = OpenAICompatibleLLMProvider(
        base_url=settings.llm_base_url or "",
        api_key=key,
        model=settings.llm_model or "",
        timeout_seconds=settings.llm_timeout_seconds,
        max_output_tokens=settings.llm_max_output_tokens,
    )
    result = provider.classify(
        question_id=QuestionId.ANCHOR_1,
        answer=answer,
        rolling_summary="",
        probes_remaining=settings.max_probes_per_anchor,
        previous_probe_questions=(),
    )
    if result is None:
        raise RuntimeError("LLM returned no classification")
    evidence = {
        "analysis_source": result.analysis_source.value,
        "concrete": result.concrete,
        "confidence": result.confidence,
        "economic_outcome": (
            result.economic_outcome.value if result.economic_outcome else None
        ),
        "mixed_evidence": result.mixed_evidence,
        "needs_probe": result.needs_probe,
        "polarity": result.polarity.value,
        "probe_type": result.probe_strategy.value,
        "probe_reason": (
            result.probe_reason.value if result.probe_reason else None
        ),
        "reflection": result.reflection,
        "suggested_probe": result.suggested_probe,
        "tough_or_complex": result.tough_or_complex,
    }
    print(
        "LLM: generated a valid classification; "
        f"polarity={result.polarity.value}; "
        "evidence="
        f"{json.dumps(evidence, ensure_ascii=True, sort_keys=True)}"
    )


def check_speechmatics(
    settings: Settings,
    *,
    audio_path: Path | None,
) -> None:
    if settings.stt_provider != "speechmatics":
        print("Speechmatics: disabled")
        return
    key = settings.stt_api_key.get_secret_value() if settings.stt_api_key else ""
    if audio_path is None:
        jobs_base = (settings.stt_base_url or "").rstrip("/")
        if not jobs_base.endswith("/jobs"):
            jobs_base = f"{jobs_base}/jobs"
        response = httpx.get(
            f"{jobs_base}/",
            headers={"Authorization": f"Bearer {key}"},
            params={"limit": 1},
            timeout=settings.stt_timeout_seconds,
        )
        response.raise_for_status()
        print("Speechmatics: connected; no transcription job created")
        return

    audio = audio_path.read_bytes()
    if not audio:
        raise ValueError("Audio file is empty")
    provider = SpeechmaticsBatchProvider(
        api_key=key,
        base_url=settings.stt_base_url or "",
        timeout_seconds=settings.stt_timeout_seconds,
        default_language=settings.stt_language,
    )
    result = provider.transcribe(
        audio=audio,
        filename=audio_path.name,
        mime_type=mimetypes.guess_type(audio_path.name)[0] or "audio/ogg",
    )
    print(
        "Speechmatics: transcription completed; "
        f"characters={len(result.text)}, confidence={result.confidence}"
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Safely verify configured external providers.",
    )
    parser.add_argument(
        "--llm-completion",
        action="store_true",
        help="Make one small, potentially billable LLM classification call.",
    )
    parser.add_argument(
        "--llm-answer",
        type=_parse_llm_answer,
        metavar="TEXT",
        help=(
            "Classify this synthetic, non-sensitive answer; requires "
            "--llm-completion and may remain in shell history."
        ),
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Create a potentially billable Speechmatics job for this audio file.",
    )
    args = parser.parse_args(argv)
    if args.llm_answer is not None and not args.llm_completion:
        parser.error("--llm-answer requires --llm-completion")

    try:
        settings = Settings.from_environment()
        check_supabase(settings)
        check_llm(
            settings,
            generate=args.llm_completion,
            answer=args.llm_answer or DEFAULT_LLM_SMOKE_ANSWER,
        )
        check_speechmatics(settings, audio_path=args.audio)
    except (httpx.HTTPError, OSError, RuntimeError, ValueError) as exc:
        print(
            f"Provider smoke check failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
