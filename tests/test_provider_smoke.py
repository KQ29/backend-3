from __future__ import annotations

import json

import pytest
from pydantic import SecretStr

from app.core.config import Settings
from app.models.domain import (
    AnalysisSource,
    AnswerAnalysis,
    Polarity,
    ProbeReason,
    ProbeStrategy,
    QuestionId,
)
from scripts import provider_smoke


def _enabled_llm_settings() -> Settings:
    return Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key=SecretStr("test-secret"),
        llm_model="test-model",
    )


def _evidence_from_output(output: str) -> dict[str, object]:
    _, separator, encoded = output.strip().partition("evidence=")
    assert separator
    return json.loads(encoded)


def test_check_llm_uses_custom_answer_and_prints_validated_evidence(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: dict[str, object] = {}

    class FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            calls["provider_kwargs"] = kwargs

        def classify(
            self,
            *,
            question_id: QuestionId,
            answer: str,
            rolling_summary: str,
            probes_remaining: int,
            previous_probe_questions: tuple[str, ...],
        ) -> AnswerAnalysis:
            calls["question_id"] = question_id
            calls["answer"] = answer
            calls["rolling_summary"] = rolling_summary
            calls["probes_remaining"] = probes_remaining
            calls["previous_probe_questions"] = previous_probe_questions
            return AnswerAnalysis(
                analysis_source=AnalysisSource.LLM,
                polarity=Polarity.POSITIVE,
                confidence=0.91,
                concrete=True,
                probe_strategy=ProbeStrategy.DRILL_DOWN,
                probe_reason=ProbeReason.OUTCOME_WITHOUT_IMPACT,
                reflection="Reporting time fell after the workflow change.",
                suggested_probe=(
                    "What did the four-hour reduction allow you to do?"
                ),
            )

    monkeypatch.setattr(
        provider_smoke,
        "OpenAICompatibleLLMProvider",
        FakeProvider,
    )
    answer = (
        "PRIVATE-INPUT-MARKER: a synthetic workflow reduced reporting time "
        "from six hours to two."
    )

    provider_smoke.check_llm(
        _enabled_llm_settings(),
        generate=True,
        answer=answer,
    )

    output = capsys.readouterr().out
    evidence = _evidence_from_output(output)
    assert calls["question_id"] == QuestionId.ANCHOR_1
    assert calls["answer"] == answer
    assert calls["rolling_summary"] == ""
    assert calls["probes_remaining"] == 2
    assert calls["previous_probe_questions"] == ()
    assert evidence == {
        "analysis_source": "llm",
        "concrete": True,
        "confidence": 0.91,
        "economic_outcome": None,
        "mixed_evidence": False,
        "needs_probe": True,
        "polarity": "positive",
        "probe_reason": "outcome_without_impact",
        "probe_type": "drill_down",
        "reflection": "Reporting time fell after the workflow change.",
        "suggested_probe": (
            "What did the four-hour reduction allow you to do?"
        ),
        "tough_or_complex": False,
    }
    assert answer not in output
    assert "PRIVATE-INPUT-MARKER" not in output
    assert "test-secret" not in output


def test_llm_evidence_escapes_terminal_control_characters(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            pass

        def classify(self, **kwargs: object) -> AnswerAnalysis:
            return AnswerAnalysis(
                analysis_source=AnalysisSource.LLM,
                reflection="The workflow\x1b[31m changed.",
            )

    monkeypatch.setattr(
        provider_smoke,
        "OpenAICompatibleLLMProvider",
        FakeProvider,
    )

    provider_smoke.check_llm(_enabled_llm_settings(), generate=True)

    output = capsys.readouterr().out
    assert "\x1b" not in output
    assert "\\u001b" in output


def test_check_llm_raises_when_provider_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            pass

        def classify(self, **kwargs: object) -> None:
            return None

    monkeypatch.setattr(
        provider_smoke,
        "OpenAICompatibleLLMProvider",
        FakeProvider,
    )

    with pytest.raises(RuntimeError, match="LLM returned no classification"):
        provider_smoke.check_llm(_enabled_llm_settings(), generate=True)


def test_main_forwards_custom_llm_answer_with_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    llm_call: dict[str, object] = {}
    monkeypatch.setattr(
        provider_smoke.Settings,
        "from_environment",
        staticmethod(lambda: settings),
    )
    monkeypatch.setattr(provider_smoke, "check_supabase", lambda value: None)

    def fake_check_llm(
        value: Settings,
        *,
        generate: bool,
        answer: str,
    ) -> None:
        llm_call.update(
            settings=value,
            generate=generate,
            answer=answer,
        )

    monkeypatch.setattr(provider_smoke, "check_llm", fake_check_llm)
    monkeypatch.setattr(
        provider_smoke,
        "check_speechmatics",
        lambda value, *, audio_path: None,
    )

    result = provider_smoke.main(
        [
            "--llm-completion",
            "--llm-answer",
            "  A synthetic answer unique to this run.  ",
        ]
    )

    assert result == 0
    assert llm_call == {
        "settings": settings,
        "generate": True,
        "answer": "A synthetic answer unique to this run.",
    }


def test_llm_answer_requires_completion(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        provider_smoke.main(["--llm-answer", "Synthetic answer"])

    assert exc_info.value.code == 2
    assert "--llm-answer requires --llm-completion" in capsys.readouterr().err


def test_legacy_llm_completion_uses_default_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings()
    llm_call: dict[str, object] = {}
    monkeypatch.setattr(
        provider_smoke.Settings,
        "from_environment",
        staticmethod(lambda: settings),
    )
    monkeypatch.setattr(provider_smoke, "check_supabase", lambda value: None)

    def fake_check_llm(
        value: Settings,
        *,
        generate: bool,
        answer: str,
    ) -> None:
        llm_call.update(generate=generate, answer=answer)

    monkeypatch.setattr(provider_smoke, "check_llm", fake_check_llm)
    monkeypatch.setattr(
        provider_smoke,
        "check_speechmatics",
        lambda value, *, audio_path: None,
    )

    assert provider_smoke.main(["--llm-completion"]) == 0
    assert llm_call == {
        "generate": True,
        "answer": provider_smoke.DEFAULT_LLM_SMOKE_ANSWER,
    }


def test_llm_answer_rejects_empty_text(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        provider_smoke.main(["--llm-completion", "--llm-answer", "   "])

    assert exc_info.value.code == 2
    assert "LLM answer cannot be empty" in capsys.readouterr().err
