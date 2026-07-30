from __future__ import annotations

import pytest

from app.core.config import Settings
from app.interview.classifier import classify_answer
from app.interview.swahili import (
    VoiceCue,
    choose_voice_cue,
    count_swahili_expressions,
)
from app.models.domain import (
    AnswerAnalysis,
    EconomicOutcome,
    Polarity,
    ProbeReason,
    ProbeStrategy,
    QuestionId,
)


def test_concrete_positive_milestone_is_detected() -> None:
    analysis = classify_answer(
        "I got a promotion in March 2026 after the AI training.",
        QuestionId.ANCHOR_1,
    )

    assert analysis.polarity == Polarity.POSITIVE
    assert analysis.concrete is True
    assert analysis.positive_milestone is True
    assert analysis.economic_outcome == "role_change_no_pay_change"


def test_no_change_and_mixed_evidence_are_distinguished() -> None:
    no_change = classify_answer(
        "My income and job role have not changed since the training.",
        QuestionId.ANCHOR_1,
    )
    mixed = classify_answer(
        "I won a new client last month, but unreliable internet limits my AI work.",
        QuestionId.ANCHOR_1,
    )

    assert no_change.polarity == Polarity.NEGATIVE
    assert no_change.mixed_evidence is False
    assert mixed.polarity == Polarity.POSITIVE
    assert mixed.mixed_evidence is True


def test_local_classifier_requests_clarity_only_for_an_unclear_anchor_answer() -> None:
    analysis = classify_answer(
        "I use AI for emails.",
        QuestionId.ANCHOR_1,
    )

    assert analysis.needs_probe is True
    assert analysis.probe_strategy == ProbeStrategy.CLARITY
    assert analysis.probe_reason == ProbeReason.VAGUE_OR_UNCLEAR


def test_local_classifier_drills_down_on_an_outcome_without_impact() -> None:
    analysis = classify_answer(
        "AI saved time and now saves me two hours each day.",
        QuestionId.ANCHOR_1,
    )

    assert analysis.needs_probe is True
    assert analysis.probe_strategy == ProbeStrategy.DRILL_DOWN
    assert analysis.probe_reason == ProbeReason.OUTCOME_WITHOUT_IMPACT


def test_local_classifier_prioritizes_tension_for_mixed_evidence() -> None:
    analysis = classify_answer(
        "I got a new client, but access to paid AI tools is still blocked.",
        QuestionId.ANCHOR_1,
    )

    assert analysis.mixed_evidence is True
    assert analysis.needs_probe is True
    assert analysis.probe_strategy == ProbeStrategy.TENSION
    assert analysis.probe_reason == ProbeReason.MIXED_OR_CONFLICTING


def test_local_classifier_does_not_probe_a_complete_negative_answer() -> None:
    analysis = classify_answer(
        (
            "No, I have not shared the training with colleagues or changed "
            "our team workflow."
        ),
        QuestionId.ANCHOR_4,
    )

    assert analysis.needs_probe is False
    assert analysis.probe_strategy == ProbeStrategy.NONE
    assert analysis.probe_reason is None


def test_local_classifier_can_request_a_second_probe_for_an_unclear_probe_answer() -> None:
    analysis = classify_answer(
        "It helps with work.",
        QuestionId.ANCHOR_1_PROBE,
    )

    assert analysis.needs_probe is True
    assert analysis.probe_strategy == ProbeStrategy.CLARITY
    assert analysis.probe_reason == ProbeReason.VAGUE_OR_UNCLEAR


def test_local_classifier_stops_probing_after_a_sufficient_probe_answer() -> None:
    analysis = classify_answer(
        "Yesterday AI drafted a customer email that I reviewed before sending.",
        QuestionId.ANCHOR_1_PROBE,
    )

    assert analysis.needs_probe is False
    assert analysis.probe_strategy == ProbeStrategy.NONE
    assert analysis.probe_reason is None


@pytest.mark.parametrize(
    "answer",
    (
        "I was fired and now I am poor.",
        "I lost my job and my income fell.",
        "I was laid off after the training.",
    ),
)
def test_job_or_income_loss_is_on_topic_negative_and_tough(answer: str) -> None:
    analysis = classify_answer(answer, QuestionId.ANCHOR_1)

    assert analysis.on_topic is True
    assert analysis.polarity == Polarity.NEGATIVE
    assert analysis.concrete is True
    assert analysis.tough_or_complex is True
    assert (
        analysis.economic_outcome
        == EconomicOutcome.INCOME_DECREASE_OR_JOB_LOSS
    )


def test_negative_voice_cues_never_fall_through_to_sawa() -> None:
    hardship = AnswerAnalysis(
        polarity=Polarity.NEGATIVE,
        economic_outcome=EconomicOutcome.INCOME_DECREASE_OR_JOB_LOSS,
        reflection="You said you were fired.",
    )
    grounded_barrier = AnswerAnalysis(
        polarity=Polarity.NEGATIVE,
        reflection="You identified access to tools as the barrier.",
    )
    fallback_barrier = AnswerAnalysis(polarity=Polarity.NEGATIVE)

    assert choose_voice_cue(hardship, moving_topic=True) == VoiceCue.POLE
    assert choose_voice_cue(grounded_barrier, moving_topic=True) == VoiceCue.NONE
    assert choose_voice_cue(fallback_barrier, moving_topic=True) == VoiceCue.NAAM


def test_apology_skip_and_off_topic_signals_are_bounded() -> None:
    apology = classify_answer(
        "Sorry for the long message, I prefer not to answer this one.",
        QuestionId.ANCHOR_2,
    )
    off_topic = classify_answer(
        "I want to discuss my favourite movie instead.",
        QuestionId.ANCHOR_3,
    )

    assert apology.apology is True
    assert apology.skip_requested is True
    assert off_topic.on_topic is False


@pytest.mark.parametrize(
    "message",
    (
        "Karibu. Welcome to the interview.",
        "Safi sana! Tell me more.",
        "Sawa. Let us continue.",
        "Naam. That sounds difficult.",
        "Usijali! We can skip it.",
        "Pole. You said you were laid off.",
        "Asante sana. That is everything.",
    ),
)
def test_localized_messages_have_one_expression(message: str) -> None:
    assert count_swahili_expressions(message) == 1


def test_live_provider_settings_fail_closed() -> None:
    with pytest.raises(ValueError, match="LLM_BASE_URL"):
        Settings(llm_enabled=True)

    with pytest.raises(ValueError, match="SPEECHMATICS_API_KEY"):
        Settings(stt_provider="speechmatics")

    with pytest.raises(ValueError, match="SUPABASE_URL"):
        Settings(repository_backend="supabase")
