from __future__ import annotations

import pytest

from app.core.config import Settings
from app.interview.engine import InterviewEngine, InvalidInterviewAction
from app.interview.swahili import count_swahili_expressions
from app.models.domain import (
    AnalysisSource,
    AnswerAnalysis,
    EconomicOutcome,
    InputMode,
    InterviewState,
    Polarity,
    ProbeReason,
    ProbeStrategy,
    QuestionId,
    SessionStatus,
)
from app.repositories.memory import MemoryInterviewRepository
from app.services.interviews import InterviewService
from tests.conftest import complete_demographics, start_consented_interview


def test_declined_consent_stores_no_response_or_follow_up(
    service: InterviewService,
) -> None:
    started = service.start_interview()
    declined = service.submit_consent(started.session_id, "consent_no")
    state = service.get_state(started.session_id)

    assert declined.status == SessionStatus.DECLINED
    assert len(state.transcript) == 1
    assert state.transcript[0].question_id == QuestionId.CONSENT
    assert state.tags == []


def test_invalid_consent_is_rejected_without_state_change(
    service: InterviewService,
) -> None:
    started = service.start_interview()

    with pytest.raises(InvalidInterviewAction):
        service.submit_consent(started.session_id, "maybe")

    state = service.get_state(started.session_id)
    assert state.status == SessionStatus.AWAITING_CONSENT
    assert len(state.transcript) == 1


def test_demographics_are_sequential_and_skippable(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    answers = (
        "Kamau Otieno",
        "skip",
        "31",
        "Male",
        "Nairobi",
        "Westlands",
        "Digital marketing officer",
    )
    response = None
    for answer in answers:
        response = service.submit_text(session_id, answer)

    record = service.export_record(session_id)
    state = service.get_state(session_id)
    assert response is not None
    assert response.question_id == QuestionId.ANCHOR_1
    assert response.message.startswith("Sawa.")
    assert state.status == SessionStatus.IN_PROGRESS
    assert record["state"]["demographics"]["email"] == "Prefer not to say"
    assert state.mode_of_input is None
    assert state.tags == []


def test_specific_positive_path_skips_probe_and_uses_benefit_branch(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    first = service.submit_text(
        session_id,
        "I got a promotion in March 2026 after using AI to improve our reports.",
    )
    assert first.question_id == QuestionId.ANCHOR_3
    assert first.message.startswith("Safi sana!")
    assert first.needs_probe is False
    assert first.probes_used == []


def test_no_change_path_goes_to_bottleneck(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    response = service.submit_text(
        session_id,
        "My income and job role have not changed since the AI training.",
    )

    assert response.question_id == QuestionId.ANCHOR_2
    assert response.anchors_covered == [QuestionId.ANCHOR_1]
    assert response.mixed_evidence is False


def test_mixed_path_covers_bottleneck_then_benefit(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    mixed = service.submit_text(
        session_id,
        "I won a new client last month, but unreliable internet limits my AI work.",
    )
    second_tension = service.submit_text(
        session_id,
        "I got a new client, but internet access is unreliable.",
    )
    pending = service.get_state(session_id)
    pending_record = service.export_record(session_id)
    after_tension = service.submit_text(
        session_id,
        "The client creates income, while unreliable internet makes delivery harder.",
    )
    after_bottleneck = service.submit_text(
        session_id,
        "In March 2026 my manager rejected a project request because our team "
        "did not have licensed AI software.",
    )

    assert mixed.question_id == QuestionId.ANCHOR_1_PROBE
    assert mixed.mixed_evidence is True
    assert mixed.message.startswith("Naam.")
    assert mixed.needs_probe is True
    assert mixed.probe_type == ProbeStrategy.TENSION
    assert mixed.probe_reason == ProbeReason.MIXED_OR_CONFLICTING
    assert mixed.probe_asked is True
    assert mixed.probe_number == 1
    assert second_tension.question_id == QuestionId.ANCHOR_1_PROBE
    assert second_tension.probe_asked is True
    assert second_tension.probe_number == 2
    assert pending.question_id == QuestionId.ANCHOR_1_PROBE
    assert pending.probe_counts == {"anchor_1": 2}
    assert pending_record["state"]["resume_question_id"] == "anchor_2"
    assert len(pending_record["state"]["probe_questions"]["anchor_1"]) == 2
    assert len(set(pending_record["state"]["probe_questions"]["anchor_1"])) == 2
    assert after_tension.question_id == QuestionId.ANCHOR_2
    assert after_bottleneck.question_id == QuestionId.ANCHOR_3
    state = service.get_state(session_id)
    final_record = service.export_record(session_id)
    assert state.branch_history == ["anchor_1:mixed"]
    assert state.probes_used.count(QuestionId.ANCHOR_1_PROBE) == 2
    assert state.probe_counts == {"anchor_1": 2}
    assert final_record["state"]["resume_question_id"] is None
    assert QuestionId.ANCHOR_2 in state.anchors_covered


def test_sufficient_first_probe_answer_resumes_without_a_second_probe(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)
    service.submit_text(
        session_id,
        "My income and job role have not changed since the AI training.",
    )

    probe = service.submit_text(session_id, "The tools are a problem.")
    after_probe = service.submit_text(
        session_id,
        "Last month a client project failed because I lacked software access.",
    )

    assert probe.question_id == QuestionId.ANCHOR_2_PROBE
    assert probe.probe_number == 1
    assert after_probe.question_id == QuestionId.ANCHOR_4
    assert after_probe.needs_probe is False
    assert after_probe.probe_asked is False
    assert after_probe.probes_used.count(QuestionId.ANCHOR_2_PROBE) == 1
    assert after_probe.probe_counts == {"anchor_2": 1}


def test_mode_is_based_only_on_substantive_answers(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    for answer in (
        "Kamau Otieno",
        "kamau.demo@example.com",
        "31",
        "Male",
        "Nairobi",
        "Westlands",
        "Digital marketing officer",
    ):
        service.submit_voice(session_id, answer, 0.95)

    assert service.get_state(session_id).mode_of_input is None
    service.submit_text(
        session_id,
        "My income and job role have not changed since the AI training.",
    )
    assert service.get_state(session_id).mode_of_input == InputMode.TEXT
    service.submit_voice(
        session_id,
        "Employer approval and software access are the main barriers at work.",
        0.91,
    )
    assert service.get_state(session_id).mode_of_input == InputMode.MIXED


def test_stop_is_deterministic_and_not_stored_as_an_answer(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)
    before = service.get_state(session_id)

    stopped = service.submit_text(session_id, "stop")
    after = service.get_state(session_id)

    assert stopped.status == SessionStatus.STOPPED
    assert after.status == SessionStatus.STOPPED
    assert len(after.transcript) == len(before.transcript) + 1
    assert after.transcript[-1].role.value == "assistant"
    assert all(turn.content.lower() != "stop" for turn in after.transcript)


def test_apology_or_skip_uses_reassurance_cue(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    response = service.submit_text(
        session_id,
        "Sorry for the long message, I prefer not to answer this one.",
    )

    assert response.question_id == QuestionId.ANCHOR_2
    assert response.message.startswith("Usijali!")


def test_off_topic_answer_is_tagged_once_and_reasks(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    response = service.submit_text(
        session_id,
        "I would rather discuss my favourite movie today.",
    )
    state = service.get_state(session_id)

    assert response.question_id == QuestionId.ANCHOR_1
    assert "return to the interview" in response.message
    assert len(state.tags) == 1
    assert state.tags[0].on_topic is False
    assert state.tags[0].turn_id == state.transcript[-2].id


def test_catch_all_is_logged_verbatim_without_force_coding(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)
    service.submit_text(
        session_id,
        "My income and job role have not changed since the AI training.",
    )
    service.submit_text(
        session_id,
        "In March 2026 my manager rejected a project request because we lacked tools.",
    )
    service.submit_text(
        session_id,
        "No, I have not shared the AI training with colleagues or my team.",
    )
    catch_text = "Paid tool access and one month of practical mentoring."
    response = service.submit_text(session_id, catch_text)
    state = service.get_state(session_id)

    assert response.question_id == QuestionId.WRAP_UP
    tag = next(tag for tag in state.tags if tag.question_id == QuestionId.CATCH_ALL)
    assert tag.raw_response == catch_text
    assert tag.force_coded is False
    assert tag.polarity is None
    assert tag.economic_outcome is None


def test_every_assistant_reply_uses_at_most_one_swahili_expression(
    service: InterviewService,
) -> None:
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)
    service.submit_text(
        session_id,
        "I got a promotion in March 2026 after using AI at work.",
    )
    service.submit_text(
        session_id,
        "I automated reports and saved time in my current role.",
    )
    service.submit_text(
        session_id,
        "Yes, I shared the training with colleagues and our team now works faster.",
    )
    service.submit_text(
        session_id,
        "Access to paid tools and practical employer projects.",
    )
    service.submit_text(session_id, "Nothing else to add.")

    assistants = [
        turn
        for turn in service.get_state(session_id).transcript
        if turn.role.value == "assistant"
    ]
    assert assistants
    assert all(count_swahili_expressions(turn.content) <= 1 for turn in assistants)
    assert assistants[-1].content.startswith("Asante sana.")
    assert service.get_state(session_id).status == SessionStatus.COMPLETED


def test_low_confidence_llm_fallback_is_capped_at_three_calls() -> None:
    class CountingProvider:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def enabled(self) -> bool:
            return True

        def classify(
            self,
            *,
            question_id: QuestionId,
            answer: str,
            rolling_summary: str,
            probes_remaining: int = 1,
            previous_probe_questions: tuple[str, ...] = (),
        ) -> AnswerAnalysis | None:
            del (
                question_id,
                answer,
                rolling_summary,
                probes_remaining,
                previous_probe_questions,
            )
            self.calls += 1
            return None

    provider = CountingProvider()
    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="fallback",
        llm_max_calls_per_session=3,
        llm_low_confidence_threshold=0.99,
    )
    service = InterviewService(
        MemoryInterviewRepository(),
        InterviewEngine(settings, provider),
    )
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    for _ in range(5):
        service.submit_text(session_id, "Maybe.")

    state = service.get_state(session_id)
    assert provider.calls == 3
    assert state.llm_calls_used == 3


def test_llm_always_mode_overrides_high_confidence_local_routing() -> None:
    class NoChangeProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(
            self,
            *,
            question_id: QuestionId,
            answer: str,
            rolling_summary: str,
            probes_remaining: int = 1,
            previous_probe_questions: tuple[str, ...] = (),
        ) -> AnswerAnalysis:
            del (
                question_id,
                answer,
                rolling_summary,
                probes_remaining,
                previous_probe_questions,
            )
            return AnswerAnalysis(
                polarity=Polarity.NEGATIVE,
                confidence=0.94,
                vague=False,
                concrete=True,
                benefit_named=False,
                on_topic=True,
                economic_outcome="no_change",
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
        llm_max_calls_per_session=16,
    )
    service = InterviewService(
        MemoryInterviewRepository(),
        InterviewEngine(settings, NoChangeProvider()),
    )
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    response = service.submit_text(
        session_id,
        "I got a promotion in March 2026 after using AI to improve our reports.",
    )
    state = service.get_state(session_id)

    assert response.question_id == QuestionId.ANCHOR_2
    assert response.analysis_source == AnalysisSource.LLM
    assert state.tags[-1].metadata["analysis_source"] == "llm"
    assert state.llm_calls_used == 1


def test_llm_no_probe_decision_is_authoritative_for_complete_answer() -> None:
    class CompleteAnswerProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(self, **kwargs) -> AnswerAnalysis:
            del kwargs
            return AnswerAnalysis(
                polarity=Polarity.POSITIVE,
                confidence=0.96,
                concrete=True,
                benefit_named=True,
                on_topic=True,
                economic_outcome="role_change_no_pay_change",
                reflection="You reported getting a promotion.",
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
    )
    service = InterviewService(
        MemoryInterviewRepository(),
        InterviewEngine(settings, CompleteAnswerProvider()),
    )
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    response = service.submit_text(session_id, "I got a promotion.")

    assert response.question_id == QuestionId.ANCHOR_3
    assert response.needs_probe is False
    assert response.probe_type == ProbeStrategy.NONE
    assert response.probe_asked is False
    assert response.probes_used == []


def test_llm_dynamic_probe_and_reflection_are_used_inside_protocol_bounds() -> None:
    class AdaptiveProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(
            self,
            *,
            question_id: QuestionId,
            answer: str,
            rolling_summary: str,
            probes_remaining: int = 1,
            previous_probe_questions: tuple[str, ...] = (),
        ) -> AnswerAnalysis:
            del (
                question_id,
                answer,
                rolling_summary,
                probes_remaining,
                previous_probe_questions,
            )
            return AnswerAnalysis(
                polarity=Polarity.POSITIVE,
                confidence=0.96,
                vague=False,
                concrete=True,
                benefit_named=True,
                on_topic=True,
                economic_outcome="role_change_no_pay_change",
                probe_strategy=ProbeStrategy.DRILL_DOWN,
                probe_reason=ProbeReason.OUTCOME_WITHOUT_IMPACT,
                reflection="You linked the promotion to using AI in your reports.",
                suggested_probe=(
                    "What did the promotion allow you to do differently at work?"
                ),
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
        llm_max_calls_per_session=16,
    )
    service = InterviewService(
        MemoryInterviewRepository(),
        InterviewEngine(settings, AdaptiveProvider()),
    )
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    response = service.submit_text(
        session_id,
        "I got a promotion after using AI for weekly reports.",
    )

    assert response.question_id == QuestionId.ANCHOR_1_PROBE
    assert response.probe_strategy == ProbeStrategy.DRILL_DOWN
    assert response.probe_type == ProbeStrategy.DRILL_DOWN
    assert response.probe_reason == ProbeReason.OUTCOME_WITHOUT_IMPACT
    assert response.needs_probe is True
    assert response.probe_asked is True
    assert response.message == (
        "You linked the promotion to using AI in your reports. "
        "What did the promotion allow you to do differently at work?"
    )
    assert response.probes_used == [QuestionId.ANCHOR_1_PROBE]


def test_two_probes_are_allowed_but_a_third_request_is_capped() -> None:
    class AlwaysProbeProvider:
        def __init__(self) -> None:
            self.contexts: list[tuple[int, tuple[str, ...]]] = []

        @property
        def enabled(self) -> bool:
            return True

        def classify(
            self,
            *,
            question_id: QuestionId,
            answer: str,
            rolling_summary: str,
            probes_remaining: int = 1,
            previous_probe_questions: tuple[str, ...] = (),
        ) -> AnswerAnalysis:
            del question_id, rolling_summary
            self.contexts.append((probes_remaining, previous_probe_questions))
            suggested_probe = {
                2: "When did the tools most recently block your work?",
                1: "What effect did the tools have on that client task?",
                0: "Which tools blocked that reporting task?",
            }[probes_remaining]
            return AnswerAnalysis(
                polarity=Polarity.NEGATIVE,
                confidence=0.94,
                vague=True,
                barrier_named=True,
                on_topic=True,
                probe_strategy=ProbeStrategy.CLARITY,
                probe_reason=ProbeReason.VAGUE_OR_UNCLEAR,
                reflection=f"The tools affected {answer.rstrip('.').lower()}.",
                suggested_probe=suggested_probe,
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
    )
    provider = AlwaysProbeProvider()
    engine = InterviewEngine(settings, provider)
    state = InterviewState(
        status=SessionStatus.IN_PROGRESS,
        consent_given=True,
        current_question_id=QuestionId.ANCHOR_2,
        max_probes_per_anchor=2,
    )

    first = engine.handle_answer(state, "Tools block my work.", InputMode.TEXT)
    second = engine.handle_answer(
        state,
        "The tools blocked a client task.",
        InputMode.TEXT,
    )
    resume_after_second = state.resume_question_id
    third = engine.handle_answer(
        state,
        "The tools blocked that reporting task.",
        InputMode.TEXT,
    )

    assert first.next_question_id == QuestionId.ANCHOR_2_PROBE
    assert second.next_question_id == QuestionId.ANCHOR_2_PROBE
    assert third.next_question_id == QuestionId.ANCHOR_4
    assert first.message != second.message
    assert resume_after_second == QuestionId.ANCHOR_4
    assert state.probes_used == [
        QuestionId.ANCHOR_2_PROBE,
        QuestionId.ANCHOR_2_PROBE,
    ]
    assert state.probe_counts == {"anchor_2": 2}
    assert len(state.probe_questions["anchor_2"]) == 2
    assert len(set(state.probe_questions["anchor_2"])) == 2
    assert state.resume_question_id is None
    assert first.tag is not None
    assert second.tag is not None
    assert third.tag is not None
    assert first.tag.metadata["probe_number"] == 1
    assert second.tag.metadata["probe_number"] == 2
    assert second.tag.metadata["probe_asked"] is True
    assert third.tag.metadata["needs_probe"] is True
    assert third.tag.metadata["probe_asked"] is False
    assert third.tag.metadata["probe_number"] is None
    assert provider.contexts == [
        (2, ()),
        (1, (state.probe_questions["anchor_2"][0],)),
        (0, tuple(state.probe_questions["anchor_2"])),
    ]


def test_off_topic_first_probe_answer_does_not_consume_a_second_probe() -> None:
    engine = InterviewEngine(Settings())
    state = InterviewState(
        status=SessionStatus.IN_PROGRESS,
        consent_given=True,
        current_question_id=QuestionId.ANCHOR_2,
        max_probes_per_anchor=2,
    )

    first = engine.handle_answer(state, "Tools block my work.", InputMode.TEXT)
    second = engine.handle_answer(
        state,
        "I would rather discuss my favourite movie today.",
        InputMode.TEXT,
    )

    assert first.next_question_id == QuestionId.ANCHOR_2_PROBE
    assert second.next_question_id == QuestionId.ANCHOR_4
    assert second.tag is not None
    assert second.tag.on_topic is False
    assert state.probes_used == [QuestionId.ANCHOR_2_PROBE]
    assert state.probe_counts == {"anchor_2": 1}


def test_repeated_dynamic_probe_uses_a_distinct_second_fallback() -> None:
    class RepeatingProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(self, **kwargs) -> AnswerAnalysis:
            answer = str(kwargs["answer"])
            return AnswerAnalysis(
                polarity=Polarity.NEGATIVE,
                confidence=0.94,
                vague=True,
                barrier_named=True,
                on_topic=True,
                probe_strategy=ProbeStrategy.CLARITY,
                probe_reason=ProbeReason.VAGUE_OR_UNCLEAR,
                reflection=f"The tools affected {answer.rstrip('.').lower()}.",
                suggested_probe="When did the tools most recently block your work?",
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
        max_probes_per_anchor=2,
    )
    state = InterviewState(
        status=SessionStatus.IN_PROGRESS,
        consent_given=True,
        current_question_id=QuestionId.ANCHOR_2,
        max_probes_per_anchor=2,
    )
    engine = InterviewEngine(settings, RepeatingProvider())

    first = engine.handle_answer(state, "Tools block my work.", InputMode.TEXT)
    second = engine.handle_answer(
        state,
        "The tools still block client work.",
        InputMode.TEXT,
    )

    assert first.next_question_id == QuestionId.ANCHOR_2_PROBE
    assert second.next_question_id == QuestionId.ANCHOR_2_PROBE
    assert state.probe_questions["anchor_2"] == [
        "When did the tools most recently block your work?",
        "What was the result in that specific example?",
    ]
    assert first.message != second.message
    assert second.tag is not None
    assert second.tag.metadata["llm_suggested_probe"] is None
    assert second.tag.metadata["probe_number"] == 2


@pytest.mark.parametrize("maximum", (0, 1, 2))
def test_per_session_probe_limit_is_enforced(maximum: int) -> None:
    settings = Settings(max_probes_per_anchor=maximum)
    engine = InterviewEngine(settings)
    state = engine.initial_state()
    state.status = SessionStatus.IN_PROGRESS
    state.consent_given = True
    state.current_question_id = QuestionId.ANCHOR_2

    results = [
        engine.handle_answer(state, "Tools block my work.", InputMode.TEXT)
        for _ in range(maximum + 1)
    ]

    assert state.max_probes_per_anchor == maximum
    assert state.probe_counts.get("anchor_2", 0) == maximum
    assert state.probes_used.count(QuestionId.ANCHOR_2_PROBE) == maximum
    assert all(
        result.next_question_id == QuestionId.ANCHOR_2_PROBE
        for result in results[:maximum]
    )
    assert results[-1].next_question_id == QuestionId.ANCHOR_4
    assert results[-1].tag is not None
    assert results[-1].tag.metadata["probe_asked"] is False


def test_engine_drops_generic_llm_language_and_keeps_canonical_probe() -> None:
    class GenericProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(self, **kwargs) -> AnswerAnalysis:
            del kwargs
            return AnswerAnalysis(
                polarity=Polarity.POSITIVE,
                confidence=0.9,
                concrete=True,
                benefit_named=True,
                on_topic=True,
                economic_outcome="role_change_no_pay_change",
                probe_strategy=ProbeStrategy.CLARITY,
                probe_reason=ProbeReason.VAGUE_OR_UNCLEAR,
                reflection="You described your experience.",
                suggested_probe="Can you tell me more?",
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
    )
    service = InterviewService(
        MemoryInterviewRepository(),
        InterviewEngine(settings, GenericProvider()),
    )
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    response = service.submit_text(
        session_id,
        "I earned a promotion after automating the weekly reporting workflow.",
    )

    assert response.analysis_source == AnalysisSource.LLM
    assert response.question_id == QuestionId.ANCHOR_1_PROBE
    assert response.message == (
        "Can you give one recent, specific example of what you mean?"
    )
    metadata = service.get_state(session_id).tags[-1].metadata
    assert metadata["llm_reflection"] is None
    assert metadata["llm_suggested_probe"] is None


def test_job_loss_gets_grounded_hardship_response_without_sawa() -> None:
    class HardshipProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(self, **kwargs) -> AnswerAnalysis:
            del kwargs
            return AnswerAnalysis(
                polarity=Polarity.NEUTRAL,
                confidence=0.95,
                concrete=True,
                on_topic=True,
                tough_or_complex=False,
                economic_outcome="too_early_to_tell",
                reflection=(
                    "You said you were fired and are now facing financial difficulty."
                ),
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
    )
    engine = InterviewEngine(settings, HardshipProvider())
    state = InterviewState(
        status=SessionStatus.IN_PROGRESS,
        consent_given=True,
        current_question_id=QuestionId.ANCHOR_1,
    )

    result = engine.handle_answer(
        state,
        "I was fired and now I am poor.",
        InputMode.TEXT,
    )

    assert result.next_question_id == QuestionId.ANCHOR_2
    assert result.message == (
        "Pole. You said you were fired and are now facing financial difficulty. "
        "What, if anything, is currently getting in the way of using what you "
        "learned for work or income - opportunities, employer support, confidence, "
        "access to tools, or something else?"
    )
    assert "Sawa." not in result.message
    assert result.tag is not None
    assert result.tag.polarity == Polarity.NEGATIVE
    assert (
        result.tag.economic_outcome
        == EconomicOutcome.INCOME_DECREASE_OR_JOB_LOSS
    )


def test_tool_barrier_gets_specific_reflection_and_probe_without_stock_cue() -> None:
    class ToolBarrierProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(self, **kwargs) -> AnswerAnalysis:
            del kwargs
            return AnswerAnalysis(
                polarity=Polarity.NEGATIVE,
                confidence=0.94,
                vague=True,
                barrier_named=True,
                on_topic=True,
                probe_strategy=ProbeStrategy.CLARITY,
                probe_reason=ProbeReason.VAGUE_OR_UNCLEAR,
                reflection="You identified the tools as the main barrier.",
                suggested_probe=(
                    "Can you describe one recent time when the tools got in the way?"
                ),
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
    )
    engine = InterviewEngine(settings, ToolBarrierProvider())
    state = InterviewState(
        status=SessionStatus.IN_PROGRESS,
        consent_given=True,
        current_question_id=QuestionId.ANCHOR_2,
    )

    result = engine.handle_answer(
        state,
        "It is because of tools.",
        InputMode.TEXT,
    )

    assert result.next_question_id == QuestionId.ANCHOR_2_PROBE
    assert result.message == (
        "You identified the tools as the main barrier. "
        "Can you describe one recent time when the tools got in the way?"
    )


@pytest.mark.parametrize(
    "answer",
    ("Yes", "Yes, I do.", "I do", "Yes there is!"),
)
def test_bare_wrap_up_affirmative_requests_elaboration_once(answer: str) -> None:
    engine = InterviewEngine(Settings())
    state = InterviewState(
        status=SessionStatus.IN_PROGRESS,
        consent_given=True,
        current_question_id=QuestionId.WRAP_UP,
    )

    first = engine.handle_answer(state, answer, InputMode.TEXT)
    second = engine.handle_answer(state, "Yes", InputMode.TEXT)

    assert first.response_type.value == "open_text"
    assert first.next_question_id == QuestionId.WRAP_UP
    assert first.message == "Please go ahead - what would you like to add?"
    assert state.wrap_up_elaboration_requested is True
    assert second.response_type.value == "complete"
    assert state.status == SessionStatus.COMPLETED


@pytest.mark.parametrize("answer", ("No", "Nothing else", "No, that is all."))
def test_wrap_up_negative_or_substantive_answer_completes(answer: str) -> None:
    engine = InterviewEngine(Settings())
    state = InterviewState(
        status=SessionStatus.IN_PROGRESS,
        consent_given=True,
        current_question_id=QuestionId.WRAP_UP,
    )

    result = engine.handle_answer(state, answer, InputMode.TEXT)

    assert result.response_type.value == "complete"
    assert state.status == SessionStatus.COMPLETED


def test_wrap_up_elaboration_is_reflected_before_closing() -> None:
    class ClosingProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(self, **kwargs) -> AnswerAnalysis:
            del kwargs
            return AnswerAnalysis(
                polarity=Polarity.NEUTRAL,
                confidence=0.92,
                concrete=True,
                on_topic=True,
                reflection="More practical tool guidance would have helped.",
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
    )
    engine = InterviewEngine(settings, ClosingProvider())
    state = InterviewState(
        status=SessionStatus.IN_PROGRESS,
        consent_given=True,
        current_question_id=QuestionId.WRAP_UP,
    )

    prompt = engine.handle_answer(state, "Yes", InputMode.TEXT)
    result = engine.handle_answer(
        state,
        "More practical tool guidance would have helped.",
        InputMode.TEXT,
    )

    assert prompt.message == "Please go ahead - what would you like to add?"
    assert result.response_type.value == "complete"
    assert result.message.startswith(
        "Asante sana. More practical tool guidance would have helped."
    )


def test_skip_at_wrap_up_completes_instead_of_returning_open_close() -> None:
    engine = InterviewEngine(Settings())
    state = InterviewState(
        status=SessionStatus.IN_PROGRESS,
        consent_given=True,
        current_question_id=QuestionId.WRAP_UP,
    )

    result = engine.handle_answer(state, "skip", InputMode.TEXT)

    assert result.response_type.value == "complete"
    assert result.next_question_id is None
    assert state.status == SessionStatus.COMPLETED


def test_reported_hardship_scenario_stays_specific_through_wrap_up() -> None:
    class ScenarioProvider:
        def __init__(self) -> None:
            self.calls = 0

        @property
        def enabled(self) -> bool:
            return True

        def classify(
            self,
            *,
            question_id: QuestionId,
            answer: str,
            rolling_summary: str,
            probes_remaining: int = 1,
            previous_probe_questions: tuple[str, ...] = (),
        ) -> AnswerAnalysis:
            del rolling_summary, probes_remaining, previous_probe_questions
            self.calls += 1
            responses = {
                QuestionId.ANCHOR_1: AnswerAnalysis(
                    polarity=Polarity.NEGATIVE,
                    confidence=0.98,
                    concrete=True,
                    on_topic=True,
                    tough_or_complex=True,
                    economic_outcome="income_decrease_or_job_loss",
                    reflection=(
                        "You said you were fired and are now facing financial "
                        "difficulty."
                    ),
                ),
                QuestionId.ANCHOR_2: AnswerAnalysis(
                    polarity=Polarity.NEGATIVE,
                    confidence=0.94,
                    vague=True,
                    barrier_named=True,
                    on_topic=True,
                    probe_strategy=ProbeStrategy.CLARITY,
                    probe_reason=ProbeReason.VAGUE_OR_UNCLEAR,
                    reflection="You identified the tools as the main barrier.",
                    suggested_probe=(
                        "Can you describe one recent time when the tools got in "
                        "the way?"
                    ),
                ),
                QuestionId.ANCHOR_2_PROBE: AnswerAnalysis(
                    polarity=Polarity.NEGATIVE,
                    confidence=0.95,
                    concrete=True,
                    barrier_named=True,
                    on_topic=True,
                    reflection=(
                        "A client task was delayed because the tools were unavailable."
                    ),
                ),
                QuestionId.ANCHOR_4: AnswerAnalysis(
                    polarity=Polarity.NEGATIVE,
                    confidence=0.9,
                    vague=True,
                    affirmative=True,
                    on_topic=True,
                    probe_strategy=ProbeStrategy.CLARITY,
                    probe_reason=ProbeReason.VAGUE_OR_UNCLEAR,
                    reflection="Nothing improved after you shared the training.",
                    suggested_probe=(
                        "What did you share, and what showed that nothing helped?"
                    ),
                ),
                QuestionId.ANCHOR_4_PROBE: AnswerAnalysis(
                    polarity=Polarity.NEGATIVE,
                    confidence=0.93,
                    concrete=True,
                    on_topic=True,
                    reflection=(
                        "You told colleagues that the tools were terrible."
                    ),
                ),
                QuestionId.CATCH_ALL: AnswerAnalysis(
                    polarity=Polarity.NEUTRAL,
                    confidence=0.93,
                    concrete=True,
                    on_topic=True,
                    reflection=(
                        "You wanted AI literacy to explain tools in more detail."
                    ),
                ),
                QuestionId.WRAP_UP: AnswerAnalysis(
                    polarity=Polarity.NEUTRAL,
                    confidence=0.93,
                    concrete=True,
                    on_topic=True,
                    reflection=(
                        "Practical examples of affordable tools were the missing "
                        "detail."
                    ),
                ),
            }
            result = responses[question_id]
            assert answer
            return result

    provider = ScenarioProvider()
    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
        llm_max_calls_per_session=16,
    )
    service = InterviewService(
        MemoryInterviewRepository(),
        InterviewEngine(settings, provider),
    )
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    responses = [
        service.submit_text(
            session_id,
            "I was fired and now I am poor.",
        ),
        service.submit_text(session_id, "It is because of tools."),
        service.submit_text(
            session_id,
            "A client task was delayed because I could not access the tools.",
        ),
        service.submit_text(
            session_id,
            "Yes, I shared the training, but nothing helped.",
        ),
        service.submit_text(
            session_id,
            "I told them the tools were terrible.",
        ),
        service.submit_text(
            session_id,
            "AI literacy should explain more about tools.",
        ),
        service.submit_text(session_id, "Yes"),
        service.submit_text(
            session_id,
            "It should include practical examples of affordable tools.",
        ),
    ]

    assert responses[0].message.startswith(
        "Pole. You said you were fired and are now facing financial difficulty."
    )
    assert responses[1].message == (
        "You identified the tools as the main barrier. "
        "Can you describe one recent time when the tools got in the way?"
    )
    assert responses[3].message == (
        "Nothing improved after you shared the training. "
        "What did you share, and what showed that nothing helped?"
    )
    assert responses[6].message == "Please go ahead - what would you like to add?"
    assert responses[7].message.startswith(
        "Asante sana. Practical examples of affordable tools were the missing detail."
    )
    assert all("Sawa." not in response.message for response in responses)
    assert provider.calls == 7
    assert service.get_state(session_id).status == SessionStatus.COMPLETED


def test_llm_grounded_reflection_is_preserved_in_closing_message() -> None:
    class ClosingProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(self, **kwargs) -> AnswerAnalysis:
            del kwargs
            return AnswerAnalysis(
                polarity=Polarity.POSITIVE,
                confidence=0.92,
                concrete=True,
                on_topic=True,
                probe_strategy=ProbeStrategy.NONE,
                reflection=(
                    "The peer mentoring circle remains the most useful support."
                ),
            )

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
    )
    engine = InterviewEngine(settings, ClosingProvider())
    state = InterviewState(
        status=SessionStatus.IN_PROGRESS,
        consent_given=True,
        current_question_id=QuestionId.WRAP_UP,
    )

    result = engine.handle_answer(
        state,
        "The peer mentoring circle was the most useful support.",
        InputMode.TEXT,
    )

    assert result.response_type.value == "complete"
    assert result.message == (
        "Asante sana. The peer mentoring circle remains the most useful support. "
        "Thanks - that's everything I needed. Your responses will be anonymized "
        "and used to shape future training."
    )


def test_llm_receives_prior_substantive_context() -> None:
    class ContextCapturingProvider:
        def __init__(self) -> None:
            self.summaries: list[str] = []

        @property
        def enabled(self) -> bool:
            return True

        def classify(
            self,
            *,
            question_id: QuestionId,
            answer: str,
            rolling_summary: str,
            probes_remaining: int = 1,
            previous_probe_questions: tuple[str, ...] = (),
        ) -> AnswerAnalysis:
            del question_id, answer, probes_remaining, previous_probe_questions
            self.summaries.append(rolling_summary)
            return AnswerAnalysis(
                polarity=Polarity.NEGATIVE,
                confidence=0.9,
                concrete=True,
                barrier_named=True,
                on_topic=True,
                economic_outcome="no_change",
            )

    provider = ContextCapturingProvider()
    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
        llm_max_calls_per_session=16,
    )
    service = InterviewService(
        MemoryInterviewRepository(),
        InterviewEngine(settings, provider),
    )
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    first_answer = "My income and role have not changed since the training."
    service.submit_text(session_id, first_answer)
    service.submit_text(
        session_id,
        "My employer did not approve access to the required AI tools.",
    )

    assert provider.summaries[0] == ""
    assert first_answer in provider.summaries[1]


def test_llm_failure_is_visible_and_uses_safe_local_fallback() -> None:
    class FailingProvider:
        @property
        def enabled(self) -> bool:
            return True

        def classify(self, **kwargs) -> AnswerAnalysis:
            del kwargs
            raise RuntimeError("provider unavailable")

    settings = Settings(
        llm_enabled=True,
        llm_base_url="https://llm.test/v1",
        llm_api_key="test-only",
        llm_model="test-model",
        llm_mode="always",
        llm_max_calls_per_session=16,
    )
    service = InterviewService(
        MemoryInterviewRepository(),
        InterviewEngine(settings, FailingProvider()),
    )
    session_id = start_consented_interview(service)
    complete_demographics(service, session_id)

    response = service.submit_text(
        session_id,
        "I use AI for emails.",
    )

    assert response.question_id == QuestionId.ANCHOR_1_PROBE
    assert response.analysis_source == AnalysisSource.LOCAL_PROVIDER_FALLBACK
    assert response.probe_type == ProbeStrategy.CLARITY
    assert response.probe_reason == ProbeReason.VAGUE_OR_UNCLEAR
