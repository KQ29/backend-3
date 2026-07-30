from __future__ import annotations

import re
from datetime import UTC, datetime

from app.core.config import Settings
from app.core.logging import get_logger
from app.interview.classifier import classify_answer
from app.interview.moderation import (
    clean_probe,
    clean_reflection,
    repeats_previous_probe,
)
from app.interview.questions import (
    ALLOWED_PROBE_BY_ANCHOR,
    ANCHOR_BY_QUESTION,
    ANCHOR_QUESTIONS,
    DEMOGRAPHIC_FIELD_BY_QUESTION,
    DEMOGRAPHIC_QUESTIONS,
    FALLBACK_PROBES_BY_STRATEGY,
    PROBE_QUESTIONS,
    QUESTION_TEXT,
    WRAP_UP_ELABORATION_TEXT,
)
from app.interview.swahili import apply_voice_cue, choose_voice_cue
from app.models.domain import (
    AnalysisSource,
    AnswerAnalysis,
    EconomicOutcome,
    EngineResult,
    InputMode,
    InterviewState,
    Polarity,
    QuestionId,
    ResponseTag,
    ResponseType,
    SessionStatus,
)
from app.providers.llm.base import LLMProvider
from app.providers.llm.mock import MockLLMProvider

YES_CHOICES = {"yes", "y", "yes, continue", "consent_yes", "ok", "okay"}
NO_CHOICES = {"no", "n", "no thanks", "consent_no"}
BARE_WRAP_UP_AFFIRMATIVES = {
    "i do",
    "yes",
    "yes i do",
    "yes please",
    "yes there is",
}
logger = get_logger(__name__)


class InvalidInterviewAction(ValueError):
    pass


class InterviewEngine:
    """Deterministic, bounded interview state machine."""

    def __init__(
        self,
        settings: Settings,
        llm_provider: LLMProvider | None = None,
    ):
        self.settings = settings
        self.llm_provider = llm_provider or MockLLMProvider()

    def initial_state(self) -> InterviewState:
        return InterviewState(
            max_probes_per_anchor=self.settings.max_probes_per_anchor,
        )

    def initial_result(self) -> EngineResult:
        return EngineResult(
            message=QUESTION_TEXT[QuestionId.CONSENT],
            next_question_id=QuestionId.CONSENT,
            response_type=ResponseType.CONSENT_CHOICE,
            allowed_choices=["consent_yes", "consent_no"],
            store_respondent_turn=False,
        )

    def handle_consent(self, state: InterviewState, choice: str) -> EngineResult:
        if state.status != SessionStatus.AWAITING_CONSENT:
            raise InvalidInterviewAction(
                "Consent has already been resolved for this session"
            )

        normalized = " ".join(choice.lower().strip().split())
        state.last_activity_at = datetime.now(UTC)
        if normalized in YES_CHOICES:
            state.consent_given = True
            state.status = SessionStatus.COLLECTING_DEMOGRAPHICS
            state.current_question_id = QuestionId.DEMO_NAME
            return EngineResult(
                message=QUESTION_TEXT[QuestionId.DEMO_NAME],
                next_question_id=QuestionId.DEMO_NAME,
                response_type=ResponseType.OPEN_TEXT,
                store_respondent_turn=False,
            )
        if normalized in NO_CHOICES:
            state.consent_given = False
            state.status = SessionStatus.DECLINED
            state.current_question_id = QuestionId.CLOSE
            state.completed_at = datetime.now(UTC)
            return EngineResult(
                message="Thank you for your time. The interview will not continue.",
                next_question_id=None,
                response_type=ResponseType.COMPLETE,
                store_respondent_turn=False,
            )
        raise InvalidInterviewAction("Choose either consent_yes or consent_no")

    def handle_answer(
        self,
        state: InterviewState,
        text: str,
        input_mode: InputMode,
        transcription_confidence: float | None = None,
    ) -> EngineResult:
        cleaned = text.strip()
        if not cleaned:
            raise InvalidInterviewAction("Response text cannot be empty")
        if state.status not in {
            SessionStatus.COLLECTING_DEMOGRAPHICS,
            SessionStatus.IN_PROGRESS,
        }:
            raise InvalidInterviewAction(
                f"Session cannot accept responses while {state.status.value}"
            )

        if cleaned.lower() == "stop":
            return self.stop(state)

        state.last_activity_at = datetime.now(UTC)
        current = state.current_question_id

        if current in DEMOGRAPHIC_QUESTIONS:
            return self._handle_demographic(state, current, cleaned)

        state.substantive_turns += 1
        self._update_input_mode(state, input_mode)
        if current == QuestionId.WRAP_UP and self._is_bare_wrap_up_affirmative(
            cleaned
        ):
            analysis = classify_answer(cleaned, current)
            self._update_rolling_summary(state, current, cleaned)
            if (
                state.wrap_up_elaboration_requested
                or state.substantive_turns >= self.settings.max_substantive_turns
            ):
                return self._complete(
                    state,
                    analysis,
                    QUESTION_TEXT[QuestionId.CLOSE].removeprefix("Asante sana. "),
                    current,
                    cleaned,
                    transcription_confidence,
                )

            state.wrap_up_elaboration_requested = True
            return EngineResult(
                message=WRAP_UP_ELABORATION_TEXT,
                next_question_id=QuestionId.WRAP_UP,
                response_type=ResponseType.OPEN_TEXT,
                tag=self._make_tag(
                    current,
                    cleaned,
                    analysis,
                    transcription_confidence,
                    force_coded=False,
                ),
            )

        analysis = self._classify(state, current, cleaned)
        self._update_rolling_summary(state, current, cleaned)

        if state.substantive_turns >= self.settings.max_substantive_turns:
            return self._complete(
                state,
                analysis,
                "We have reached the end of the interview. Your responses will be "
                "anonymized and used to shape future training.",
                current,
                cleaned,
                transcription_confidence,
            )

        if analysis.skip_requested:
            next_question = self._next_after_skip(state, current)
            tag = self._make_tag(
                current,
                cleaned,
                analysis,
                transcription_confidence,
                force_coded=False,
            )
            if next_question == QuestionId.CLOSE:
                return self._complete(
                    state,
                    analysis,
                    QUESTION_TEXT[QuestionId.CLOSE].removeprefix("Asante sana. "),
                    current,
                    cleaned,
                    transcription_confidence,
                    existing_tag=tag,
                )
            message = apply_voice_cue(
                QUESTION_TEXT[next_question],
                choose_voice_cue(analysis, moving_topic=True),
            )
            state.current_question_id = next_question
            return EngineResult(
                message=message,
                next_question_id=next_question,
                response_type=ResponseType.OPEN_TEXT,
                tag=tag,
            )

        if not analysis.on_topic and current not in {
            QuestionId.CATCH_ALL,
            QuestionId.WRAP_UP,
            *PROBE_QUESTIONS,
        }:
            message = (
                "Thanks for sharing that. To return to the interview: "
                f"{QUESTION_TEXT[current]}"
            )
            return EngineResult(
                message=message,
                next_question_id=current,
                response_type=ResponseType.OPEN_TEXT,
                tag=self._make_tag(
                    current,
                    cleaned,
                    analysis,
                    transcription_confidence,
                ),
            )

        next_question = self._transition(state, current, analysis)
        probe_number = (
            self._probe_count(state, next_question)
            if next_question in PROBE_QUESTIONS
            else None
        )
        tag = self._make_tag(
            current,
            cleaned,
            analysis,
            transcription_confidence,
            force_coded=current != QuestionId.CATCH_ALL,
            probe_asked=next_question in PROBE_QUESTIONS,
            probe_number=probe_number,
        )

        if next_question == QuestionId.CLOSE:
            return self._complete(
                state,
                analysis,
                QUESTION_TEXT[QuestionId.CLOSE].removeprefix("Asante sana. "),
                current,
                cleaned,
                transcription_confidence,
                existing_tag=tag,
            )

        moving_topic = self._is_topic_transition(current, next_question)
        question = self._question_text(
            next_question,
            analysis,
            probe_number=probe_number,
        )
        message = self._compose_next_message(
            question,
            analysis,
            moving_topic=moving_topic,
        )
        if next_question in PROBE_QUESTIONS:
            anchor = ANCHOR_BY_QUESTION[next_question]
            state.probe_questions.setdefault(anchor.value, []).append(question)
        state.current_question_id = next_question
        return EngineResult(
            message=message,
            next_question_id=next_question,
            response_type=ResponseType.OPEN_TEXT,
            tag=tag,
        )

    def stop(self, state: InterviewState) -> EngineResult:
        if state.status in {
            SessionStatus.COMPLETED,
            SessionStatus.DECLINED,
            SessionStatus.STOPPED,
            SessionStatus.ABANDONED,
        }:
            raise InvalidInterviewAction("This interview is already closed")
        state.status = SessionStatus.STOPPED
        state.current_question_id = QuestionId.CLOSE
        state.resume_question_id = None
        state.completed_at = datetime.now(UTC)
        state.last_activity_at = datetime.now(UTC)
        return EngineResult(
            message=(
                "Asante sana. You have stopped the interview. Your previously "
                "consented answers remain saved for the research record."
            ),
            next_question_id=None,
            response_type=ResponseType.STOPPED,
            store_respondent_turn=False,
        )

    def _handle_demographic(
        self,
        state: InterviewState,
        current: QuestionId,
        text: str,
    ) -> EngineResult:
        field = DEMOGRAPHIC_FIELD_BY_QUESTION[current]
        analysis = classify_answer(text, QuestionId.CATCH_ALL)
        if analysis.skip_requested:
            state.demographics[field] = "Prefer not to say"
        else:
            state.demographics[field] = text

        index = DEMOGRAPHIC_QUESTIONS.index(current)
        if index + 1 < len(DEMOGRAPHIC_QUESTIONS):
            next_question = DEMOGRAPHIC_QUESTIONS[index + 1]
            message = QUESTION_TEXT[next_question]
            if analysis.skip_requested or analysis.correction_or_error:
                message = apply_voice_cue(
                    message,
                    choose_voice_cue(analysis),
                )
        else:
            state.status = SessionStatus.IN_PROGRESS
            next_question = QuestionId.ANCHOR_1
            message = f"Sawa. {QUESTION_TEXT[next_question]}"

        state.current_question_id = next_question
        return EngineResult(
            message=message,
            next_question_id=next_question,
            response_type=ResponseType.OPEN_TEXT,
        )

    def _transition(
        self,
        state: InterviewState,
        current: QuestionId,
        analysis: AnswerAnalysis,
    ) -> QuestionId:
        if current in ANCHOR_QUESTIONS and current not in state.anchors_covered:
            state.anchors_covered.append(current)

        if current == QuestionId.ANCHOR_1:
            state.mixed_evidence = analysis.mixed_evidence
            if analysis.mixed_evidence:
                state.branch_history.append("anchor_1:mixed")
                state.branch_queue = [QuestionId.ANCHOR_3]
                return self._probe_or_continue(
                    state,
                    current,
                    analysis,
                    resume_at=QuestionId.ANCHOR_2,
                )
            if analysis.polarity.value == "positive" or analysis.benefit_named:
                state.branch_history.append("anchor_1:change")
                return self._probe_or_continue(
                    state,
                    current,
                    analysis,
                    resume_at=QuestionId.ANCHOR_3,
                )
            if analysis.needs_probe:
                state.branch_history.append("anchor_1:needs_clarity")
                return self._probe_or_continue(
                    state,
                    current,
                    analysis,
                    resume_at=QuestionId.ANCHOR_2,
                )
            state.branch_history.append("anchor_1:no_or_limited_change")
            return QuestionId.ANCHOR_2

        if current == QuestionId.ANCHOR_1_PROBE:
            fallback = (
                QuestionId.ANCHOR_2
                if analysis.polarity.value == "negative"
                and not analysis.benefit_named
                else QuestionId.ANCHOR_3
            )
            return self._probe_or_continue(
                state,
                current,
                analysis,
                resume_at=state.resume_question_id or fallback,
            )

        if current == QuestionId.ANCHOR_2:
            resume_at = self._after_anchor_2(state)
            return self._probe_or_continue(
                state,
                current,
                analysis,
                resume_at=resume_at,
            )

        if current == QuestionId.ANCHOR_2_PROBE:
            resume_at = (
                state.resume_question_id
                if state.resume_question_id is not None
                else self._after_anchor_2(state)
            )
            return self._probe_or_continue(
                state,
                current,
                analysis,
                resume_at=resume_at,
            )

        if current == QuestionId.ANCHOR_3:
            return self._probe_or_continue(
                state,
                current,
                analysis,
                resume_at=QuestionId.ANCHOR_4,
            )

        if current == QuestionId.ANCHOR_3_PROBE:
            return self._probe_or_continue(
                state,
                current,
                analysis,
                resume_at=state.resume_question_id or QuestionId.ANCHOR_4,
            )

        if current == QuestionId.ANCHOR_4:
            return self._probe_or_continue(
                state,
                current,
                analysis,
                resume_at=QuestionId.CATCH_ALL,
            )

        if current == QuestionId.ANCHOR_4_PROBE:
            return self._probe_or_continue(
                state,
                current,
                analysis,
                resume_at=state.resume_question_id or QuestionId.CATCH_ALL,
            )

        if current == QuestionId.CATCH_ALL:
            return QuestionId.WRAP_UP

        if current == QuestionId.WRAP_UP:
            return QuestionId.CLOSE

        raise InvalidInterviewAction(f"Unsupported question state: {current.value}")

    def _after_anchor_2(self, state: InterviewState) -> QuestionId:
        if state.branch_queue:
            return state.branch_queue.pop(0)
        return QuestionId.ANCHOR_4

    def _probe_or_continue(
        self,
        state: InterviewState,
        current: QuestionId,
        analysis: AnswerAnalysis,
        *,
        resume_at: QuestionId,
    ) -> QuestionId:
        anchor = ANCHOR_BY_QUESTION[current]
        probe = ALLOWED_PROBE_BY_ANCHOR[anchor]
        count = state.probe_counts.get(anchor.value, 0)
        if (
            analysis.needs_probe
            and count < state.max_probes_per_anchor
        ):
            state.probes_used.append(probe)
            state.probe_counts[anchor.value] = count + 1
            if current in ANCHOR_QUESTIONS or state.resume_question_id is None:
                state.resume_question_id = resume_at
            return probe
        if current in PROBE_QUESTIONS:
            return self._resume_after_probe(state, fallback=resume_at)
        if state.resume_question_id is not None:
            state.resume_question_id = None
            return resume_at
        return resume_at

    @staticmethod
    def _resume_after_probe(
        state: InterviewState,
        *,
        fallback: QuestionId,
    ) -> QuestionId:
        resume_at = state.resume_question_id or fallback
        state.resume_question_id = None
        return resume_at

    def _next_after_skip(
        self,
        state: InterviewState,
        current: QuestionId,
    ) -> QuestionId:
        if current in PROBE_QUESTIONS and state.resume_question_id is not None:
            resume_at = state.resume_question_id
            state.resume_question_id = None
            return resume_at
        if current in {QuestionId.ANCHOR_1, QuestionId.ANCHOR_1_PROBE}:
            return QuestionId.ANCHOR_2
        if current in {QuestionId.ANCHOR_2, QuestionId.ANCHOR_2_PROBE}:
            return self._after_anchor_2(state)
        if current in {QuestionId.ANCHOR_3, QuestionId.ANCHOR_3_PROBE}:
            return QuestionId.ANCHOR_4
        if current in {QuestionId.ANCHOR_4, QuestionId.ANCHOR_4_PROBE}:
            return QuestionId.CATCH_ALL
        if current == QuestionId.CATCH_ALL:
            return QuestionId.WRAP_UP
        return QuestionId.CLOSE

    def _complete(
        self,
        state: InterviewState,
        analysis: AnswerAnalysis,
        message: str,
        current: QuestionId,
        raw_response: str,
        transcription_confidence: float | None,
        existing_tag: ResponseTag | None = None,
    ) -> EngineResult:
        state.status = SessionStatus.COMPLETED
        state.current_question_id = QuestionId.CLOSE
        state.resume_question_id = None
        state.completed_at = datetime.now(UTC)
        grounded_message = (
            f"{analysis.reflection} {message}" if analysis.reflection else message
        )
        return EngineResult(
            message=apply_voice_cue(
                grounded_message,
                choose_voice_cue(analysis, closing=True),
            ),
            next_question_id=None,
            response_type=ResponseType.COMPLETE,
            tag=existing_tag
            or self._make_tag(
                current,
                raw_response,
                analysis,
                transcription_confidence,
                force_coded=current != QuestionId.CATCH_ALL,
            ),
        )

    def _make_tag(
        self,
        question_id: QuestionId,
        text: str,
        analysis: AnswerAnalysis,
        transcription_confidence: float | None,
        *,
        force_coded: bool = True,
        probe_asked: bool = False,
        probe_number: int | None = None,
    ) -> ResponseTag:
        return ResponseTag(
            question_id=question_id,
            raw_response=text,
            polarity=analysis.polarity if force_coded else None,
            mixed_evidence=analysis.mixed_evidence,
            confidence_in_tagging=analysis.confidence,
            vague=analysis.vague,
            concrete=analysis.concrete,
            on_topic=analysis.on_topic,
            economic_outcome=analysis.economic_outcome if force_coded else None,
            bottleneck_types=analysis.bottleneck_types if force_coded else [],
            benefit_mechanism=analysis.benefit_mechanism if force_coded else None,
            transcription_confidence=transcription_confidence,
            quotable_snippet=text[:160],
            force_coded=force_coded,
            metadata={
                "word_count": analysis.word_count,
                "analysis_source": analysis.analysis_source.value,
                "needs_probe": analysis.needs_probe,
                "probe_type": analysis.probe_strategy.value,
                "probe_reason": (
                    analysis.probe_reason.value if analysis.probe_reason else None
                ),
                "probe_asked": probe_asked,
                "probe_number": probe_number,
                "probe_strategy": analysis.probe_strategy.value,
                "llm_reflection": analysis.reflection,
                "llm_suggested_probe": analysis.suggested_probe,
            },
        )

    def _classify(
        self,
        state: InterviewState,
        question_id: QuestionId,
        answer: str,
    ) -> AnswerAnalysis:
        local_analysis = classify_answer(answer, question_id)
        anchor_question_id = ANCHOR_BY_QUESTION.get(question_id)
        probe_count = (
            state.probe_counts.get(anchor_question_id.value, 0)
            if anchor_question_id is not None
            else 0
        )
        probes_remaining = (
            max(0, state.max_probes_per_anchor - probe_count)
            if anchor_question_id is not None
            else 0
        )
        previous_probe_questions = (
            tuple(state.probe_questions.get(anchor_question_id.value, ()))
            if anchor_question_id is not None
            else ()
        )
        provider_available = self.settings.llm_enabled and self.llm_provider.enabled
        under_call_limit = (
            state.llm_calls_used < self.settings.llm_max_calls_per_session
        )
        should_use_llm = (
            provider_available
            and under_call_limit
            and (
                self.settings.llm_mode == "always"
                or local_analysis.confidence
                < self.settings.llm_low_confidence_threshold
            )
        )
        if not should_use_llm:
            if provider_available and not under_call_limit:
                return local_analysis.model_copy(
                    update={
                        "analysis_source": AnalysisSource.LOCAL_LIMIT_FALLBACK,
                    }
                )
            return local_analysis

        state.llm_calls_used += 1
        try:
            llm_analysis = self.llm_provider.classify(
                question_id=question_id,
                answer=answer,
                rolling_summary=state.rolling_summary,
                probes_remaining=probes_remaining,
                previous_probe_questions=previous_probe_questions,
            )
        except Exception as exc:  # noqa: BLE001 - fail safe at provider boundary
            logger.warning(
                "LLM classification fallback failed; retaining local result",
                extra={
                    "context": {
                        "session_id": state.session_id,
                        "question_id": question_id.value,
                        "provider_call": state.llm_calls_used,
                        "llm_mode": self.settings.llm_mode,
                        "error_type": type(exc).__name__,
                    }
                },
            )
            return local_analysis.model_copy(
                update={
                    "analysis_source": AnalysisSource.LOCAL_PROVIDER_FALLBACK,
                }
            )
        if llm_analysis is None:
            return local_analysis.model_copy(
                update={
                    "analysis_source": AnalysisSource.LOCAL_PROVIDER_FALLBACK,
                }
            )
        return self._apply_deterministic_safety_signals(
            local_analysis,
            llm_analysis,
            answer=answer,
            previous_probe_questions=previous_probe_questions,
        )

    @staticmethod
    def _apply_deterministic_safety_signals(
        local_analysis: AnswerAnalysis,
        llm_analysis: AnswerAnalysis,
        *,
        answer: str,
        previous_probe_questions: tuple[str, ...],
    ) -> AnswerAnalysis:
        """Keep control phrases and generated language inside deterministic bounds."""

        hardship_disclosed = (
            local_analysis.economic_outcome
            == EconomicOutcome.INCOME_DECREASE_OR_JOB_LOSS
        )
        tough_or_complex = (
            llm_analysis.tough_or_complex
            or local_analysis.tough_or_complex
            or hardship_disclosed
        )
        suggested_probe = clean_probe(
            llm_analysis.suggested_probe,
            answer=answer,
        )
        if repeats_previous_probe(suggested_probe, previous_probe_questions):
            suggested_probe = None
        return llm_analysis.model_copy(
            update={
                "analysis_source": AnalysisSource.LLM,
                "word_count": local_analysis.word_count,
                "affirmative": (
                    llm_analysis.affirmative or local_analysis.affirmative
                ),
                "barrier_named": (
                    llm_analysis.barrier_named or local_analysis.barrier_named
                ),
                "concrete": llm_analysis.concrete or local_analysis.concrete,
                "economic_outcome": (
                    local_analysis.economic_outcome
                    if hardship_disclosed
                    else llm_analysis.economic_outcome
                ),
                "on_topic": llm_analysis.on_topic or hardship_disclosed,
                "polarity": (
                    Polarity.NEGATIVE
                    if hardship_disclosed
                    else llm_analysis.polarity
                ),
                "positive_milestone": (
                    llm_analysis.positive_milestone and not tough_or_complex
                ),
                "reflection": clean_reflection(
                    llm_analysis.reflection,
                    answer=answer,
                ),
                "suggested_probe": suggested_probe,
                "tough_or_complex": tough_or_complex,
                "skip_requested": (
                    llm_analysis.skip_requested or local_analysis.skip_requested
                ),
                "apology": llm_analysis.apology or local_analysis.apology,
                "correction_or_error": (
                    llm_analysis.correction_or_error
                    or local_analysis.correction_or_error
                ),
            }
        )

    @staticmethod
    def _update_rolling_summary(
        state: InterviewState,
        question_id: QuestionId,
        answer: str,
    ) -> None:
        bounded_answer = " ".join(answer.split())[:600]
        new_line = f"{question_id.value}: {bounded_answer}"
        combined = "\n".join(item for item in (state.rolling_summary, new_line) if item)
        state.rolling_summary = combined[-3000:]

    @staticmethod
    def _probe_count(
        state: InterviewState,
        question_id: QuestionId,
    ) -> int:
        anchor = ANCHOR_BY_QUESTION.get(question_id)
        return state.probe_counts.get(anchor.value, 0) if anchor else 0

    @staticmethod
    def _question_text(
        next_question: QuestionId,
        analysis: AnswerAnalysis,
        *,
        probe_number: int | None,
    ) -> str:
        question = QUESTION_TEXT[next_question]
        if next_question in PROBE_QUESTIONS:
            fallbacks = FALLBACK_PROBES_BY_STRATEGY.get(analysis.probe_strategy)
            if fallbacks:
                fallback_index = min(max((probe_number or 1) - 1, 0), 1)
                question = analysis.suggested_probe or fallbacks[fallback_index]
            elif analysis.suggested_probe:
                question = analysis.suggested_probe
        return question

    @staticmethod
    def _compose_next_message(
        question: str,
        analysis: AnswerAnalysis,
        *,
        moving_topic: bool,
    ) -> str:
        message = question
        if analysis.reflection:
            message = f"{analysis.reflection} {question}"
        cue = choose_voice_cue(analysis, moving_topic=moving_topic)
        return apply_voice_cue(message, cue)

    @staticmethod
    def _update_input_mode(state: InterviewState, input_mode: InputMode) -> None:
        if input_mode == InputMode.MIXED:
            state.mode_of_input = InputMode.MIXED
        elif state.mode_of_input is None:
            state.mode_of_input = input_mode
        elif state.mode_of_input != input_mode:
            state.mode_of_input = InputMode.MIXED

    @staticmethod
    def _is_topic_transition(current: QuestionId, next_question: QuestionId) -> bool:
        return (
            current.value.split("_probe")[0] != next_question.value.split("_probe")[0]
        )

    @staticmethod
    def _is_bare_wrap_up_affirmative(answer: str) -> bool:
        normalized = " ".join(re.findall(r"[^\W_]+", answer.casefold()))
        return normalized in BARE_WRAP_UP_AFFIRMATIVES
