from __future__ import annotations

from typing import Any

from app.interview.engine import EngineResult, InterviewEngine
from app.interview.questions import interview_progress
from app.models.api import (
    BatchAuditResponse,
    InterviewResponse,
    InterviewStateResponse,
)
from app.models.domain import (
    InputMode,
    InterviewRecord,
    QuestionId,
    Role,
    SessionStatus,
    Turn,
)
from app.repositories.base import InterviewRepository


class InterviewService:
    def __init__(
        self,
        repository: InterviewRepository,
        engine: InterviewEngine,
    ) -> None:
        self.repository = repository
        self.engine = engine

    def start_interview(self) -> InterviewResponse:
        state = self.engine.initial_state()
        result = self.engine.initial_result()
        first_turn = Turn(
            turn_number=1,
            role=Role.ASSISTANT,
            content=result.message,
            question_id=QuestionId.CONSENT,
        )
        record = self.repository.create(
            InterviewRecord(state=state, turns=[first_turn])
        )
        return self._response(record, result)

    def submit_consent(self, session_id: str, choice: str) -> InterviewResponse:
        def operation(record: InterviewRecord) -> EngineResult:
            result = self.engine.handle_consent(record.state, choice)
            if record.state.status == SessionStatus.COLLECTING_DEMOGRAPHICS:
                self._append_assistant(record, result)
            return result

        record, result = self.repository.transact(session_id, operation)
        return self._response(record, result)

    def submit_text(self, session_id: str, text: str) -> InterviewResponse:
        return self._submit_answer(session_id, text, InputMode.TEXT, None)

    def submit_voice(
        self,
        session_id: str,
        transcript: str,
        transcription_confidence: float | None,
    ) -> InterviewResponse:
        return self._submit_answer(
            session_id,
            transcript,
            InputMode.VOICE,
            transcription_confidence,
        )

    def stop(self, session_id: str) -> InterviewResponse:
        def operation(record: InterviewRecord) -> EngineResult:
            result = self.engine.stop(record.state)
            self._append_assistant(record, result)
            return result

        record, result = self.repository.transact(session_id, operation)
        return self._response(record, result)

    def get_state(self, session_id: str) -> InterviewStateResponse:
        record = self.repository.get(session_id)
        state = record.state
        return InterviewStateResponse(
            session_id=state.session_id,
            status=state.status,
            consent_given=state.consent_given,
            question_id=state.current_question_id,
            progress=interview_progress(
                state.current_question_id,
                state.status.value,
            ),
            mode_of_input=state.mode_of_input,
            demographics_completed=list(state.demographics.keys()),
            anchors_covered=state.anchors_covered,
            probes_used=state.probes_used,
            probe_counts=state.probe_counts,
            max_probes_per_anchor=state.max_probes_per_anchor,
            branch_history=state.branch_history,
            mixed_evidence=state.mixed_evidence,
            substantive_turns=state.substantive_turns,
            llm_calls_used=state.llm_calls_used,
            last_activity_at=state.last_activity_at,
            nudge_sent_at=state.nudge_sent_at,
            nudge_delivery=state.nudge_delivery,
            transcript=record.turns,
            tags=record.tags,
        )

    def batch_audit_status(self, session_id: str) -> BatchAuditResponse:
        record = self.repository.get(session_id)
        provider_state = (
            "The configured LLM is active for "
            f"{self.engine.settings.llm_mode} live moderation."
            if self.engine.settings.llm_enabled
            else "No live LLM is configured."
        )
        return BatchAuditResponse(
            session_id=session_id,
            status="not_run",
            live_tag_count=len(record.tags),
            message=(
                "The optional post-interview batch audit has not been run. "
                f"{provider_state}"
            ),
        )

    def export_record(self, session_id: str) -> dict[str, Any]:
        return self.repository.get(session_id).model_dump(mode="json")

    def _submit_answer(
        self,
        session_id: str,
        text: str,
        input_mode: InputMode,
        transcription_confidence: float | None,
    ) -> InterviewResponse:
        def operation(record: InterviewRecord) -> EngineResult:
            answered_question = record.state.current_question_id
            result = self.engine.handle_answer(
                record.state,
                text,
                input_mode,
                transcription_confidence,
            )
            respondent_turn: Turn | None = None
            if result.store_respondent_turn:
                respondent_turn = self._append_turn(
                    record,
                    role=Role.RESPONDENT,
                    content=text,
                    question_id=answered_question,
                    input_mode=input_mode,
                )
            if result.tag is not None:
                result.tag.turn_id = respondent_turn.id if respondent_turn else None
                record.tags.append(result.tag)
            self._append_assistant(record, result)
            return result

        record, result = self.repository.transact(session_id, operation)
        return self._response(record, result)

    def _append_assistant(
        self,
        record: InterviewRecord,
        result: EngineResult,
    ) -> Turn:
        return self._append_turn(
            record,
            role=Role.ASSISTANT,
            content=result.message,
            question_id=result.next_question_id,
            input_mode=None,
        )

    @staticmethod
    def _append_turn(
        record: InterviewRecord,
        *,
        role: Role,
        content: str,
        question_id: QuestionId | None,
        input_mode: InputMode | None,
    ) -> Turn:
        turn = Turn(
            turn_number=len(record.turns) + 1,
            role=role,
            content=content,
            question_id=question_id,
            input_mode=input_mode,
        )
        record.turns.append(turn)
        return turn

    @staticmethod
    def _response(
        record: InterviewRecord,
        result: EngineResult,
    ) -> InterviewResponse:
        state = record.state
        return InterviewResponse(
            session_id=state.session_id,
            status=state.status,
            response_type=result.response_type,
            question_id=result.next_question_id,
            message=result.message,
            allowed_choices=result.allowed_choices,
            progress=interview_progress(
                state.current_question_id,
                state.status.value,
            ),
            mode_of_input=state.mode_of_input,
            anchors_covered=state.anchors_covered,
            probes_used=state.probes_used,
            probe_counts=state.probe_counts,
            max_probes_per_anchor=state.max_probes_per_anchor,
            mixed_evidence=state.mixed_evidence,
            llm_calls_used=state.llm_calls_used,
            analysis_source=(
                result.tag.metadata.get("analysis_source")
                if result.tag is not None
                else None
            ),
            analysis_confidence=(
                result.tag.confidence_in_tagging if result.tag is not None else None
            ),
            analysis_polarity=(result.tag.polarity if result.tag is not None else None),
            needs_probe=(
                result.tag.metadata.get("needs_probe")
                if result.tag is not None
                else None
            ),
            probe_type=(
                result.tag.metadata.get("probe_type")
                if result.tag is not None
                else None
            ),
            probe_reason=(
                result.tag.metadata.get("probe_reason")
                if result.tag is not None
                else None
            ),
            probe_asked=(
                result.tag.metadata.get("probe_asked")
                if result.tag is not None
                else None
            ),
            probe_number=(
                result.tag.metadata.get("probe_number")
                if result.tag is not None
                else None
            ),
            probe_strategy=(
                result.tag.metadata.get("probe_strategy")
                if result.tag is not None
                else None
            ),
        )
