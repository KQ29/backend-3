from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from app.models.domain import (
    AnalysisSource,
    InputMode,
    Polarity,
    ProbeReason,
    ProbeStrategy,
    QuestionId,
    ResponseTag,
    ResponseType,
    SessionStatus,
    Turn,
)


class StartInterviewRequest(BaseModel):
    channel: str = Field(default="streamlit_demo", min_length=1, max_length=40)


class ConsentRequest(BaseModel):
    choice: str = Field(min_length=1, max_length=40)


class TextAnswerRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)


class InterviewResponse(BaseModel):
    session_id: str
    status: SessionStatus
    response_type: ResponseType
    question_id: QuestionId | None
    message: str
    allowed_choices: list[str] = Field(default_factory=list)
    progress: int = Field(ge=0, le=100)
    mode_of_input: InputMode | None = None
    anchors_covered: list[QuestionId] = Field(default_factory=list)
    probes_used: list[QuestionId] = Field(default_factory=list)
    probe_counts: dict[str, int] = Field(default_factory=dict)
    max_probes_per_anchor: int = Field(default=1, ge=0, le=2)
    mixed_evidence: bool = False
    llm_calls_used: int = 0
    analysis_source: AnalysisSource | None = None
    analysis_confidence: float | None = Field(default=None, ge=0, le=1)
    analysis_polarity: Polarity | None = None
    needs_probe: bool | None = None
    probe_type: ProbeStrategy | None = None
    probe_reason: ProbeReason | None = None
    probe_asked: bool | None = None
    probe_number: int | None = Field(default=None, ge=1, le=2)
    probe_strategy: ProbeStrategy | None = None
    transcript: str | None = None
    transcription_confidence: float | None = Field(default=None, ge=0, le=1)


class InterviewStateResponse(BaseModel):
    session_id: str
    status: SessionStatus
    consent_given: bool
    question_id: QuestionId
    progress: int = Field(ge=0, le=100)
    mode_of_input: InputMode | None
    demographics_completed: list[str]
    anchors_covered: list[QuestionId]
    probes_used: list[QuestionId]
    probe_counts: dict[str, int]
    max_probes_per_anchor: int = Field(ge=0, le=2)
    branch_history: list[str]
    mixed_evidence: bool
    substantive_turns: int
    llm_calls_used: int
    last_activity_at: datetime
    nudge_sent_at: datetime | None
    nudge_delivery: str | None
    transcript: list[Turn]
    tags: list[ResponseTag]


class BatchAuditResponse(BaseModel):
    session_id: str
    status: str
    live_tag_count: int
    message: str


class ErrorResponse(BaseModel):
    detail: str
