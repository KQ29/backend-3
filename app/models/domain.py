from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator


def utc_now() -> datetime:
    return datetime.now(UTC)


class SessionStatus(StrEnum):
    AWAITING_CONSENT = "awaiting_consent"
    COLLECTING_DEMOGRAPHICS = "collecting_demographics"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DECLINED = "declined"
    STOPPED = "stopped"
    ABANDONED = "abandoned"


class InputMode(StrEnum):
    TEXT = "text"
    VOICE = "voice"
    MIXED = "mixed"


class Role(StrEnum):
    ASSISTANT = "assistant"
    RESPONDENT = "respondent"


class ResponseType(StrEnum):
    CONSENT_CHOICE = "consent_choice"
    OPEN_TEXT = "open_text"
    COMPLETE = "complete"
    STOPPED = "stopped"


class Polarity(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class AnalysisSource(StrEnum):
    LOCAL = "local"
    LLM = "llm"
    LOCAL_PROVIDER_FALLBACK = "local_provider_fallback"
    LOCAL_LIMIT_FALLBACK = "local_limit_fallback"


class ProbeStrategy(StrEnum):
    NONE = "none"
    CLARITY = "clarity"
    DRILL_DOWN = "drill_down"
    TENSION = "tension"


class ProbeReason(StrEnum):
    VAGUE_OR_UNCLEAR = "vague_or_unclear"
    OUTCOME_WITHOUT_IMPACT = "outcome_without_impact"
    MIXED_OR_CONFLICTING = "mixed_or_conflicting"


class EconomicOutcome(StrEnum):
    INCOME_INCREASE = "income_increase"
    INCOME_DECREASE_OR_JOB_LOSS = "income_decrease_or_job_loss"
    ROLE_CHANGE_NO_PAY_CHANGE = "role_change_no_pay_change"
    IMPROVED_CURRENT_ROLE_ONLY = "improved_current_role_only"
    NO_CHANGE = "no_change"
    TOO_EARLY_TO_TELL = "too_early_to_tell"


class BottleneckType(StrEnum):
    OPPORTUNITY = "bottleneck_opportunity"
    EMPLOYER_BUYIN = "bottleneck_employer_buyin"
    CONFIDENCE = "bottleneck_confidence"
    TOOLING_ACCESS = "bottleneck_tooling_access"
    SKILL_GAP = "bottleneck_skill_gap"
    MARKET = "bottleneck_market"
    TIME_OR_FUNDING = "bottleneck_time_or_funding"
    NONE_REPORTED = "bottleneck_none_reported"


class BenefitMechanism(StrEnum):
    NEW_INCOME_STREAM = "new_income_stream"
    INTERNAL_MOBILITY = "internal_mobility"
    EXTERNAL_MOBILITY = "external_mobility"
    EFFICIENCY_IN_CURRENT_ROLE = "efficiency_in_current_role"
    CREDIBILITY_SIGNAL = "credibility_signal"
    NOT_APPLICABLE = "not_applicable"


class QuestionId(StrEnum):
    CONSENT = "consent"
    DEMO_NAME = "demo_name"
    DEMO_EMAIL = "demo_email"
    DEMO_AGE = "demo_age"
    DEMO_GENDER = "demo_gender"
    DEMO_COUNTY = "demo_county"
    DEMO_SUB_COUNTY = "demo_sub_county"
    DEMO_OCCUPATION = "demo_occupation"
    ANCHOR_1 = "anchor_1"
    ANCHOR_1_PROBE = "anchor_1_probe"
    ANCHOR_2 = "anchor_2"
    ANCHOR_2_PROBE = "anchor_2_probe"
    ANCHOR_3 = "anchor_3"
    ANCHOR_3_PROBE = "anchor_3_probe"
    ANCHOR_4 = "anchor_4"
    ANCHOR_4_PROBE = "anchor_4_probe"
    CATCH_ALL = "catch_all"
    WRAP_UP = "wrap_up"
    CLOSE = "close"


class Turn(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    turn_number: int = Field(ge=1)
    role: Role
    content: str = Field(min_length=1)
    question_id: QuestionId | None = None
    input_mode: InputMode | None = None
    created_at: datetime = Field(default_factory=utc_now)


class AnswerAnalysis(BaseModel):
    polarity: Polarity = Polarity.NEUTRAL
    mixed_evidence: bool = False
    confidence: float = Field(default=0.5, ge=0, le=1)
    vague: bool = False
    concrete: bool = False
    barrier_named: bool = False
    benefit_named: bool = False
    affirmative: bool = False
    on_topic: bool = True
    apology: bool = False
    correction_or_error: bool = False
    skip_requested: bool = False
    tough_or_complex: bool = False
    positive_milestone: bool = False
    economic_outcome: EconomicOutcome | None = None
    bottleneck_types: list[BottleneckType] = Field(default_factory=list)
    benefit_mechanism: BenefitMechanism | None = None
    word_count: int = 0
    probe_strategy: ProbeStrategy = ProbeStrategy.NONE
    probe_reason: ProbeReason | None = None
    suggested_probe: str | None = Field(default=None, max_length=240)
    reflection: str | None = Field(default=None, max_length=180)
    analysis_source: AnalysisSource = AnalysisSource.LOCAL

    @property
    def needs_probe(self) -> bool:
        return self.probe_strategy != ProbeStrategy.NONE

    @model_validator(mode="after")
    def validate_probe_decision(self) -> AnswerAnalysis:
        expected_reason = {
            ProbeStrategy.CLARITY: ProbeReason.VAGUE_OR_UNCLEAR,
            ProbeStrategy.DRILL_DOWN: ProbeReason.OUTCOME_WITHOUT_IMPACT,
            ProbeStrategy.TENSION: ProbeReason.MIXED_OR_CONFLICTING,
        }.get(self.probe_strategy)
        if self.probe_reason != expected_reason:
            raise ValueError("Probe strategy and reason must agree")
        if self.probe_strategy == ProbeStrategy.TENSION and not self.mixed_evidence:
            raise ValueError("A tension probe requires mixed evidence")
        return self


class ResponseTag(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    turn_id: str | None = None
    question_id: QuestionId
    source: str = "live"
    raw_response: str
    polarity: Polarity | None = None
    mixed_evidence: bool = False
    confidence_in_tagging: float = Field(default=0.5, ge=0, le=1)
    vague: bool = False
    concrete: bool = False
    on_topic: bool = True
    economic_outcome: EconomicOutcome | None = None
    bottleneck_types: list[BottleneckType] = Field(default_factory=list)
    benefit_mechanism: BenefitMechanism | None = None
    transcription_confidence: float | None = Field(default=None, ge=0, le=1)
    quotable_snippet: str | None = None
    force_coded: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class InterviewState(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    status: SessionStatus = SessionStatus.AWAITING_CONSENT
    consent_given: bool = False
    current_question_id: QuestionId = QuestionId.CONSENT
    demographics: dict[str, str] = Field(default_factory=dict)
    anchors_covered: list[QuestionId] = Field(default_factory=list)
    probes_used: list[QuestionId] = Field(default_factory=list)
    max_probes_per_anchor: int = Field(default=1, ge=0, le=2)
    probe_counts: dict[str, int] = Field(default_factory=dict)
    probe_questions: dict[str, list[str]] = Field(default_factory=dict)
    resume_question_id: QuestionId | None = None
    branch_queue: list[QuestionId] = Field(default_factory=list)
    branch_history: list[str] = Field(default_factory=list)
    mixed_evidence: bool = False
    mode_of_input: InputMode | None = None
    substantive_turns: int = 0
    llm_calls_used: int = 0
    rolling_summary: str = ""
    wrap_up_elaboration_requested: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    last_activity_at: datetime = Field(default_factory=utc_now)
    nudge_sent_at: datetime | None = None
    nudge_delivery: str | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def hydrate_legacy_probe_counts(self) -> InterviewState:
        if not self.probe_counts:
            for probe in self.probes_used:
                if probe.value.endswith("_probe"):
                    anchor = probe.value.removesuffix("_probe")
                    self.probe_counts[anchor] = (
                        self.probe_counts.get(anchor, 0) + 1
                    )
        if any(count < 0 or count > 2 for count in self.probe_counts.values()):
            raise ValueError("Probe counts must be between zero and two")
        return self


class InterviewRecord(BaseModel):
    state: InterviewState
    turns: list[Turn] = Field(default_factory=list)
    tags: list[ResponseTag] = Field(default_factory=list)
    revision: int = 0


class EngineResult(BaseModel):
    message: str
    next_question_id: QuestionId | None
    response_type: ResponseType
    allowed_choices: list[str] = Field(default_factory=list)
    tag: ResponseTag | None = None
    store_respondent_turn: bool = True
