from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from app.interview.questions import ANCHOR_BY_QUESTION
from app.models.domain import (
    AnswerAnalysis,
    Polarity,
    ProbeReason,
    ProbeStrategy,
    QuestionId,
)


_MEASURABLE_OUTCOME_RE = re.compile(
    r"(?:\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|"
    r"ten|eleven|twelve)\s*(?:minutes?|hours?|days?|percent|%|shillings?|"
    r"dollars?|clients?)\b)|(?:\b(?:ksh|kes|usd)\s*\d)|(?:\$\s*\d)",
    flags=re.IGNORECASE,
)
_IMPACT_TERMS = (
    "allow me",
    "allows me",
    "allowed me",
    "enable me",
    "enables me",
    "enabled me",
    "so i can",
    "so that i",
    "which means",
    "as a result",
    "take more clients",
    "finish work earlier",
    "spend time",
    "study",
    "move up",
)


@lru_cache(maxsize=1)
def _rules() -> dict[str, list[str]]:
    path = Path(__file__).with_name("rules.json")
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_any(text: str, terms: list[str]) -> bool:
    return any(term.lower() in text for term in terms)


def _bottleneck_types(text: str) -> list[str]:
    groups = {
        "bottleneck_opportunity": ("opportunity", "opening", "chance"),
        "bottleneck_employer_buyin": (
            "employer",
            "manager",
            "management",
            "buy-in",
            "approval",
        ),
        "bottleneck_confidence": ("confidence", "afraid", "fear", "imposter"),
        "bottleneck_tooling_access": (
            "tool",
            "tools",
            "software",
            "hardware",
            "access",
            "internet",
            "laptop",
            "license",
        ),
        "bottleneck_skill_gap": ("skill", "knowledge", "experience", "practice"),
        "bottleneck_market": ("market", "demand", "client", "economy"),
        "bottleneck_time_or_funding": ("time", "funding", "money to invest"),
    }
    return [
        category
        for category, terms in groups.items()
        if any(term in text for term in terms)
    ]


def _benefit_mechanism(text: str) -> str | None:
    if any(
        term in text for term in ("new client", "side income", "freelance", "consult")
    ):
        return "new_income_stream"
    if any(term in text for term in ("promotion", "new role", "moved up")):
        return "internal_mobility"
    if any(term in text for term in ("new job", "was hired", "new employer")):
        return "external_mobility"
    if any(
        term in text
        for term in ("faster", "efficient", "productivity", "automated", "saved time")
    ):
        return "efficiency_in_current_role"
    return None


def _probe_decision(
    *,
    text: str,
    question_id: QuestionId,
    mixed: bool,
    vague: bool,
    concrete: bool,
    barrier_named: bool,
    benefit_named: bool,
    affirmative: bool,
    no_change: bool,
    adverse_change: bool,
    on_topic: bool,
    skip_requested: bool,
) -> tuple[ProbeStrategy, ProbeReason | None]:
    """Return a conservative local decision when the live LLM is unavailable."""

    anchor_question_id = ANCHOR_BY_QUESTION.get(question_id)
    if anchor_question_id is None or not on_topic or skip_requested:
        return ProbeStrategy.NONE, None

    if mixed:
        return ProbeStrategy.TENSION, ProbeReason.MIXED_OR_CONFLICTING

    measurable_outcome = bool(_MEASURABLE_OUTCOME_RE.search(text))
    impact_explained = any(term in text for term in _IMPACT_TERMS)
    if measurable_outcome and benefit_named and not impact_explained:
        return ProbeStrategy.DRILL_DOWN, ProbeReason.OUTCOME_WITHOUT_IMPACT

    complete_negative = (
        anchor_question_id == QuestionId.ANCHOR_1
        and (no_change or adverse_change)
    ) or (
        anchor_question_id == QuestionId.ANCHOR_4
        and not affirmative
        and text.startswith(("no", "not", "nothing", "i have not", "i haven't"))
    )
    if complete_negative:
        return ProbeStrategy.NONE, None

    lacks_required_detail = (
        vague
        or (
            anchor_question_id == QuestionId.ANCHOR_2
            and barrier_named
            and not concrete
        )
    )
    if lacks_required_detail:
        return ProbeStrategy.CLARITY, ProbeReason.VAGUE_OR_UNCLEAR

    return ProbeStrategy.NONE, None


def classify_answer(text: str, question_id: QuestionId) -> AnswerAnalysis:
    normalized = " ".join(text.lower().strip().split())
    words = re.findall(r"\b[\w'-]+\b", normalized)
    word_count = len(words)
    rules = _rules()

    no_change = _contains_any(normalized, rules["no_change"])
    adverse_change = _contains_any(normalized, rules["adverse_change"])
    positive = _contains_any(normalized, rules["positive_change"])
    barrier_named = _contains_any(normalized, rules["barrier"])
    benefit_named = _contains_any(normalized, rules["benefit"]) or positive
    vague_signal = _contains_any(normalized, rules["vague"])
    concrete_signal = _contains_any(normalized, rules["concrete"])
    sharing_signal = _contains_any(normalized, rules["sharing"])

    has_datable_detail = bool(
        re.search(
            r"\b(?:20\d{2}|january|february|march|april|may|june|july|august|"
            r"september|october|november|december|last week|last month|"
            r"\d{1,3}%|ksh|kes|shillings?|\d{1,3}(?:,\d{3})+)\b",
            normalized,
        )
    )
    concrete = adverse_change or (
        concrete_signal
        and (
            has_datable_detail
            or any(
                verb in normalized
                for verb in ("got ", "received ", "started ", "signed ", "completed ")
            )
        )
    )

    mixed = positive and (no_change or adverse_change or barrier_named)
    if mixed or positive:
        polarity = Polarity.POSITIVE
    elif no_change or adverse_change or barrier_named:
        polarity = Polarity.NEGATIVE
    else:
        polarity = Polarity.NEUTRAL

    vague = word_count < 8 or vague_signal
    off_topic = _contains_any(normalized, rules["off_topic"])
    topical_terms = (
        "training",
        "ai",
        "income",
        "job",
        "role",
        "work",
        "client",
        "team",
        "colleague",
        "opportunity",
        "employer",
        "fired",
        "laid off",
        "lost my job",
        "unemployed",
        "tool",
        "confidence",
        "salary",
        "business",
        "promotion",
        "raise",
        "responsibilit",
        "freelance",
        "side income",
    )
    on_topic = not off_topic and (
        question_id in {QuestionId.CATCH_ALL, QuestionId.WRAP_UP}
        or word_count <= 7
        or any(term in normalized for term in topical_terms)
    )

    apology = _contains_any(normalized, rules["apology"])
    correction = _contains_any(normalized, rules["correction"])
    skip = _contains_any(normalized, rules["skip"])
    sharing_denied = normalized.startswith(("no", "not", "nothing")) or any(
        phrase in normalized
        for phrase in (
            "did not share",
            "didn't share",
            "have not shared",
            "haven't shared",
            "never shared",
        )
    )
    affirmative = (
        sharing_signal or normalized.startswith(("yes", "i have", "we have"))
    ) and not sharing_denied
    tough_or_complex = (
        mixed
        or adverse_change
        or (polarity == Polarity.NEGATIVE and word_count >= 12)
    )
    positive_milestone = positive and concrete and not mixed

    confidence = 0.86
    if vague:
        confidence -= 0.22
    if mixed:
        confidence -= 0.08
    if not on_topic:
        confidence -= 0.25
    if word_count < 4:
        confidence = min(confidence, 0.35)
    confidence = max(0.10, min(0.98, round(confidence, 2)))

    economic_outcome: str | None = None
    if question_id in {QuestionId.ANCHOR_1, QuestionId.ANCHOR_1_PROBE}:
        if adverse_change and not positive:
            economic_outcome = "income_decrease_or_job_loss"
        elif no_change and not positive:
            economic_outcome = "no_change"
        elif any(
            term in normalized for term in ("raise", "salary", "income", "earned more")
        ):
            economic_outcome = "income_increase"
        elif any(
            term in normalized for term in ("promotion", "new role", "responsibilit")
        ):
            economic_outcome = "role_change_no_pay_change"
        elif positive:
            economic_outcome = "improved_current_role_only"
        else:
            economic_outcome = "too_early_to_tell"

    bottlenecks = _bottleneck_types(normalized)
    benefit_mechanism = _benefit_mechanism(normalized)
    probe_strategy, probe_reason = _probe_decision(
        text=normalized,
        question_id=question_id,
        mixed=mixed,
        vague=vague,
        concrete=concrete,
        barrier_named=barrier_named,
        benefit_named=benefit_named,
        affirmative=affirmative,
        no_change=no_change,
        adverse_change=adverse_change,
        on_topic=on_topic,
        skip_requested=skip,
    )

    return AnswerAnalysis(
        polarity=polarity,
        mixed_evidence=mixed,
        confidence=confidence,
        vague=vague,
        concrete=concrete,
        barrier_named=barrier_named,
        benefit_named=benefit_named,
        affirmative=affirmative,
        on_topic=on_topic,
        apology=apology,
        correction_or_error=correction,
        skip_requested=skip,
        tough_or_complex=tough_or_complex,
        positive_milestone=positive_milestone,
        economic_outcome=economic_outcome,
        bottleneck_types=bottlenecks,
        benefit_mechanism=benefit_mechanism,
        word_count=word_count,
        probe_strategy=probe_strategy,
        probe_reason=probe_reason,
    )
