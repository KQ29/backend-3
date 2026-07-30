from __future__ import annotations

from enum import StrEnum

from app.models.domain import AnswerAnalysis, EconomicOutcome, Polarity


class VoiceCue(StrEnum):
    NONE = "none"
    SAFI = "safi"
    SAWA = "sawa"
    NAAM = "naam"
    USIJALI = "usijali"
    POLE = "pole"
    ASANTE = "asante"


PREFIXES: dict[VoiceCue, str] = {
    VoiceCue.NONE: "",
    VoiceCue.SAFI: "Safi sana!",
    VoiceCue.SAWA: "Sawa.",
    VoiceCue.NAAM: "Naam.",
    VoiceCue.USIJALI: "Usijali!",
    VoiceCue.POLE: "Pole.",
    VoiceCue.ASANTE: "Asante sana.",
}


def choose_voice_cue(
    analysis: AnswerAnalysis,
    *,
    closing: bool = False,
    moving_topic: bool = False,
) -> VoiceCue:
    if closing:
        return VoiceCue.ASANTE
    if (
        analysis.economic_outcome
        == EconomicOutcome.INCOME_DECREASE_OR_JOB_LOSS
    ):
        return VoiceCue.POLE
    if analysis.apology or analysis.correction_or_error or analysis.skip_requested:
        return VoiceCue.USIJALI
    if analysis.tough_or_complex:
        return VoiceCue.NAAM
    if analysis.positive_milestone:
        return VoiceCue.SAFI
    if moving_topic and analysis.reflection is None:
        if analysis.polarity == Polarity.NEGATIVE:
            return VoiceCue.NAAM
        return VoiceCue.SAWA
    return VoiceCue.NONE


def apply_voice_cue(message: str, cue: VoiceCue) -> str:
    prefix = PREFIXES[cue]
    if not prefix:
        return message
    if message.startswith(prefix):
        return message
    return f"{prefix} {message}"


def count_swahili_expressions(message: str) -> int:
    return sum(
        expression.lower() in message.lower()
        for expression in (
            "Safi sana!",
            "Sawa",
            "Naam",
            "Usijali!",
            "Pole.",
            "Asante sana",
            "Karibu",
        )
    )
