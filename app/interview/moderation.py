from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any


_VALIDATION_PHRASES = (
    "that's great",
    "that is great",
    "great job",
    "good job",
    "well done",
    "i agree",
    "i understand",
    "i'm glad",
    "amazing",
    "excellent",
)
_SWAHILI_CUES = (
    "safi sana",
    "sawa",
    "naam",
    "usijali",
    "asante sana",
    "karibu",
    "pole",
)
_GENERIC_REFLECTION_PREFIXES = (
    "thank you for sharing",
    "you described your experience",
    "you shared your experience",
    "you mentioned a change",
    "you mentioned some changes",
    "you provided an answer",
)
_GENERIC_PROBES = {
    "can you elaborate",
    "can you tell me more",
    "could you elaborate",
    "could you tell me more",
    "how so",
    "what happened",
}
_HARMFUL_REFLECTION_PHRASES = (
    "at least",
    "do not worry",
    "don't worry",
    "everything will be",
    "it will be okay",
    "training caused",
    "you caused",
    "you failed",
    "you must",
    "you need to",
    "you should",
    "your fault",
)
_HARMFUL_PROBE_PHRASES = (
    "api key",
    "bank account",
    "email address",
    "home address",
    "login details",
    "password",
    "phone number",
    "secret key",
    "why did you not",
    "why didn't you",
    "why do you not",
    "why don't you",
)
_CONTENT_STOPWORDS = {
    "about",
    "after",
    "again",
    "also",
    "answer",
    "because",
    "been",
    "before",
    "being",
    "could",
    "described",
    "did",
    "does",
    "doing",
    "during",
    "experience",
    "from",
    "have",
    "having",
    "into",
    "just",
    "maybe",
    "mentioned",
    "more",
    "most",
    "no",
    "other",
    "response",
    "said",
    "share",
    "shared",
    "should",
    "since",
    "some",
    "something",
    "tell",
    "than",
    "that",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "thing",
    "things",
    "this",
    "those",
    "training",
    "very",
    "want",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "work",
    "would",
    "your",
    "yes",
}
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’\-][^\W_]+)*", flags=re.UNICODE)
_EMAIL_RE = re.compile(r"\b[^@\s]+@[^@\s]+\.[^@\s]+\b", flags=re.IGNORECASE)
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\w)\+?\d[\d .()\-]{7,}\d(?!\w)")


def _tokens(value: str) -> list[str]:
    return _TOKEN_RE.findall(value.casefold())


def _normalized_phrase(value: str) -> str:
    return " ".join(_tokens(value))


def _meaningful_tokens(value: str) -> set[str]:
    return {
        token
        for token in _tokens(value)
        if (len(token) >= 3 or token == "ai") and token not in _CONTENT_STOPWORDS
    }


def _contains_sensitive_text(value: str) -> bool:
    return bool(
        _EMAIL_RE.search(value)
        or _URL_RE.search(value)
        or _PHONE_RE.search(value)
        or "`" in value
    )


def shares_specific_detail(generated: str, source: str) -> bool:
    """Return whether generated text reuses a meaningful source detail."""

    source_tokens = _meaningful_tokens(source)
    if not source_tokens:
        return False
    return bool(source_tokens & _meaningful_tokens(generated))


def repeats_previous_probe(candidate: str | None, previous: tuple[str, ...]) -> bool:
    if not candidate:
        return False
    normalized = _normalized_phrase(candidate)
    return any(
        normalized == _normalized_phrase(prior)
        or SequenceMatcher(
            None,
            normalized,
            _normalized_phrase(prior),
        ).ratio()
        >= 0.88
        for prior in previous
    )


def clean_grounding_quote(value: Any, *, answer: str) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split()).strip("\"'")
    quote_tokens = _tokens(cleaned)
    answer_tokens = _tokens(answer)
    if (
        not cleaned
        or not quote_tokens
        or len(quote_tokens) > 12
        or (len(answer_tokens) > 3 and len(quote_tokens) < 2)
        or _contains_sensitive_text(cleaned)
    ):
        return None

    width = len(quote_tokens)
    if not any(
        answer_tokens[index : index + width] == quote_tokens
        for index in range(len(answer_tokens) - width + 1)
    ):
        return None
    return cleaned


def clean_reflection(value: Any, *, answer: str) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split()).strip("\"'")
    lowered = cleaned.casefold()
    normalized = _normalized_phrase(cleaned)
    if (
        not cleaned
        or "?" in cleaned
        or len(cleaned) > 180
        or len(cleaned.split()) > 18
        or any(phrase in lowered for phrase in _VALIDATION_PHRASES)
        or any(cue in lowered for cue in _SWAHILI_CUES)
        or any(phrase in lowered for phrase in _HARMFUL_REFLECTION_PHRASES)
        or any(normalized.startswith(prefix) for prefix in _GENERIC_REFLECTION_PREFIXES)
        or _contains_sensitive_text(cleaned)
        or not shares_specific_detail(cleaned, answer)
    ):
        return None
    cleaned = f"{cleaned.rstrip('.!')}."
    return cleaned if len(cleaned) <= 180 else None


def clean_probe(value: Any, *, answer: str) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.strip().split()).strip("\"'")
    lowered = cleaned.casefold()
    normalized = _normalized_phrase(cleaned)
    if (
        not cleaned
        or len(cleaned) > 240
        or len(cleaned.split()) > 28
        or any(phrase in lowered for phrase in _VALIDATION_PHRASES)
        or any(cue in lowered for cue in _SWAHILI_CUES)
        or any(phrase in lowered for phrase in _HARMFUL_PROBE_PHRASES)
        or normalized in _GENERIC_PROBES
        or _contains_sensitive_text(cleaned)
        or not shares_specific_detail(cleaned, answer)
    ):
        return None
    if cleaned.count("?") == 0:
        cleaned = f"{cleaned.rstrip('.!')}?"
    if cleaned.count("?") != 1 or not cleaned.endswith("?"):
        return None
    return cleaned if len(cleaned) <= 240 else None
