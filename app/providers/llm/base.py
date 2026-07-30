from __future__ import annotations

from typing import Protocol

from app.models.domain import AnswerAnalysis, QuestionId


class LLMProvider(Protocol):
    @property
    def enabled(self) -> bool: ...

    def classify(
        self,
        *,
        question_id: QuestionId,
        answer: str,
        rolling_summary: str,
        probes_remaining: int | None = None,
        previous_probe_questions: tuple[str, ...] = (),
    ) -> AnswerAnalysis | None: ...
