from __future__ import annotations

from app.models.domain import AnswerAnalysis, QuestionId


class MockLLMProvider:
    """No-cost provider used by default.

    Returning ``None`` explicitly tells the caller to keep the deterministic
    local classification instead of inventing an AI result.
    """

    @property
    def enabled(self) -> bool:
        return False

    def classify(
        self,
        *,
        question_id: QuestionId,
        answer: str,
        rolling_summary: str,
        probes_remaining: int | None = None,
        previous_probe_questions: tuple[str, ...] = (),
    ) -> AnswerAnalysis | None:
        del (
            question_id,
            answer,
            rolling_summary,
            probes_remaining,
            previous_probe_questions,
        )
        return None
