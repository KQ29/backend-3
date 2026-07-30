from __future__ import annotations

import pytest

from app.interview.moderation import (
    clean_grounding_quote,
    clean_probe,
    clean_reflection,
)


ANSWER = (
    "The Cedar-947 workflow reduced weekly reporting from six hours to two, "
    "but paid software access is still blocked."
)


def test_grounded_reflection_and_probe_are_preserved() -> None:
    assert clean_reflection(
        "Weekly reporting now takes two hours while software access remains blocked.",
        answer=ANSWER,
    ) == (
        "Weekly reporting now takes two hours while software access remains blocked."
    )
    assert clean_probe(
        "When did the Cedar-947 workflow first reduce weekly reporting time?",
        answer=ANSWER,
    ) == "When did the Cedar-947 workflow first reduce weekly reporting time?"


def test_generic_or_ungrounded_language_is_rejected() -> None:
    assert clean_reflection("You described your experience.", answer=ANSWER) is None
    assert clean_reflection(
        "A completely unrelated topic was introduced.",
        answer=ANSWER,
    ) is None
    assert clean_probe("Can you tell me more?", answer=ANSWER) is None
    assert clean_probe(
        "Which unrelated hobby matters most?",
        answer=ANSWER,
    ) is None
    assert (
        clean_reflection(
            "Your employer blocked software access.",
            answer="Yes",
        )
        is None
    )


@pytest.mark.parametrize(
    "reflection",
    (
        "At least being fired gives you more time.",
        "Do not worry, being fired is temporary.",
        "Training caused you to be fired.",
        "You failed because you were fired.",
        "You should recover quickly after being fired.",
    ),
)
def test_reflection_rejects_blame_advice_or_minimization(
    reflection: str,
) -> None:
    assert (
        clean_reflection(
            reflection,
            answer="I was fired and my income stopped.",
        )
        is None
    )


def test_probe_rejects_accusatory_question() -> None:
    assert (
        clean_probe(
            "Why didn't you buy better tools?",
            answer="The tools are too expensive.",
        )
        is None
    )
    assert clean_probe(
        "Can you describe one recent time when the tools got in the way?",
        answer="It is because of tools.",
    ) == "Can you describe one recent time when the tools got in the way?"


def test_generated_identifiers_and_links_are_rejected() -> None:
    assert clean_reflection(
        "Cedar-947 details were sent to person@example.com.",
        answer=ANSWER,
    ) is None
    assert clean_probe(
        "Is Cedar-947 documented at https://example.com?",
        answer=ANSWER,
    ) is None


def test_grounding_quote_must_be_an_exact_safe_answer_span() -> None:
    assert (
        clean_grounding_quote(
            "weekly reporting from six hours to two",
            answer=ANSWER,
        )
        == "weekly reporting from six hours to two"
    )
    assert (
        clean_grounding_quote(
            "reporting became dramatically faster",
            answer=ANSWER,
        )
        is None
    )
