from __future__ import annotations

import json

import httpx
import pytest

from app.models.domain import (
    AnalysisSource,
    Polarity,
    ProbeReason,
    ProbeStrategy,
    QuestionId,
)
from app.providers.llm.openai_compatible import (
    LLMProviderError,
    OpenAICompatibleLLMProvider,
)
from app.providers.stt.speechmatics import (
    SpeechmaticsBatchProvider,
    SpeechToTextError,
)


def _valid_llm_classification(**overrides: object) -> dict[str, object]:
    classification: dict[str, object] = {
        "polarity": "positive",
        "mixed_evidence": False,
        "confidence": 0.93,
        "vague": False,
        "concrete": True,
        "barrier_named": False,
        "benefit_named": True,
        "affirmative": True,
        "on_topic": True,
        "apology": False,
        "correction_or_error": False,
        "skip_requested": False,
        "tough_or_complex": False,
        "positive_milestone": True,
        "economic_outcome": "income_increase",
        "bottleneck_types": [],
        "benefit_mechanism": "internal_mobility",
        "needs_probe": True,
        "probe_type": "drill_down",
        "probe_reason": "outcome_without_impact",
        "suggested_probe": "What did the raise allow you to do at work?",
        "reflection": "You connected the raise to using AI at work.",
        "grounding_quote": "received a raise",
    }
    classification.update(overrides)
    return classification


def _completion_response(
    classification: dict[str, object],
    *,
    finish_reason: str | None = "stop",
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [
                {
                    "finish_reason": finish_reason,
                    "message": {
                        "content": json.dumps(classification),
                    }
                }
            ]
        },
    )


def test_openai_compatible_llm_returns_validated_classification() -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return _completion_response(_valid_llm_classification())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=256,
        client=client,
    )

    result = provider.classify(
        question_id=QuestionId.ANCHOR_1,
        answer="I received a raise after using AI at work.",
        rolling_summary="",
    )

    assert result is not None
    assert result.polarity == Polarity.POSITIVE
    assert result.concrete is True
    assert result.analysis_source == AnalysisSource.LLM
    assert result.needs_probe is True
    assert result.probe_strategy == ProbeStrategy.DRILL_DOWN
    assert result.probe_reason == ProbeReason.OUTCOME_WITHOUT_IMPACT
    assert result.suggested_probe == "What did the raise allow you to do at work?"
    assert result.reflection == "You connected the raise to using AI at work."
    assert len(calls) == 1
    assert calls[0].url.path == "/v1/chat/completions"
    assert calls[0].headers["authorization"] == "Bearer test-secret"
    request_body = json.loads(calls[0].content)
    assert request_body["stream"] is False
    assert request_body["guided_json"]["type"] == "object"
    assert set(request_body["guided_json"]["required"]) == set(
        _valid_llm_classification()
    )
    user_payload = json.loads(request_body["messages"][1]["content"])
    assert user_payload["previous_probe_questions"] == []
    assert user_payload["allowed_follow_up"] == {
        "question_id": "anchor_1_probe",
        "anchor_research_objective": (
            "Understand whether work, income, responsibilities, or opportunities "
            "changed and what the respondent means by the change."
        ),
        "allowed_probe_types": ["clarity", "drill_down", "tension"],
        "probes_remaining": 1,
    }


def test_openai_compatible_llm_rejects_malformed_output() -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "not-json"}}]},
            )
        )
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=256,
        client=client,
    )

    with pytest.raises(LLMProviderError, match="invalid classification"):
        provider.classify(
            question_id=QuestionId.ANCHOR_1,
            answer="Maybe.",
            rolling_summary="",
        )


def test_openai_compatible_llm_accepts_json_wrapped_in_prose() -> None:
    classification = _valid_llm_classification()
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": (
                                    "Here is the result:\n"
                                    f"{json.dumps(classification)}\nDone."
                                )
                            },
                        }
                    ]
                },
            )
        )
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=256,
        client=client,
    )

    result = provider.classify(
        question_id=QuestionId.ANCHOR_1,
        answer="I received a raise after using AI at work.",
        rolling_summary="",
    )

    assert result is not None
    assert result.analysis_source == AnalysisSource.LLM


def test_openai_compatible_llm_rejects_validating_moderator_language() -> None:
    classification = _valid_llm_classification(
        confidence=0.9,
        vague=True,
        concrete=False,
        positive_milestone=False,
        economic_outcome="improved_current_role_only",
        benefit_mechanism="efficiency_in_current_role",
        probe_type="clarity",
        probe_reason="vague_or_unclear",
        suggested_probe="Great job! Can you give one example about improvement?",
        reflection="That's great, I understand how work improved.",
        grounding_quote="improved at work",
    )
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: _completion_response(classification)
        )
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=512,
        client=client,
    )

    with pytest.raises(LLMProviderError, match="invalid classification"):
        provider.classify(
            question_id=QuestionId.ANCHOR_1,
            answer="Things have improved at work since training.",
            rolling_summary="",
        )


@pytest.mark.parametrize(
    "classification",
    [
        {},
        _valid_llm_classification(reflection="You described your experience."),
        _valid_llm_classification(suggested_probe="Can you tell me more?"),
        _valid_llm_classification(unexpected_field=True),
    ],
)
def test_openai_compatible_llm_rejects_sparse_generic_or_unknown_output(
    classification: dict[str, object],
) -> None:
    client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: _completion_response(classification)
        )
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=512,
        client=client,
    )

    with pytest.raises(LLMProviderError, match="invalid classification"):
        provider.classify(
            question_id=QuestionId.ANCHOR_1,
            answer="I received a raise after using AI at work.",
            rolling_summary="",
        )


@pytest.mark.parametrize(
    (
        "answer",
        "classification",
        "expected_type",
        "expected_reason",
    ),
    [
        pytest.param(
            "I use AI for emails.",
            _valid_llm_classification(
                polarity="neutral",
                vague=True,
                concrete=False,
                benefit_named=False,
                positive_milestone=False,
                economic_outcome="too_early_to_tell",
                benefit_mechanism=None,
                probe_type="clarity",
                probe_reason="vague_or_unclear",
                suggested_probe=(
                    "What happened the last time you used AI for emails?"
                ),
                reflection="You use AI for emails.",
                grounding_quote="AI for emails",
            ),
            ProbeStrategy.CLARITY,
            ProbeReason.VAGUE_OR_UNCLEAR,
            id="clarity",
        ),
        pytest.param(
            "AI saves me two hours every week.",
            _valid_llm_classification(
                economic_outcome="improved_current_role_only",
                benefit_mechanism="efficiency_in_current_role",
                probe_type="drill_down",
                probe_reason="outcome_without_impact",
                suggested_probe=(
                    "What do those two hours allow you to do at work?"
                ),
                reflection="AI saves you two hours every week.",
                grounding_quote="saves me two hours",
            ),
            ProbeStrategy.DRILL_DOWN,
            ProbeReason.OUTCOME_WITHOUT_IMPACT,
            id="drill-down",
        ),
        pytest.param(
            "AI makes reports faster, but I worry about depending on it.",
            _valid_llm_classification(
                mixed_evidence=True,
                tough_or_complex=True,
                positive_milestone=False,
                economic_outcome="improved_current_role_only",
                benefit_mechanism="efficiency_in_current_role",
                probe_type="tension",
                probe_reason="mixed_or_conflicting",
                suggested_probe=(
                    "How does making reports faster fit with your worry about "
                    "depending on AI?"
                ),
                reflection=(
                    "Reports became faster alongside your worry about depending on AI."
                ),
                grounding_quote="reports faster, but I worry",
            ),
            ProbeStrategy.TENSION,
            ProbeReason.MIXED_OR_CONFLICTING,
            id="tension",
        ),
        pytest.param(
            "AI drafts the report, which lets me take one more client each week.",
            _valid_llm_classification(
                economic_outcome="improved_current_role_only",
                benefit_mechanism="efficiency_in_current_role",
                needs_probe=False,
                probe_type="none",
                probe_reason=None,
                suggested_probe=None,
                reflection="AI lets you take one more client each week.",
                grounding_quote="lets me take one more client",
            ),
            ProbeStrategy.NONE,
            None,
            id="no-probe",
        ),
    ],
)
def test_openai_compatible_llm_accepts_each_bounded_probe_decision(
    answer: str,
    classification: dict[str, object],
    expected_type: ProbeStrategy,
    expected_reason: ProbeReason | None,
) -> None:
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=512,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _completion_response(classification)
            )
        ),
    )

    result = provider.classify(
        question_id=QuestionId.ANCHOR_1,
        answer=answer,
        rolling_summary="",
    )

    assert result is not None
    assert result.probe_strategy == expected_type
    assert result.probe_reason == expected_reason
    assert result.needs_probe is (expected_type != ProbeStrategy.NONE)
    assert (result.suggested_probe is not None) is result.needs_probe


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "needs_probe": False,
            "probe_type": "drill_down",
            "probe_reason": "outcome_without_impact",
        },
        {
            "needs_probe": True,
            "probe_type": "none",
            "probe_reason": None,
            "suggested_probe": None,
        },
        {
            "needs_probe": True,
            "probe_type": "clarity",
            "probe_reason": "outcome_without_impact",
        },
    ],
    ids=("false-with-type", "true-with-none", "reason-type-mismatch"),
)
def test_openai_compatible_llm_rejects_inconsistent_probe_decisions(
    overrides: dict[str, object],
) -> None:
    classification = _valid_llm_classification(**overrides)
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=512,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _completion_response(classification)
            )
        ),
    )

    with pytest.raises(LLMProviderError, match="invalid classification"):
        provider.classify(
            question_id=QuestionId.ANCHOR_1,
            answer="I received a raise after using AI at work.",
            rolling_summary="",
        )


def test_openai_compatible_llm_rejects_tension_without_mixed_evidence() -> None:
    classification = _valid_llm_classification(
        probe_type="tension",
        probe_reason="mixed_or_conflicting",
        suggested_probe=(
            "How does the raise fit with your concern about using AI at work?"
        ),
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=512,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _completion_response(classification)
            )
        ),
    )

    with pytest.raises(LLMProviderError, match="invalid classification"):
        provider.classify(
            question_id=QuestionId.ANCHOR_1,
            answer="I received a raise after using AI at work.",
            rolling_summary="",
        )


def test_openai_compatible_llm_rejects_truncated_completion() -> None:
    classification = _valid_llm_classification()
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=512,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _completion_response(
                    classification,
                    finish_reason="length",
                )
            )
        ),
    )

    with pytest.raises(LLMProviderError, match="invalid classification"):
        provider.classify(
            question_id=QuestionId.ANCHOR_1,
            answer="I received a raise after using AI at work.",
            rolling_summary="",
        )


def test_openai_compatible_llm_rejects_probe_requesting_sensitive_data() -> None:
    classification = _valid_llm_classification(
        polarity="negative",
        vague=True,
        concrete=False,
        barrier_named=True,
        benefit_named=False,
        positive_milestone=False,
        economic_outcome="too_early_to_tell",
        benefit_mechanism=None,
        probe_type="clarity",
        probe_reason="vague_or_unclear",
        suggested_probe="What is your password for the AI tool?",
        reflection="You cannot access the AI tool.",
        grounding_quote="access the AI tool",
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=512,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _completion_response(classification)
            )
        ),
    )

    with pytest.raises(LLMProviderError, match="invalid classification"):
        provider.classify(
            question_id=QuestionId.ANCHOR_2,
            answer="I cannot access the AI tool.",
            rolling_summary="",
        )


def test_openai_compatible_llm_allows_distinct_second_probe_when_one_remains() -> None:
    previous_probe = (
        "Can you describe one recent time when the tools blocked your work?"
    )
    classification = _valid_llm_classification(
        polarity="negative",
        vague=False,
        concrete=True,
        barrier_named=True,
        benefit_named=False,
        positive_milestone=False,
        economic_outcome="too_early_to_tell",
        benefit_mechanism=None,
        probe_type="drill_down",
        probe_reason="outcome_without_impact",
        suggested_probe=(
            "What did the three-hour client report delay prevent you from doing?"
        ),
        reflection=(
            "The tools delayed a client report by three hours yesterday."
        ),
        grounding_quote="delayed a client report by three hours",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _completion_response(classification)

    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=512,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = provider.classify(
        question_id=QuestionId.ANCHOR_2_PROBE,
        answer="Yesterday the tools delayed a client report by three hours.",
        rolling_summary="anchor_2: The tools are unreliable.",
        probes_remaining=1,
        previous_probe_questions=(previous_probe,),
    )

    assert result is not None
    assert result.needs_probe is True
    assert result.probe_strategy == ProbeStrategy.DRILL_DOWN
    request_body = json.loads(requests[0].content)
    user_payload = json.loads(request_body["messages"][1]["content"])
    assert user_payload["question"] == previous_probe
    assert user_payload["previous_probe_questions"] == [previous_probe]
    assert user_payload["allowed_follow_up"] == {
        "question_id": "anchor_2_probe",
        "anchor_research_objective": (
            "Understand the respondent's current barrier and how it affects work or "
            "income."
        ),
        "allowed_probe_types": ["clarity", "drill_down", "tension"],
        "probes_remaining": 1,
    }


def test_openai_compatible_llm_requires_no_probe_when_none_remain() -> None:
    previous_probes = (
        "Can you describe one recent time when the tools blocked your work?",
        "What did the three-hour client report delay prevent you from doing?",
    )
    classification = _valid_llm_classification(
        polarity="negative",
        benefit_named=False,
        positive_milestone=False,
        economic_outcome="too_early_to_tell",
        benefit_mechanism=None,
        needs_probe=False,
        probe_type="none",
        probe_reason=None,
        suggested_probe=None,
        reflection="You missed the client deadline because of the delay.",
        grounding_quote="missed the client deadline",
    )
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return _completion_response(classification)

    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=512,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    result = provider.classify(
        question_id=QuestionId.ANCHOR_2_PROBE,
        answer="The delay meant I missed the client deadline.",
        rolling_summary="",
        probes_remaining=0,
        previous_probe_questions=previous_probes,
    )

    assert result is not None
    assert result.needs_probe is False
    assert result.reflection == "You missed the client deadline because of the delay."
    request_body = json.loads(requests[0].content)
    user_payload = json.loads(request_body["messages"][1]["content"])
    assert user_payload["question"] == previous_probes[-1]
    assert user_payload["previous_probe_questions"] == list(previous_probes)
    assert user_payload["allowed_follow_up"] is None


def test_openai_compatible_llm_rejects_near_duplicate_second_probe() -> None:
    previous_probe = (
        "When did the AI tools most recently block your client report?"
    )
    classification = _valid_llm_classification(
        polarity="negative",
        vague=True,
        concrete=False,
        barrier_named=True,
        benefit_named=False,
        positive_milestone=False,
        economic_outcome="too_early_to_tell",
        benefit_mechanism=None,
        probe_type="clarity",
        probe_reason="vague_or_unclear",
        suggested_probe=(
            "When did the AI tools most recently block your client reports?"
        ),
        reflection="The AI tools blocked your client report yesterday.",
        grounding_quote="AI tools blocked my client report",
    )
    provider = OpenAICompatibleLLMProvider(
        base_url="https://integrate.test/v1",
        api_key="test-secret",
        model="meta/llama-test",
        timeout_seconds=10,
        max_output_tokens=512,
        client=httpx.Client(
            transport=httpx.MockTransport(
                lambda request: _completion_response(classification)
            )
        ),
    )

    with pytest.raises(LLMProviderError, match="invalid classification"):
        provider.classify(
            question_id=QuestionId.ANCHOR_2_PROBE,
            answer="The AI tools blocked my client report yesterday.",
            rolling_summary="",
            probes_remaining=1,
            previous_probe_questions=(previous_probe,),
        )


def test_speechmatics_batch_provider_transcribes_audio() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "POST" and request.url.path.endswith("/jobs/"):
            return httpx.Response(201, json={"id": "job-123"})
        if request.method == "GET" and request.url.path.endswith("/jobs/job-123"):
            return httpx.Response(
                200,
                json={"job": {"status": "done", "duration": 2.4}},
            )
        if request.url.path.endswith("/jobs/job-123/transcript"):
            return httpx.Response(
                200,
                json={
                    "results": [
                        {
                            "type": "word",
                            "alternatives": [{"content": "Hello", "confidence": 0.96}],
                        },
                        {
                            "type": "word",
                            "alternatives": [{"content": "Kenya", "confidence": 0.94}],
                        },
                        {
                            "type": "punctuation",
                            "alternatives": [{"content": "."}],
                        },
                    ]
                },
            )
        return httpx.Response(404)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = SpeechmaticsBatchProvider(
        api_key="speech-test-secret",
        base_url="https://asr.test/v2",
        timeout_seconds=10,
        client=client,
    )

    result = provider.transcribe(
        audio=b"mock-ogg-audio",
        filename="voice.ogg",
        mime_type="audio/ogg",
        language_hint="en",
    )

    assert result.text == "Hello Kenya."
    assert result.confidence == 0.95
    assert result.provider_request_id == "job-123"
    assert result.duration_seconds == 2.4
    assert len(requests) == 3
    assert all(
        request.headers["authorization"] == "Bearer speech-test-secret"
        for request in requests
    )


def test_speechmatics_rejects_empty_transcript() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(201, json={"id": "job-empty"})
        if request.url.path.endswith("/job-empty"):
            return httpx.Response(200, json={"job": {"status": "done"}})
        return httpx.Response(200, json={"results": []})

    provider = SpeechmaticsBatchProvider(
        api_key="speech-test-secret",
        base_url="https://asr.test/v2",
        timeout_seconds=10,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(SpeechToTextError, match="empty transcript"):
        provider.transcribe(
            audio=b"mock-audio",
            filename="voice.ogg",
            mime_type="audio/ogg",
        )
