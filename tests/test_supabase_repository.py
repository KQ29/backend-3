from __future__ import annotations

from copy import deepcopy
from typing import Any

import httpx
import pytest

from app.core.config import Settings
from app.interview.engine import InterviewEngine
from app.models.domain import InterviewState, QuestionId, SessionStatus
from app.repositories.supabase import (
    SupabaseInterviewRepository,
    SupabaseRepositoryError,
    SupabaseRestClient,
)
from app.services.interviews import InterviewService
from tests.conftest import complete_demographics


class FakeSupabaseRestClient:
    def __init__(self) -> None:
        self.tables: dict[str, list[dict[str, Any]]] = {
            "protocols": [
                {
                    "id": "11111111-1111-1111-1111-111111111111",
                    "version": "test-v1",
                    "is_active": True,
                }
            ],
            "respondents": [],
            "respondent_anon_map": [],
            "sessions": [],
            "turns": [],
            "response_tags": [],
        }
        self.methods: list[str] = []

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: Any = None,
        prefer: str | None = None,
    ) -> Any:
        del prefer
        self.methods.append(method)
        table = path.strip("/")
        if method == "POST":
            rows = json_body if isinstance(json_body, list) else [json_body]
            self.tables[table].extend(deepcopy(rows))
            return None
        if method == "PATCH":
            for row in self._filtered(table, params or {}):
                row.update(deepcopy(json_body))
            return None
        if method == "GET":
            rows = self._filtered(table, params or {})
            order = (params or {}).get("order")
            if order:
                field, direction = order.split(".", maxsplit=1)
                rows.sort(
                    key=lambda row: row.get(field) or "",
                    reverse=direction == "desc",
                )
            limit = int((params or {}).get("limit", len(rows)))
            return deepcopy(rows[:limit])
        raise AssertionError(f"Unexpected method: {method}")

    def _filtered(
        self,
        table: str,
        params: dict[str, str],
    ) -> list[dict[str, Any]]:
        rows = self.tables[table]
        result: list[dict[str, Any]] = []
        for row in rows:
            include = True
            for key, expression in params.items():
                if key in {"select", "order", "limit"}:
                    continue
                if key == "metadata->>runtime":
                    expected = expression.removeprefix("eq.")
                    include = (row.get("metadata") or {}).get("runtime") == expected
                elif expression.startswith("eq."):
                    expected = expression.removeprefix("eq.")
                    actual = row.get(key)
                    if isinstance(actual, bool):
                        include = str(actual).lower() == expected
                    else:
                        include = str(actual) == expected
                elif expression.startswith("in.("):
                    allowed = expression[4:-1].split(",")
                    include = str(row.get(key)) in allowed
                if not include:
                    break
            if include:
                result.append(row)
        return result


def test_supabase_repository_persists_state_turns_and_tags() -> None:
    rest = FakeSupabaseRestClient()
    repository = SupabaseInterviewRepository(rest)  # type: ignore[arg-type]
    service = InterviewService(repository, InterviewEngine(Settings()))

    started = service.start_interview()
    session_id = started.session_id
    service.submit_consent(session_id, "consent_yes")
    complete_demographics(service, session_id)
    service.submit_text(
        session_id,
        "My income and job role have not changed since the AI training.",
    )
    state = service.get_state(session_id)

    assert state.status == SessionStatus.IN_PROGRESS
    assert state.question_id == QuestionId.ANCHOR_2
    assert len(rest.tables["respondents"]) == 1
    assert rest.tables["respondents"][0]["phone"].startswith("streamlit:")
    assert len(rest.tables["respondent_anon_map"]) == 1
    assert len(rest.tables["sessions"]) == 1
    assert len(rest.tables["turns"]) == len(state.transcript)
    assert len(rest.tables["response_tags"]) == 1
    assert rest.tables["sessions"][0]["metadata"]["runtime"] == ("fastapi_streamlit_v2")
    assert "DELETE" not in rest.methods


def test_supabase_round_trips_state_while_second_probe_is_pending() -> None:
    rest = FakeSupabaseRestClient()
    repository = SupabaseInterviewRepository(rest)  # type: ignore[arg-type]
    service = InterviewService(
        repository,
        InterviewEngine(Settings(max_probes_per_anchor=2)),
    )

    started = service.start_interview()
    session_id = started.session_id
    service.submit_consent(session_id, "consent_yes")
    complete_demographics(service, session_id)

    first_probe = service.submit_text(session_id, "I use AI for emails.")
    first_pending = service.get_state(session_id)

    assert first_probe.question_id == QuestionId.ANCHOR_1_PROBE
    assert first_probe.probe_number == 1
    assert first_pending.question_id == QuestionId.ANCHOR_1_PROBE
    assert first_pending.probes_used == [QuestionId.ANCHOR_1_PROBE]
    assert first_pending.probe_counts == {"anchor_1": 1}

    second_probe = service.submit_text(session_id, "It helps with work.")
    second_pending = service.get_state(session_id)
    stored_state = rest.tables["sessions"][0]["metadata"]["interview_state"]

    assert second_probe.question_id == QuestionId.ANCHOR_1_PROBE
    assert second_probe.probe_number == 2
    assert second_probe.probe_counts == {"anchor_1": 2}
    assert second_pending.question_id == QuestionId.ANCHOR_1_PROBE
    assert second_pending.probes_used == [
        QuestionId.ANCHOR_1_PROBE,
        QuestionId.ANCHOR_1_PROBE,
    ]
    assert second_pending.probe_counts == {"anchor_1": 2}
    assert second_pending.max_probes_per_anchor == 2
    assert stored_state["max_probes_per_anchor"] == 2
    assert stored_state["probe_counts"] == {"anchor_1": 2}
    assert stored_state["probe_questions"] == {
        "anchor_1": [
            "Can you give one recent, specific example of what you mean?",
            "What was the result in that specific example?",
        ]
    }
    assert stored_state["resume_question_id"] == "anchor_2"
    analysis_metadata = rest.tables["response_tags"][1]["metadata"][
        "analysis_metadata"
    ]
    assert analysis_metadata["needs_probe"] is True
    assert analysis_metadata["probe_type"] == "clarity"
    assert analysis_metadata["probe_reason"] == "vague_or_unclear"
    assert analysis_metadata["probe_asked"] is True
    assert analysis_metadata["probe_number"] == 2

    resumed = service.submit_text(
        session_id,
        "Yesterday AI drafted a customer email that I reviewed before sending.",
    )
    stored_state = rest.tables["sessions"][0]["metadata"]["interview_state"]

    assert resumed.question_id == QuestionId.ANCHOR_2
    assert stored_state["resume_question_id"] is None
    assert stored_state["probe_counts"] == {"anchor_1": 2}
    assert service.get_state(session_id).probes_used == [
        QuestionId.ANCHOR_1_PROBE,
        QuestionId.ANCHOR_1_PROBE,
    ]


def test_legacy_interview_state_defaults_to_one_probe_and_hydrates_count() -> None:
    state = InterviewState.model_validate(
        {
            "probes_used": [QuestionId.ANCHOR_2_PROBE],
        }
    )

    assert state.max_probes_per_anchor == 1
    assert state.probe_counts == {"anchor_2": 1}


def test_supabase_rest_client_keeps_secret_out_of_errors() -> None:
    secret = "service-role-test-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["apikey"] == secret
        assert request.headers["authorization"] == f"Bearer {secret}"
        return httpx.Response(500, text=f"server echoed {secret}")

    client = SupabaseRestClient(
        project_url="https://project.test",
        service_role_key=secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(SupabaseRepositoryError) as captured:
        client.request("GET", "protocols")
    assert secret not in str(captured.value)


def test_supabase_rest_client_uses_opaque_secret_only_as_api_key() -> None:
    secret = "sb_secret_test-only-placeholder"

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["apikey"] == secret
        assert "authorization" not in request.headers
        return httpx.Response(200, json=[])

    client = SupabaseRestClient(
        project_url="https://project.test",
        service_role_key=secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    assert client.request("GET", "protocols") == []


def test_supabase_rest_client_reports_safe_schema_error_detail() -> None:
    secret = "service-role-test-secret"

    def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            400,
            json={
                "code": "PGRST204",
                "message": (f"Could not find the demographics column; token={secret}"),
                "details": "A potentially sensitive row is deliberately omitted",
                "hint": "Refresh the schema cache",
            },
        )

    client = SupabaseRestClient(
        project_url="https://project.test",
        service_role_key=secret,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    with pytest.raises(SupabaseRepositoryError) as captured:
        client.request("POST", "sessions", json_body={"id": "example"})

    message = str(captured.value)
    assert "PGRST204" in message
    assert "demographics column" in message
    assert "Refresh the schema cache" in message
    assert "potentially sensitive row" not in message
    assert secret not in message
