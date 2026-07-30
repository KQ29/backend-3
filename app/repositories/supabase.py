from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from threading import RLock
from typing import Any, TypeVar
from uuid import uuid4

import httpx

from app.models.domain import (
    InterviewRecord,
    InterviewState,
    ResponseTag,
    SessionStatus,
    Turn,
)
from app.repositories.base import RecordNotFound

T = TypeVar("T")
RUNTIME_MARKER = "fastapi_streamlit_v2"


class SupabaseRepositoryError(RuntimeError):
    pass


class SupabaseRestClient:
    """Small server-side PostgREST client that never logs credentials."""

    def __init__(
        self,
        *,
        project_url: str,
        service_role_key: str,
        timeout_seconds: int = 20,
        client: httpx.Client | None = None,
    ) -> None:
        self.rest_url = f"{project_url.rstrip('/')}/rest/v1"
        self.service_role_key = service_role_key
        self.timeout_seconds = timeout_seconds
        self._client = client

    def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, str] | None = None,
        json_body: Any = None,
        prefer: str | None = None,
    ) -> Any:
        headers = {
            "apikey": self.service_role_key,
            "Content-Type": "application/json",
        }
        # Supabase's newer ``sb_secret_`` keys are opaque API keys, not JWTs.
        # Sending them as bearer tokens makes hosted projects reject the
        # request. Legacy service-role JWTs still require the bearer header.
        if not self.service_role_key.startswith("sb_secret_"):
            headers["Authorization"] = f"Bearer {self.service_role_key}"
        if prefer:
            headers["Prefer"] = prefer
        url = f"{self.rest_url}/{path.lstrip('/')}"
        try:
            if self._client is not None:
                response = self._client.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                )
            else:
                response = httpx.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
        except httpx.HTTPError as exc:
            raise SupabaseRepositoryError("Supabase request failed") from exc

        if response.is_error:
            safe_detail = self._safe_error_detail(response)
            suffix = f" ({safe_detail})" if safe_detail else ""
            raise SupabaseRepositoryError(
                f"Supabase request failed with status {response.status_code}{suffix}"
            )
        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except ValueError as exc:
            raise SupabaseRepositoryError(
                "Supabase returned a non-JSON response"
            ) from exc

    def _safe_error_detail(self, response: httpx.Response) -> str:
        try:
            payload = response.json()
        except ValueError:
            return ""
        if not isinstance(payload, dict):
            return ""

        parts: list[str] = []
        for key in ("code", "message", "hint"):
            value = payload.get(key)
            if not isinstance(value, str) or not value.strip():
                continue
            cleaned = " ".join(value.split())
            if self.service_role_key:
                cleaned = cleaned.replace(
                    self.service_role_key,
                    "[REDACTED]",
                )
            parts.append(cleaned[:240])
        return ": ".join(parts)


class SupabaseInterviewRepository:
    """Persists interview state, turns, and tags in the supplied schema.

    No schema changes are made. Extended state that has no dedicated legacy
    column is stored inside ``sessions.metadata.interview_state``.
    """

    def __init__(self, client: SupabaseRestClient) -> None:
        self.client = client
        self._lock = RLock()

    def create(self, record: InterviewRecord) -> InterviewRecord:
        with self._lock:
            protocol = self._active_protocol()
            state = record.state
            respondent_id = str(uuid4())
            anon_id = f"R-{uuid4().hex[:8]}"

            self.client.request(
                "POST",
                "respondents",
                json_body={
                    "id": respondent_id,
                    "phone": f"streamlit:{state.session_id}",
                    "cohort": "FastAPI Streamlit Demo",
                    "metadata": {
                        "demo": True,
                        "source": RUNTIME_MARKER,
                    },
                },
                prefer="return=minimal",
            )
            self.client.request(
                "POST",
                "respondent_anon_map",
                json_body={
                    "respondent_id": respondent_id,
                    "anon_id": anon_id,
                },
                prefer="return=minimal",
            )
            self.client.request(
                "POST",
                "sessions",
                json_body={
                    "id": state.session_id,
                    "respondent_id": respondent_id,
                    "protocol_id": protocol["id"],
                    "channel": "streamlit",
                    "status": self._database_status(state.status),
                    "consent_given": state.consent_given,
                    "last_activity_at": state.last_activity_at.isoformat(),
                    "metadata": self._state_metadata(state, revision=0),
                },
                prefer="return=minimal",
            )
            if record.turns:
                self.client.request(
                    "POST",
                    "turns",
                    json_body=[
                        self._turn_to_row(state.session_id, turn)
                        for turn in record.turns
                    ],
                    prefer="return=minimal",
                )
            return deepcopy(record)

    def get(self, session_id: str) -> InterviewRecord:
        rows = self.client.request(
            "GET",
            "sessions",
            params={
                "select": (
                    "id,status,consent_given,last_activity_at,nudge_sent_at,"
                    "completed_at,metadata"
                ),
                "id": f"eq.{session_id}",
                "limit": "1",
            },
        )
        if not rows:
            raise RecordNotFound(session_id)

        session = rows[0]
        metadata = session.get("metadata") or {}
        if metadata.get("runtime") != RUNTIME_MARKER:
            raise RecordNotFound(session_id)
        state_payload = metadata.get("interview_state")
        if not isinstance(state_payload, dict):
            raise SupabaseRepositoryError(
                "Supabase session is missing serialized interview state"
            )
        state = InterviewState.model_validate(state_payload)

        turn_rows = self.client.request(
            "GET",
            "turns",
            params={
                "select": (
                    "id,turn_number,role,content,question_id,input_mode,created_at"
                ),
                "session_id": f"eq.{session_id}",
                "order": "turn_number.asc",
            },
        )
        tag_rows = self.client.request(
            "GET",
            "response_tags",
            params={
                "select": (
                    "id,turn_id,question_id,source,raw_response,"
                    "economic_outcome,bottleneck_types,benefit_mechanism,"
                    "sentiment,confidence_in_tagging,"
                    "transcription_confidence,quotable_snippet,metadata"
                ),
                "session_id": f"eq.{session_id}",
                "order": "id.asc",
            },
        )
        return InterviewRecord(
            state=state,
            turns=[Turn.model_validate(row) for row in (turn_rows or [])],
            tags=[self._row_to_tag(row) for row in (tag_rows or [])],
            revision=int(metadata.get("revision", 0)),
        )

    def transact(
        self,
        session_id: str,
        operation: Callable[[InterviewRecord], T],
    ) -> tuple[InterviewRecord, T]:
        with self._lock:
            working = self.get(session_id)
            existing_turn_ids = {turn.id for turn in working.turns}
            existing_tag_ids = {tag.id for tag in working.tags}

            result = operation(working)
            working.revision += 1
            new_turns = [
                turn for turn in working.turns if turn.id not in existing_turn_ids
            ]
            new_tags = [tag for tag in working.tags if tag.id not in existing_tag_ids]

            if new_turns:
                self.client.request(
                    "POST",
                    "turns",
                    json_body=[
                        self._turn_to_row(session_id, turn) for turn in new_turns
                    ],
                    prefer="return=minimal",
                )
            if new_tags:
                self.client.request(
                    "POST",
                    "response_tags",
                    json_body=[self._tag_to_row(session_id, tag) for tag in new_tags],
                    prefer="return=minimal",
                )

            state = working.state
            self.client.request(
                "PATCH",
                "sessions",
                params={"id": f"eq.{session_id}"},
                json_body={
                    "status": self._database_status(state.status),
                    "consent_given": state.consent_given,
                    "last_activity_at": state.last_activity_at.isoformat(),
                    "nudge_sent_at": (
                        state.nudge_sent_at.isoformat() if state.nudge_sent_at else None
                    ),
                    "completed_at": (
                        state.completed_at.isoformat() if state.completed_at else None
                    ),
                    "metadata": self._state_metadata(
                        state,
                        revision=working.revision,
                    ),
                },
                prefer="return=minimal",
            )
            return deepcopy(working), result

    def list_records(self) -> list[InterviewRecord]:
        rows = self.client.request(
            "GET",
            "sessions",
            params={
                "select": "id,metadata",
                "status": "in.(consented,in_progress)",
                "metadata->>runtime": f"eq.{RUNTIME_MARKER}",
            },
        )
        records: list[InterviewRecord] = []
        for row in rows or []:
            try:
                records.append(self.get(row["id"]))
            except (RecordNotFound, SupabaseRepositoryError):
                continue
        return records

    def _active_protocol(self) -> dict[str, Any]:
        rows = self.client.request(
            "GET",
            "protocols",
            params={
                "select": "id,version,is_active",
                "is_active": "eq.true",
                "order": "version.desc",
                "limit": "1",
            },
        )
        if not rows:
            raise SupabaseRepositoryError(
                "No active interview protocol exists in Supabase"
            )
        return rows[0]

    @staticmethod
    def _database_status(status: SessionStatus) -> str:
        return {
            SessionStatus.AWAITING_CONSENT: "invited",
            SessionStatus.COLLECTING_DEMOGRAPHICS: "consented",
            SessionStatus.IN_PROGRESS: "in_progress",
            SessionStatus.COMPLETED: "completed",
            SessionStatus.DECLINED: "declined",
            SessionStatus.STOPPED: "completed",
            SessionStatus.ABANDONED: "abandoned",
        }[status]

    @staticmethod
    def _state_metadata(
        state: InterviewState,
        *,
        revision: int,
    ) -> dict[str, Any]:
        return {
            "runtime": RUNTIME_MARKER,
            "revision": revision,
            "interview_status": state.status.value,
            "interview_state": state.model_dump(mode="json"),
        }

    @staticmethod
    def _turn_to_row(session_id: str, turn: Turn) -> dict[str, Any]:
        return {
            "id": turn.id,
            "session_id": session_id,
            "turn_number": turn.turn_number,
            "role": turn.role.value,
            "content": turn.content,
            "question_id": (
                turn.question_id.value if turn.question_id is not None else None
            ),
            "input_mode": (
                turn.input_mode.value if turn.input_mode is not None else None
            ),
            "created_at": turn.created_at.isoformat(),
        }

    @staticmethod
    def _tag_to_row(
        session_id: str,
        tag: ResponseTag,
    ) -> dict[str, Any]:
        return {
            "id": tag.id,
            "session_id": session_id,
            "turn_id": tag.turn_id,
            "question_id": tag.question_id.value,
            "source": tag.source,
            "raw_response": tag.raw_response,
            "economic_outcome": tag.economic_outcome,
            "bottleneck_types": tag.bottleneck_types,
            "benefit_mechanism": tag.benefit_mechanism,
            "sentiment": tag.polarity.value if tag.polarity else None,
            "confidence_in_tagging": tag.confidence_in_tagging,
            "transcription_confidence": tag.transcription_confidence,
            "quotable_snippet": tag.quotable_snippet,
            "metadata": {
                "mixed_evidence": tag.mixed_evidence,
                "vague": tag.vague,
                "concrete": tag.concrete,
                "on_topic": tag.on_topic,
                "force_coded": tag.force_coded,
                "analysis_metadata": tag.metadata,
            },
        }

    @staticmethod
    def _row_to_tag(row: dict[str, Any]) -> ResponseTag:
        metadata = row.get("metadata") or {}
        return ResponseTag(
            id=row["id"],
            turn_id=row.get("turn_id"),
            question_id=row["question_id"],
            source=row.get("source", "live"),
            raw_response=row["raw_response"],
            polarity=row.get("sentiment"),
            mixed_evidence=bool(metadata.get("mixed_evidence", False)),
            confidence_in_tagging=float(row.get("confidence_in_tagging") or 0.5),
            vague=bool(metadata.get("vague", False)),
            concrete=bool(metadata.get("concrete", False)),
            on_topic=bool(metadata.get("on_topic", True)),
            economic_outcome=row.get("economic_outcome"),
            bottleneck_types=row.get("bottleneck_types") or [],
            benefit_mechanism=row.get("benefit_mechanism"),
            transcription_confidence=row.get("transcription_confidence"),
            quotable_snippet=row.get("quotable_snippet"),
            force_coded=bool(metadata.get("force_coded", True)),
            metadata=metadata.get("analysis_metadata") or {},
        )
