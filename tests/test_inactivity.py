from __future__ import annotations

from datetime import UTC, datetime, timedelta

from app.core.config import Settings
from app.models.domain import InterviewRecord, SessionStatus
from app.providers.transport.logging import LoggingTransport
from app.repositories.memory import MemoryInterviewRepository
from app.services.interviews import InterviewService
from app.workers.inactivity import run_inactivity_pass
from tests.conftest import start_consented_interview


def set_last_activity(
    repository: MemoryInterviewRepository,
    session_id: str,
    timestamp: datetime,
) -> None:
    def update(record: InterviewRecord) -> None:
        record.state.last_activity_at = timestamp

    repository.transact(session_id, update)


def test_nudge_is_sent_once_during_demographics(
    service: InterviewService,
    repository: MemoryInterviewRepository,
    settings: Settings,
) -> None:
    session_id = start_consented_interview(service)
    now = datetime.now(UTC)
    set_last_activity(repository, session_id, now - timedelta(hours=11))

    first = run_inactivity_pass(
        repository,
        LoggingTransport(),
        settings,
        now=now,
    )
    second = run_inactivity_pass(
        repository,
        LoggingTransport(),
        settings,
        now=now + timedelta(minutes=5),
    )

    state = service.get_state(session_id)
    assert first.nudged == 1
    assert second.nudged == 0
    assert state.nudge_delivery == "simulated"
    assert state.nudge_sent_at is not None


def test_session_is_abandoned_after_24_hours(
    service: InterviewService,
    repository: MemoryInterviewRepository,
    settings: Settings,
) -> None:
    session_id = start_consented_interview(service)
    now = datetime.now(UTC)
    set_last_activity(repository, session_id, now - timedelta(hours=25))

    report = run_inactivity_pass(
        repository,
        LoggingTransport(),
        settings,
        now=now,
    )

    assert report.abandoned == 1
    assert service.get_state(session_id).status == SessionStatus.ABANDONED
