from __future__ import annotations

from datetime import UTC, datetime, timedelta

from pydantic import BaseModel

from app.core.config import Settings
from app.models.domain import InterviewRecord, SessionStatus
from app.providers.transport.base import ChannelTransport
from app.repositories.base import InterviewRepository

NUDGE_MESSAGE = (
    "Sawa. Your interview is still open. Continue whenever you are ready. "
    "You can reply by text or voice note, or say 'stop' at any time."
)


class InactivityReport(BaseModel):
    checked: int = 0
    nudged: int = 0
    abandoned: int = 0
    delivery_failures: int = 0


def run_inactivity_pass(
    repository: InterviewRepository,
    transport: ChannelTransport,
    settings: Settings,
    *,
    now: datetime | None = None,
) -> InactivityReport:
    current_time = now or datetime.now(UTC)
    nudge_cutoff = current_time - timedelta(hours=settings.nudge_after_hours)
    abandon_cutoff = current_time - timedelta(hours=settings.abandon_after_hours)
    report = InactivityReport()

    for snapshot in repository.list_records():
        report.checked += 1
        state = snapshot.state
        if state.status not in {
            SessionStatus.COLLECTING_DEMOGRAPHICS,
            SessionStatus.IN_PROGRESS,
        }:
            continue

        if state.last_activity_at <= abandon_cutoff:

            def abandon(record: InterviewRecord) -> None:
                if record.state.status in {
                    SessionStatus.COLLECTING_DEMOGRAPHICS,
                    SessionStatus.IN_PROGRESS,
                }:
                    record.state.status = SessionStatus.ABANDONED
                    record.state.completed_at = current_time

            repository.transact(state.session_id, abandon)
            report.abandoned += 1
            continue

        if state.last_activity_at <= nudge_cutoff and state.nudge_sent_at is None:
            delivery = transport.send_message(state.session_id, NUDGE_MESSAGE)
            if not delivery.delivered:
                report.delivery_failures += 1
                continue
            delivery_mode = "simulated" if delivery.simulated else "production"

            def mark_nudged(
                record: InterviewRecord,
                delivery_mode: str = delivery_mode,
            ) -> None:
                if (
                    record.state.status
                    in {
                        SessionStatus.COLLECTING_DEMOGRAPHICS,
                        SessionStatus.IN_PROGRESS,
                    }
                    and record.state.nudge_sent_at is None
                ):
                    record.state.nudge_sent_at = current_time
                    record.state.nudge_delivery = delivery_mode

            repository.transact(state.session_id, mark_nudged)
            report.nudged += 1

    return report
