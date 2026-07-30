from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.interview.engine import InterviewEngine
from app.main import create_app
from app.repositories.memory import MemoryInterviewRepository
from app.services.interviews import InterviewService


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def repository() -> MemoryInterviewRepository:
    return MemoryInterviewRepository()


@pytest.fixture
def service(
    settings: Settings,
    repository: MemoryInterviewRepository,
) -> InterviewService:
    return InterviewService(repository, InterviewEngine(settings))


@pytest.fixture
def client(
    settings: Settings,
    repository: MemoryInterviewRepository,
) -> TestClient:
    with TestClient(
        create_app(settings=settings, repository=repository)
    ) as test_client:
        yield test_client


def start_consented_interview(service: InterviewService) -> str:
    started = service.start_interview()
    service.submit_consent(started.session_id, "consent_yes")
    return started.session_id


def complete_demographics(service: InterviewService, session_id: str) -> None:
    for answer in (
        "Kamau Otieno",
        "kamau.demo@example.com",
        "31",
        "Male",
        "Nairobi",
        "Westlands",
        "Digital marketing officer",
    ):
        service.submit_text(session_id, answer)
