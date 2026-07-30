from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, TypeVar

from app.models.domain import InterviewRecord


class RecordNotFound(KeyError):
    pass


T = TypeVar("T")


class InterviewRepository(Protocol):
    def create(self, record: InterviewRecord) -> InterviewRecord: ...

    def get(self, session_id: str) -> InterviewRecord: ...

    def transact(
        self,
        session_id: str,
        operation: Callable[[InterviewRecord], T],
    ) -> tuple[InterviewRecord, T]: ...

    def list_records(self) -> list[InterviewRecord]: ...
