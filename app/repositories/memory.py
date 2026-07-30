from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from threading import RLock
from typing import TypeVar

from app.models.domain import InterviewRecord
from app.repositories.base import RecordNotFound

T = TypeVar("T")


class MemoryInterviewRepository:
    """Thread-safe in-memory repository used only for the local demo and tests."""

    def __init__(self) -> None:
        self._records: dict[str, InterviewRecord] = {}
        self._lock = RLock()

    def create(self, record: InterviewRecord) -> InterviewRecord:
        with self._lock:
            session_id = record.state.session_id
            if session_id in self._records:
                raise ValueError(f"Session already exists: {session_id}")
            self._records[session_id] = deepcopy(record)
            return deepcopy(record)

    def get(self, session_id: str) -> InterviewRecord:
        with self._lock:
            if session_id not in self._records:
                raise RecordNotFound(session_id)
            return deepcopy(self._records[session_id])

    def transact(
        self,
        session_id: str,
        operation: Callable[[InterviewRecord], T],
    ) -> tuple[InterviewRecord, T]:
        with self._lock:
            if session_id not in self._records:
                raise RecordNotFound(session_id)
            working = deepcopy(self._records[session_id])
            result = operation(working)
            working.revision += 1
            self._records[session_id] = deepcopy(working)
            return deepcopy(working), result

    def list_records(self) -> list[InterviewRecord]:
        with self._lock:
            return [deepcopy(record) for record in self._records.values()]
