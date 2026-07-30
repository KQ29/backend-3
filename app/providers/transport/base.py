from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class DeliveryResult:
    delivered: bool
    simulated: bool
    provider_message_id: str | None = None
    error: str | None = None


class ChannelTransport(Protocol):
    def send_message(self, destination: str, message: str) -> DeliveryResult: ...
