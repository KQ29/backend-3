from __future__ import annotations

from app.core.logging import get_logger
from app.providers.transport.base import DeliveryResult

logger = get_logger(__name__)


class LoggingTransport:
    """Records simulated delivery without logging message text or personal data."""

    def send_message(self, destination: str, message: str) -> DeliveryResult:
        logger.info(
            "Simulated transport accepted message",
            extra={
                "context": {
                    "session_id": destination,
                    "delivery": "simulated",
                    "message_length": len(message),
                }
            },
        )
        return DeliveryResult(
            delivered=True,
            simulated=True,
            provider_message_id=f"simulated:{destination}",
        )
