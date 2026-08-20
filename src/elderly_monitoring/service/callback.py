from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import httpx

from elderly_monitoring.common.schemas import AlgorithmEvent

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeliveryAttempt:
    success: bool
    status_code: int | None = None
    error: str | None = None


class CallbackSender:
    def __init__(self, *, token: str, client: httpx.Client | None = None, timeout: float = 5.0, retry_delays: Iterable[float] = (0.5, 1.0, 2.0)) -> None:
        self.token = token
        self.client = client or httpx.Client(timeout=timeout, trust_env=False)
        self.retry_delays = tuple(retry_delays)
        self._owns_client = client is None

    def send(self, callback_url: str, event: AlgorithmEvent, *, session_id: str) -> bool:
        event_id = str(uuid.uuid4())
        payload = event.to_dict()
        payload.update({"event_id": event_id, "session_id": session_id, "schema_version": "1.0"})
        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        for attempt, delay in enumerate(self.retry_delays, start=1):
            result = self.send_once(callback_url, payload)
            if result.success:
                return True
            if attempt < len(self.retry_delays):
                time.sleep(delay)
        logger.error("risk event callback failed after retries")
        return False

    def send_once(self, callback_url: str, payload: Mapping[str, Any]) -> DeliveryAttempt:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        try:
            response = self.client.post(callback_url, json=dict(payload), headers=headers)
        except httpx.HTTPError as exc:
            return DeliveryAttempt(
                success=False,
                error=f"{type(exc).__name__}: callback request failed",
            )
        if 200 <= response.status_code < 300:
            return DeliveryAttempt(success=True, status_code=response.status_code)
        return DeliveryAttempt(
            success=False,
            status_code=response.status_code,
            error=f"callback_http_status_{response.status_code}",
        )

    def close(self) -> None:
        if self._owns_client:
            self.client.close()
