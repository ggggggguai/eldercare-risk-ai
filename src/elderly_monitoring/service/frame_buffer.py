from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FramePacket:
    frame: Any
    stream_epoch: int
    source_pts_sec: float | None
    received_monotonic_sec: float


class LatestFrameBuffer:
    """Bounded producer/consumer buffer that drops the oldest queued frame."""

    def __init__(self, *, capacity: int = 2) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self.capacity = capacity
        self._items: deque[FramePacket] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._close_reason: str | None = None
        self._put_count = 0
        self._get_count = 0
        self._dropped_oldest = 0
        self._last_dropped_source_pts_sec: float | None = None

    def put(self, packet: FramePacket) -> bool:
        with self._condition:
            if self._closed:
                return False
            if len(self._items) >= self.capacity:
                dropped = self._items.popleft()
                self._dropped_oldest += 1
                self._last_dropped_source_pts_sec = dropped.source_pts_sec
            self._items.append(packet)
            self._put_count += 1
            self._condition.notify()
            return True

    def get(self, *, timeout: float | None = None) -> FramePacket | None:
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._condition:
            while not self._items and not self._closed:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    return None
                self._condition.wait(remaining)
            if not self._items:
                return None
            self._get_count += 1
            return self._items.popleft()

    def close(self, *, reason: str) -> None:
        with self._condition:
            self._closed = True
            self._close_reason = reason
            self._condition.notify_all()

    def discard_pending(self, *, reason: str) -> int:
        with self._condition:
            discarded = len(self._items)
            if self._items:
                self._last_dropped_source_pts_sec = self._items[-1].source_pts_sec
            self._items.clear()
            self._dropped_oldest += discarded
            self._close_reason = reason
            return discarded

    def snapshot(self) -> dict[str, Any]:
        with self._condition:
            return {
                "capacity": self.capacity,
                "depth": len(self._items),
                "closed": self._closed,
                "close_reason": self._close_reason,
                "put_count": self._put_count,
                "get_count": self._get_count,
                "dropped_oldest": self._dropped_oldest,
                "last_dropped_source_pts_sec": self._last_dropped_source_pts_sec,
            }
