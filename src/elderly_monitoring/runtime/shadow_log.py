"""Bounded non-blocking JSONL logging for shadow environment features."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import queue
import threading
from typing import Any, Mapping


@dataclass(frozen=True)
class ShadowLogStats:
    enqueued: int
    written: int
    dropped: int
    errors: int


class ShadowJsonlLogger:
    """A small daemon writer; callers never wait on disk I/O."""

    def __init__(self, path: str | Path, *, capacity: int = 256) -> None:
        self.path = Path(path)
        if int(capacity) < 1:
            raise ValueError("shadow logger capacity must be positive")
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=int(capacity))
        self._lock = threading.Lock()
        self._stats = {"enqueued": 0, "written": 0, "dropped": 0, "errors": 0}
        self._closed = False
        self._thread = threading.Thread(target=self._run, daemon=True, name="environment-shadow-log")
        self._thread.start()

    def log(self, payload: Mapping[str, Any]) -> bool:
        item = dict(payload)
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            with self._lock:
                self._stats["dropped"] += 1
            return False
        with self._lock:
            self._stats["enqueued"] += 1
        return True

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def close(self, *, timeout: float = 1.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            with self._lock:
                self._stats["dropped"] += 1
        self._thread.join(timeout=max(0.0, float(timeout)))

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                while True:
                    item = self._queue.get()
                    if item is None:
                        return
                    try:
                        handle.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
                        handle.flush()
                    except Exception:
                        with self._lock:
                            self._stats["errors"] += 1
                    else:
                        with self._lock:
                            self._stats["written"] += 1
        except Exception:
            with self._lock:
                self._stats["errors"] += 1
