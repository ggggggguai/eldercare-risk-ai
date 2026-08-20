from __future__ import annotations

from typing import Any


class StreamReader:
    def __init__(self, stream_url: str, *, open_timeout_ms: int = 5000, read_timeout_ms: int = 5000) -> None:
        self.stream_url = stream_url
        self.open_timeout_ms = open_timeout_ms
        self.read_timeout_ms = read_timeout_ms
        self.capture: Any | None = None
        self.source_pts_sec: float | None = None

    def open(self) -> None:
        import cv2

        self.capture = cv2.VideoCapture()
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            self.capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms)
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            self.capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.read_timeout_ms)
        self.capture.open(self.stream_url)
        if not self.capture.isOpened():
            self.release()
            raise RuntimeError("stream could not be opened")

    def read(self) -> Any | None:
        if self.capture is None:
            return None
        ok, frame = self.capture.read()
        if ok:
            self.source_pts_sec = self._read_source_pts_sec()
        return frame if ok else None

    def _read_source_pts_sec(self) -> float | None:
        import cv2

        value = float(self.capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
        return value if value >= 0.0 else None

    def update_url(self, stream_url: str) -> None:
        self.release()
        self.stream_url = stream_url

    def release(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.source_pts_sec = None
