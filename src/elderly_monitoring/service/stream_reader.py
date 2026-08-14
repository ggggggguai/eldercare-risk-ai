from __future__ import annotations

import json
import os
import select
import subprocess
import time
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


class FFmpegStreamReader:
    """Decode a stream through the system FFmpeg binary into BGR frames."""

    def __init__(
        self,
        stream_url: str,
        *,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
        scale_width: int = 1280,
        max_fps: float | None = None,
        ffmpeg_binary: str = "ffmpeg",
        ffprobe_binary: str = "ffprobe",
    ) -> None:
        if open_timeout_ms <= 0 or read_timeout_ms <= 0:
            raise ValueError("stream timeouts must be positive")
        if scale_width < 2:
            raise ValueError("ffmpeg scale_width must be at least 2")
        if max_fps is not None and max_fps <= 0:
            raise ValueError("ffmpeg max_fps must be positive")
        self.stream_url = stream_url
        self.open_timeout_ms = int(open_timeout_ms)
        self.read_timeout_ms = int(read_timeout_ms)
        self.scale_width = int(scale_width)
        self.max_fps = float(max_fps) if max_fps is not None else None
        self.ffmpeg_binary = ffmpeg_binary
        self.ffprobe_binary = ffprobe_binary
        self.process: subprocess.Popen[bytes] | None = None
        self.source_pts_sec: float | None = None
        self._frame_height: int | None = None
        self._frame_size: int | None = None
        self._prefetched_frame: Any | None = None

    def open(self) -> None:
        self.release()
        source_width, source_height = self._probe_dimensions()
        self._frame_height = max(
            1, int(round(source_height * self.scale_width / source_width))
        )
        self._frame_size = self.scale_width * self._frame_height * 3
        filters = [f"scale={self.scale_width}:{self._frame_height}"]
        if self.max_fps is not None:
            interval_sec = 1.0 / self.max_fps
            filters.insert(
                0,
                "select='isnan(prev_selected_t)+"
                f"gte(t-prev_selected_t\\,{interval_sec:.9f})'",
            )
        command = [
            self.ffmpeg_binary,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            self.stream_url,
            "-map",
            "0:v:0",
            "-an",
            "-sn",
            "-dn",
            "-vf",
            ",".join(filters),
            "-pix_fmt",
            "bgr24",
            "-f",
            "rawvideo",
            "pipe:1",
        ]
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
        except OSError as exc:
            raise RuntimeError(f"ffmpeg could not be started: {exc}") from exc
        self.process = process
        payload, status = self._read_frame_bytes(
            process, timeout_sec=self.open_timeout_ms / 1000.0
        )
        if status != "frame":
            detail = self._stderr_text(process)
            self.release()
            if status == "timeout":
                raise RuntimeError("ffmpeg timed out before first frame")
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"ffmpeg exited before first frame{suffix}")
        self._prefetched_frame = self._decode_frame(payload)

    def read(self) -> Any | None:
        if self._prefetched_frame is not None:
            frame = self._prefetched_frame
            self._prefetched_frame = None
            return frame
        process = self.process
        if process is None:
            return None
        payload, status = self._read_frame_bytes(
            process, timeout_sec=self.read_timeout_ms / 1000.0
        )
        if status != "frame":
            self.release()
            return None
        return self._decode_frame(payload)

    def update_url(self, stream_url: str) -> None:
        self.release()
        self.stream_url = stream_url

    def release(self) -> None:
        process = self.process
        self.process = None
        self._prefetched_frame = None
        self.source_pts_sec = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    pass
        for pipe in (process.stdout, process.stderr):
            if pipe is not None:
                pipe.close()

    def _probe_dimensions(self) -> tuple[int, int]:
        command = [
            self.ffprobe_binary,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_type,codec_name,width,height",
            "-of",
            "json",
            self.stream_url,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                check=False,
                timeout=self.open_timeout_ms / 1000.0,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ffprobe timed out while opening stream") from exc
        except OSError as exc:
            raise RuntimeError(f"ffprobe could not be started: {exc}") from exc
        if completed.returncode != 0:
            detail = self._sanitize_error_text(
                completed.stderr.decode("utf-8", errors="replace").strip()
            )
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"ffprobe failed{suffix}")
        try:
            payload = json.loads(completed.stdout.decode("utf-8"))
            stream = next(
                row
                for row in payload["streams"]
                if row.get("codec_type", "video") == "video"
            )
            width = int(stream["width"])
            height = int(stream["height"])
        except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as exc:
            raise RuntimeError("ffprobe returned invalid video dimensions") from exc
        if width <= 0 or height <= 0:
            raise RuntimeError("ffprobe returned non-positive video dimensions")
        return width, height

    def _read_frame_bytes(
        self, process: subprocess.Popen[bytes], *, timeout_sec: float
    ) -> tuple[bytes, str]:
        if process.stdout is None or self._frame_size is None:
            return b"", "eof"
        deadline = time.monotonic() + timeout_sec
        chunks = bytearray()
        descriptor = process.stdout.fileno()
        while len(chunks) < self._frame_size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return bytes(chunks), "timeout"
            try:
                ready, _, _ = select.select([descriptor], [], [], remaining)
                if not ready:
                    return bytes(chunks), "timeout"
                block = os.read(descriptor, self._frame_size - len(chunks))
            except (OSError, ValueError):
                # A concurrent session stop or URL update may close the pipe.
                return bytes(chunks), "eof"
            if not block:
                return bytes(chunks), "eof"
            chunks.extend(block)
        return bytes(chunks), "frame"

    def _decode_frame(self, payload: bytes) -> Any:
        import numpy as np

        assert self._frame_height is not None
        return np.frombuffer(payload, dtype=np.uint8).reshape(
            self._frame_height, self.scale_width, 3
        ).copy()

    def _stderr_text(self, process: subprocess.Popen[bytes]) -> str:
        if process.poll() is None or process.stderr is None:
            return ""
        return self._sanitize_error_text(
            process.stderr.read().decode("utf-8", errors="replace").strip()
        )

    def _sanitize_error_text(self, detail: str) -> str:
        if not detail:
            return ""
        return detail.replace(self.stream_url, "<redacted-stream-url>")
