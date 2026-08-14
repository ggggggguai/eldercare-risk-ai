from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from elderly_monitoring.service.stream_reader import FFmpegStreamReader

# 假 ffprobe：返回固定分辨率；FAKE_FFPROBE_MODE=fail 时模拟探测失败。
_FAKE_FFPROBE = '''\
#!/usr/bin/env python3
import json
import os
import sys
if os.environ.get("FAKE_ARGV_FILE"):
    with open(os.environ["FAKE_ARGV_FILE"], "a") as handle:
        handle.write("ffprobe:" + "|".join(sys.argv) + "\\n")
if os.environ.get("FAKE_FFPROBE_MODE") == "fail":
    sys.stderr.write("fake ffprobe failure: " + sys.argv[-1] + "\\n")
    sys.exit(1)
width = int(os.environ.get("FAKE_WIDTH", "1280"))
height = int(os.environ.get("FAKE_HEIGHT", "720"))
print(json.dumps({"streams": [{"codec_type": "video", "codec_name": "h264",
                               "width": width, "height": height}]}))
'''

# 假 ffmpeg：从 -vf 过滤器解析 scale=WxH 输出原始 BGR 帧。
# FAKE_FFMPEG_MODE=fail 立即失败；=hang 永不输出；=stall_after_frames 输出
# FAKE_FRAMES 帧后挂起；默认输出 FAKE_FRAMES 帧后正常退出。
_FAKE_FFMPEG = '''\
#!/usr/bin/env python3
import os
import sys
import time
if os.environ.get("FAKE_ARGV_FILE"):
    with open(os.environ["FAKE_ARGV_FILE"], "a") as handle:
        handle.write("ffmpeg:" + "|".join(sys.argv) + "\\n")
mode = os.environ.get("FAKE_FFMPEG_MODE", "ok")
if mode == "fail":
    sys.stderr.write("fake ffmpeg failure\\n")
    sys.exit(1)
frame_bytes = None
for arg in sys.argv:
    if "scale=" in arg:
        spec = arg.split("scale=", 1)[1]
        width, height = (int(value) for value in spec.split(":"))
        frame_bytes = width * height * 3
if frame_bytes is None:
    width = int(os.environ.get("FAKE_WIDTH", "1280"))
    height = int(os.environ.get("FAKE_HEIGHT", "720"))
    frame_bytes = width * height * 3
if mode == "hang":
    time.sleep(3600)
frame = b"\\x00" * frame_bytes
count = int(os.environ.get("FAKE_FRAMES", "5"))
if mode == "stall_after_frames":
    for _ in range(count):
        sys.stdout.buffer.write(frame)
        sys.stdout.buffer.flush()
    time.sleep(3600)
for _ in range(count):
    sys.stdout.buffer.write(frame)
    sys.stdout.buffer.flush()
sys.exit(0)
'''

_ENV_KEYS = (
    "FAKE_WIDTH",
    "FAKE_HEIGHT",
    "FAKE_FRAMES",
    "FAKE_FFMPEG_MODE",
    "FAKE_FFPROBE_MODE",
    "FAKE_ARGV_FILE",
)


class FFmpegStreamReaderTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._bin = Path(self._tmp.name)
        self.ffmpeg_bin = self._write_tool("fake_ffmpeg", _FAKE_FFMPEG)
        self.ffprobe_bin = self._write_tool("fake_ffprobe", _FAKE_FFPROBE)
        os.environ["FAKE_WIDTH"] = "64"
        os.environ["FAKE_HEIGHT"] = "36"
        for key in _ENV_KEYS:
            if key not in {"FAKE_WIDTH", "FAKE_HEIGHT"}:
                os.environ.pop(key, None)

    def tearDown(self) -> None:
        for key in _ENV_KEYS:
            os.environ.pop(key, None)
        self._tmp.cleanup()

    def _write_tool(self, name: str, source: str) -> Path:
        path = self._bin / name
        path.write_text(source, encoding="utf-8")
        os.chmod(path, 0o755)
        return path

    def _reader(self, **kwargs: object) -> FFmpegStreamReader:
        options = {
            "open_timeout_ms": 2000,
            "read_timeout_ms": 2000,
            "scale_width": 32,
            "ffmpeg_binary": str(self.ffmpeg_bin),
            "ffprobe_binary": str(self.ffprobe_bin),
        }
        options.update(kwargs)
        return FFmpegStreamReader("rtmp://fake.invalid/live", **options)

    def test_open_read_and_eof(self) -> None:
        os.environ["FAKE_FRAMES"] = "5"
        reader = self._reader()
        reader.open()
        self.assertIsNone(reader.source_pts_sec)
        frames = []
        while True:
            frame = reader.read()
            if frame is None:
                break
            frames.append(frame)
        self.assertEqual(len(frames), 5)
        self.assertEqual(frames[0].shape, (18, 32, 3))
        reader.release()

    def test_open_ffprobe_failure_raises_quickly(self) -> None:
        os.environ["FAKE_FFPROBE_MODE"] = "fail"
        secret = "review-secret-token"
        reader = FFmpegStreamReader(
            f"https://camera.example/private/live.flv?accessToken={secret}",
            open_timeout_ms=5000,
            read_timeout_ms=2000,
            scale_width=32,
            ffmpeg_binary=str(self.ffmpeg_bin),
            ffprobe_binary=str(self.ffprobe_bin),
        )
        started = time.monotonic()
        with self.assertRaises(RuntimeError) as context:
            reader.open()
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertIn("ffprobe", str(context.exception))
        self.assertNotIn(secret, str(context.exception))
        self.assertNotIn("/private/live.flv", str(context.exception))

    def test_max_fps_filter_only_selects_original_frames(self) -> None:
        os.environ["FAKE_ARGV_FILE"] = str(self._bin / "argv.log")
        reader = self._reader(max_fps=8.0)
        reader.open()
        reader.release()

        log = (self._bin / "argv.log").read_text(encoding="utf-8")
        ffmpeg_line = next(line for line in log.splitlines() if line.startswith("ffmpeg:"))
        self.assertIn("select=", ffmpeg_line)
        self.assertNotIn("fps=8", ffmpeg_line)

    def test_open_ffmpeg_failure_raises(self) -> None:
        os.environ["FAKE_FFMPEG_MODE"] = "fail"
        reader = self._reader()
        with self.assertRaises(RuntimeError) as context:
            reader.open()
        self.assertIn("before first frame", str(context.exception))
        self.assertIsNone(reader.process)

    def test_open_timeout_raises_and_releases(self) -> None:
        os.environ["FAKE_FFMPEG_MODE"] = "hang"
        reader = self._reader(open_timeout_ms=300, read_timeout_ms=300)
        started = time.monotonic()
        with self.assertRaises(RuntimeError):
            reader.open()
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertIsNone(reader.process)

    def test_read_timeout_returns_none_and_kills(self) -> None:
        os.environ["FAKE_FFMPEG_MODE"] = "stall_after_frames"
        os.environ["FAKE_FRAMES"] = "2"
        reader = self._reader(read_timeout_ms=300)
        reader.open()
        self.assertIsNotNone(reader.read())
        self.assertIsNotNone(reader.read())
        self.assertIsNone(reader.read())
        self.assertIsNone(reader.process)

    def test_release_kills_hung_process_and_is_idempotent(self) -> None:
        os.environ["FAKE_FFMPEG_MODE"] = "stall_after_frames"
        os.environ["FAKE_FRAMES"] = "1"
        reader = self._reader()
        reader.open()
        self.assertIsNotNone(reader.read())
        started = time.monotonic()
        reader.release()
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertIsNone(reader.process)
        reader.release()

    def test_update_url_releases_and_reopens_new_url(self) -> None:
        os.environ["FAKE_ARGV_FILE"] = str(self._bin / "argv.log")
        reader = self._reader()
        reader.open()
        reader.read()
        reader.update_url("rtmp://fake.invalid/second")
        self.assertEqual(reader.stream_url, "rtmp://fake.invalid/second")
        self.assertIsNone(reader.process)
        reader.open()
        log = (self._bin / "argv.log").read_text(encoding="utf-8")
        self.assertIn("rtmp://fake.invalid/live", log)
        self.assertIn("rtmp://fake.invalid/second", log)
        reader.release()

    @unittest.skipUnless(shutil.which("ffmpeg"), "system ffmpeg required")
    def test_integration_decodes_hevc_flv(self) -> None:
        """HEVC-over-FLV regression: the bundled OpenCV FFmpeg fails on this,
        the ffmpeg backend must succeed."""
        ffmpeg_bin = shutil.which("ffmpeg")
        assert ffmpeg_bin is not None
        encoders = subprocess.run(
            [ffmpeg_bin, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        if "libx265" not in encoders:
            self.skipTest("libx265 encoder not available")
        source = self._bin / "tiny.hevc.flv"
        try:
            subprocess.run(
                [
                    ffmpeg_bin, "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc=duration=2:size=64x64:rate=10",
                    "-c:v", "libx265", "-f", "flv", "-y", str(source),
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError as exc:
            self.skipTest(f"cannot mux HEVC into FLV with this ffmpeg: {exc.stderr}")
        ffprobe_bin = shutil.which("ffprobe") or "ffprobe"
        reader = FFmpegStreamReader(
            str(source),
            open_timeout_ms=5000,
            read_timeout_ms=5000,
            scale_width=64,
            ffmpeg_binary=ffmpeg_bin,
            ffprobe_binary=ffprobe_bin,
        )
        reader.open()
        frames = []
        while True:
            frame = reader.read()
            if frame is None:
                break
            frames.append(frame)
        reader.release()
        self.assertGreater(len(frames), 3)
        self.assertEqual(frames[0].shape, (64, 64, 3))


class ReaderBackendWiringTest(unittest.TestCase):
    def test_settings_defaults_and_env_overrides(self) -> None:
        from elderly_monitoring.service.settings import ServiceSettings

        missing = Path("/path/that/does/not/exist.yaml")
        defaults = ServiceSettings.load(path=missing, environ={})
        self.assertEqual(defaults.stream_reader_backend, "ffmpeg")
        self.assertEqual(defaults.ffmpeg_scale_width, 640)
        overridden = ServiceSettings.load(
            path=missing,
            environ={"STREAM_READER_BACKEND": "ffmpeg", "FFMPEG_SCALE_WIDTH": "640"},
        )
        self.assertEqual(overridden.stream_reader_backend, "ffmpeg")
        self.assertEqual(overridden.ffmpeg_scale_width, 640)

    def test_settings_rejects_invalid_backend(self) -> None:
        from elderly_monitoring.service.settings import ServiceSettings

        with self.assertRaises(ValueError):
            ServiceSettings(stream_reader_backend="vlc")

    def test_reader_factory_for_ffmpeg_binds_original_frame_selector(self) -> None:
        from elderly_monitoring.service.app import reader_factory_for
        from elderly_monitoring.service.settings import ServiceSettings
        from elderly_monitoring.service.stream_reader import (
            FFmpegStreamReader,
            StreamReader,
        )

        settings = ServiceSettings(
            stream_reader_backend="ffmpeg",
            max_inference_fps=10.0,
            ffmpeg_scale_width=640,
        )
        reader = reader_factory_for(settings)(
            "rtmp://fake.invalid/live", open_timeout_ms=1234, read_timeout_ms=567
        )
        self.assertIsInstance(reader, FFmpegStreamReader)
        self.assertEqual(reader.max_fps, 10.0)
        self.assertEqual(reader.scale_width, 640)
        self.assertEqual(reader.open_timeout_ms, 1234)
        self.assertEqual(reader.read_timeout_ms, 567)

        opencv_settings = ServiceSettings(stream_reader_backend="opencv")
        self.assertIs(reader_factory_for(opencv_settings), StreamReader)


if __name__ == "__main__":
    unittest.main()
