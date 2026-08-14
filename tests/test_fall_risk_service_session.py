import threading
import time
import unittest

from elderly_monitoring.service.session import SessionManager, SessionStatus


class FakeReader:
    instances = []

    def __init__(self, url, **kwargs):
        self.url = url
        self.closed = False
        self.frames = [object(), object(), None]
        self.__class__.instances.append(self)

    def open(self):
        if self.url == "bad://url":
            raise RuntimeError("cannot open")

    def read(self):
        return self.frames.pop(0) if self.frames else None

    def release(self):
        self.closed = True

    def update_url(self, url):
        self.url = url
        self.frames = [object(), None]


class FakeEngine:
    instances = []

    def __init__(self, **kwargs):
        self.closed = False
        self.epochs = []
        self.processed = []
        self.closed_diagnostics = None
        self.__class__.instances.append(self)

    def process_frame(self, frame, **kwargs):
        self.processed.append((frame, kwargs))

    def begin_stream_epoch(self, stream_epoch, **kwargs):
        self.epochs.append((stream_epoch, kwargs))

    def reset_window(self):
        pass

    def close(self):
        self.closed_diagnostics = {"reason": "session_stopped"}
        self.closed = True


class SessionServiceTest(unittest.TestCase):
    def setUp(self):
        FakeReader.instances.clear()
        FakeEngine.instances.clear()
        self.manager = SessionManager(
            reader_factory=FakeReader,
            engine_factory=FakeEngine,
            reconnect_attempts=1,
            reconnect_delay_sec=0.0,
            frame_queue_capacity=1,
        )

    def tearDown(self):
        for session in list(self.manager.sessions.values()):
            self.manager.stop(session.session_id)

    def test_start_stop_and_idempotency(self):
        session = self.manager.start(request_id="r1", stream_url="https://example/live", device_id="cam", person_id="elder", scene_region="home", callback_url="https://backend/events")
        self.assertIn(session.status, {SessionStatus.STARTING, SessionStatus.RUNNING, SessionStatus.RECONNECTING})
        self.assertIs(self.manager.start(request_id="r1", stream_url="https://example/live", device_id="cam", person_id="elder", scene_region="home", callback_url="https://backend/events"), session)
        self.manager.stop(session.session_id)
        session.thread.join(timeout=2)
        self.assertEqual(session.status, SessionStatus.STOPPED)
        self.assertFalse(
            session.runtime_diagnostics["stop"]["thread_alive_after_budget"]
        )
        self.manager.stop(session.session_id)
        self.assertIs(self.manager.start(request_id="r1", stream_url="https://example/live", device_id="cam", person_id="elder", scene_region="home", callback_url="https://backend/events"), session)

    def test_only_one_different_request_is_allowed(self):
        self.manager.start(request_id="r1", stream_url="https://example/live", device_id="cam", person_id="elder", scene_region="home", callback_url="https://backend/events")
        with self.assertRaises(ValueError):
            self.manager.start(request_id="r2", stream_url="https://example/live", device_id="cam", person_id="elder", scene_region="home", callback_url="https://backend/events")

    def test_bad_start_fails(self):
        session = self.manager.start(request_id="r1", stream_url="bad://url", device_id="cam", person_id="elder", scene_region="home", callback_url="https://backend/events")
        session.thread.join(timeout=2)
        self.assertEqual(session.status, SessionStatus.FAILED)
        self.assertIn("cannot open", session.last_error)

    def test_update_url_releases_old_reader(self):
        session = self.manager.start(request_id="r1", stream_url="https://example/live", device_id="cam", person_id="elder", scene_region="home", callback_url="https://backend/events")
        time.sleep(0.02)
        old = FakeReader.instances[0]
        self.manager.update_url(session.session_id, "https://example/new")
        self.assertTrue(old.closed)
        self.manager.stop(session.session_id)

    def test_url_update_creates_epoch_with_explicit_boundary_reason(self):
        class BlockingReader(FakeReader):
            def __init__(self, url, **kwargs):
                super().__init__(url, **kwargs)
                self.frames = [object()] * 1000

            def read(self):
                if self.closed:
                    return None
                time.sleep(0.001)
                return super().read()

        manager = SessionManager(
            reader_factory=BlockingReader,
            engine_factory=FakeEngine,
            reconnect_attempts=2,
            reconnect_delay_sec=0.0,
            frame_queue_capacity=1,
        )
        session = manager.start(
            request_id="url-boundary",
            stream_url="https://example/live",
            device_id="cam",
            person_id="elder",
            scene_region="home",
            callback_url="https://backend/events",
        )
        deadline = time.time() + 1.0
        while session.stream_epoch < 1 and time.time() < deadline:
            time.sleep(0.005)
        manager.update_url(session.session_id, "https://example/new")
        deadline = time.time() + 1.0
        while session.stream_epoch < 2 and time.time() < deadline:
            time.sleep(0.005)
        manager.stop(session.session_id)

        engine = FakeEngine.instances[-1]
        reasons = [kwargs["reason"] for _, kwargs in engine.epochs]
        self.assertEqual(reasons[0], "session_started")
        self.assertIn("stream_url_updated", reasons[1:])

    def test_url_update_during_open_discards_stale_connection(self):
        open_started = threading.Event()
        allow_open = threading.Event()

        class SlowOpenReader(FakeReader):
            def __init__(self, url, **kwargs):
                super().__init__(url, **kwargs)
                if url.endswith("/new"):
                    self.frames = [object()] * 1000

            def open(self):
                if self.url.endswith("/old"):
                    open_started.set()
                    allow_open.wait(timeout=1.0)

            def read(self):
                if self.closed:
                    return None
                if self.url.endswith("/new"):
                    time.sleep(0.001)
                return super().read()

        manager = SessionManager(
            reader_factory=SlowOpenReader,
            engine_factory=FakeEngine,
            reconnect_attempts=1,
            reconnect_delay_sec=0.0,
            frame_queue_capacity=1,
        )
        session = manager.start(
            request_id="open-update",
            stream_url="https://example/old",
            device_id="cam",
            person_id="elder",
            scene_region="home",
            callback_url="https://backend/events",
        )
        self.assertTrue(open_started.wait(timeout=1.0))
        manager.update_url(session.session_id, "https://example/new")
        allow_open.set()
        deadline = time.time() + 1.0
        while (
            not any(reader.url.endswith("/new") for reader in FakeReader.instances)
            or session.stream_epoch < 1
        ) and time.time() < deadline:
            time.sleep(0.005)
        manager.stop(session.session_id)

        engine = FakeEngine.instances[-1]
        self.assertTrue(engine.epochs)
        self.assertEqual(engine.epochs[0][1]["reason"], "stream_url_updated")
        self.assertEqual(session.stream_epoch, 1)
        self.assertEqual(len(engine.epochs), 1)
        self.assertTrue(FakeReader.instances[0].closed)
        self.assertTrue(any(reader.url.endswith("/new") for reader in FakeReader.instances))

    def test_each_open_creates_epoch_and_passes_clock_fields_to_engine(self):
        session = self.manager.start(request_id="r1", stream_url="https://example/live", device_id="cam", person_id="elder", scene_region="home", callback_url="https://backend/events")
        session.thread.join(timeout=2)

        engine = FakeEngine.instances[0]
        self.assertGreaterEqual(session.stream_epoch, 2)
        self.assertEqual([epoch for epoch, _ in engine.epochs], list(range(1, session.stream_epoch + 1)))
        self.assertTrue(engine.processed)
        _, kwargs = engine.processed[0]
        self.assertIn("source_pts_sec", kwargs)
        self.assertIn("received_monotonic_sec", kwargs)
        self.assertIn("stream_epoch", kwargs)
        self.assertGreaterEqual(session.frame_diagnostics["put_count"], len(engine.processed))

    def test_repeated_short_streams_exhaust_reconnect_budget(self):
        manager = SessionManager(
            reader_factory=FakeReader,
            engine_factory=FakeEngine,
            reconnect_attempts=2,
            reconnect_delay_sec=0.0,
            reconnect_stable_after_sec=10.0,
            reconnect_stable_after_frames=30,
            frame_queue_capacity=2,
        )
        session = manager.start(
            request_id="short-stream",
            stream_url="https://example/live",
            device_id="cam",
            person_id="elder",
            scene_region="home",
            callback_url="https://backend/events",
        )
        session.thread.join(timeout=2)

        self.assertFalse(session.thread.is_alive())
        self.assertEqual(session.status, SessionStatus.FAILED)
        self.assertEqual(session.stream_epoch, 3)
        self.assertEqual(session.last_error, "stream reconnect attempts exhausted")

    def test_slow_inference_keeps_queue_bounded_and_drops_old_frames(self):
        class BurstReader(FakeReader):
            def __init__(self, url, **kwargs):
                super().__init__(url, **kwargs)
                self.frames = [object() for _ in range(50)] + [None]

        class SlowEngine(FakeEngine):
            def process_frame(self, frame, **kwargs):
                time.sleep(0.005)
                super().process_frame(frame, **kwargs)

        manager = SessionManager(
            reader_factory=BurstReader,
            engine_factory=SlowEngine,
            reconnect_attempts=0,
            reconnect_delay_sec=0.0,
            frame_queue_capacity=2,
        )
        session = manager.start(request_id="pressure", stream_url="https://example/live", device_id="cam", person_id="elder", scene_region="home", callback_url="https://backend/events")
        session.thread.join(timeout=2)

        self.assertEqual(session.status, SessionStatus.FAILED)
        self.assertGreater(session.frame_diagnostics["dropped_oldest"], 0)
        self.assertLess(len(SlowEngine.instances[-1].processed), 50)
        self.assertTrue(all(item["capacity"] == 2 for item in session.epoch_history))


if __name__ == "__main__":
    unittest.main()
