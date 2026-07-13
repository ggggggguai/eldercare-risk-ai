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
    def __init__(self, **kwargs):
        self.closed = False

    def process_frame(self, frame, **kwargs):
        return None

    def reset_window(self):
        pass

    def close(self):
        self.closed = True


class SessionServiceTest(unittest.TestCase):
    def setUp(self):
        FakeReader.instances.clear()
        self.manager = SessionManager(reader_factory=FakeReader, engine_factory=FakeEngine, reconnect_attempts=1, reconnect_delay_sec=0.0)

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


if __name__ == "__main__":
    unittest.main()
