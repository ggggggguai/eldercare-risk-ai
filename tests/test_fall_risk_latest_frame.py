import unittest

from elderly_monitoring.service.frame_buffer import FramePacket, LatestFrameBuffer


class LatestFrameBufferTest(unittest.TestCase):
    def test_full_buffer_discards_oldest_frame_with_diagnostics(self) -> None:
        buffer = LatestFrameBuffer(capacity=2)
        buffer.put(FramePacket(frame="old", stream_epoch=1, source_pts_sec=1.0, received_monotonic_sec=10.0))
        buffer.put(FramePacket(frame="middle", stream_epoch=1, source_pts_sec=2.0, received_monotonic_sec=11.0))
        buffer.put(FramePacket(frame="latest", stream_epoch=1, source_pts_sec=3.0, received_monotonic_sec=12.0))

        self.assertEqual(buffer.get(timeout=0.0).frame, "middle")
        self.assertEqual(buffer.get(timeout=0.0).frame, "latest")
        self.assertEqual(buffer.snapshot()["dropped_oldest"], 1)
        self.assertEqual(buffer.snapshot()["last_dropped_source_pts_sec"], 1.0)

    def test_close_unblocks_empty_consumer(self) -> None:
        buffer = LatestFrameBuffer(capacity=1)
        buffer.close(reason="stream_ended")

        self.assertIsNone(buffer.get(timeout=0.0))
        self.assertEqual(buffer.snapshot()["close_reason"], "stream_ended")


if __name__ == "__main__":
    unittest.main()
