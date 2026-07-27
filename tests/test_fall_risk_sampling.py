import unittest

from elderly_monitoring.runtime.sampling import SamplingMonitor


class SamplingMonitorTest(unittest.TestCase):
    def test_regular_samples_report_effective_fps_and_valid_branches(self) -> None:
        monitor = SamplingMonitor(window_sec=10.0)
        for index in range(12):
            source_time = index * 0.125
            monitor.observe(
                source_time_sec=source_time,
                received_monotonic_sec=10.0 + source_time,
                inference_started_monotonic_sec=10.01 + source_time,
            )

        result = monitor.snapshot()
        self.assertEqual(result["effective_fps"], 8.0)
        self.assertEqual(result["source_effective_fps"], 8.0)
        self.assertEqual(result["received_effective_fps"], 8.0)
        self.assertEqual(result["max_gap_sec"], 0.125)
        self.assertEqual(result["branches"]["gait"]["status"], "valid")
        self.assertEqual(result["branches"]["near_fall"]["status"], "valid")

    def test_large_gap_makes_time_sensitive_branches_unavailable(self) -> None:
        monitor = SamplingMonitor(window_sec=10.0)
        for source_time in (0.0, 0.1, 0.2, 1.3, 1.4, 1.5, 1.6, 1.7):
            monitor.observe(
                source_time_sec=source_time,
                received_monotonic_sec=20.0 + source_time,
                inference_started_monotonic_sec=20.0 + source_time,
            )

        result = monitor.snapshot()
        self.assertEqual(result["max_gap_sec"], 1.1)
        self.assertIn("source_gap_exceeded", result["branches"]["gait"]["reasons"])
        self.assertIn("source_gap_exceeded", result["branches"]["near_fall"]["reasons"])

    def test_non_increasing_pts_creates_boundary_and_does_not_join_windows(self) -> None:
        monitor = SamplingMonitor(window_sec=10.0)
        monitor.observe(source_time_sec=1.0, received_monotonic_sec=1.0, inference_started_monotonic_sec=1.0)
        reason = monitor.observe(source_time_sec=0.5, received_monotonic_sec=2.0, inference_started_monotonic_sec=2.1)

        result = monitor.snapshot()
        self.assertEqual(reason, "source_pts_regression")
        self.assertEqual(result["frame_count"], 1)
        self.assertEqual(result["time_boundary_count"], 1)
        self.assertEqual(result["last_frame_age_sec"], 0.1)


if __name__ == "__main__":
    unittest.main()
