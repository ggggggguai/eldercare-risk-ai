import unittest

from elderly_monitoring.runtime.fall_state import FallStateConfig, FallStateDetector


def _pose(timestamp, hip_y, angle, center_y=None, quality=0.9, motion=0.0):
    return {
        "timestamp_sec": timestamp,
        "hip_center_y": hip_y,
        "trunk_angle_deg": angle,
        "bbox_center_y": hip_y if center_y is None else center_y,
        "core_keypoint_quality": quality,
        "motion_score": motion,
    }


class FallStateDetectorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = FallStateDetector(
            FallStateConfig(
                observation_window_sec=1.0,
                min_quality=0.6,
                hip_drop_threshold=0.18,
                center_drop_threshold=0.15,
                horizontal_angle_threshold=60.0,
                static_duration_sec=2.0,
                static_motion_threshold=0.02,
            )
        )

    def test_normal_or_tilt_only_does_not_trigger(self) -> None:
        self.detector.update(_pose(0.0, 0.4, 10))
        normal = self.detector.update(_pose(0.8, 0.45, 15))
        tilted = self.detector.update(_pose(1.0, 0.45, 75))
        self.assertEqual(normal.fall_event_score, 0.0)
        self.assertEqual(tilted.fall_event_score, 0.0)

    def test_drop_horizontal_and_quality_trigger_then_static(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10, center_y=0.35))
        fall = self.detector.update(_pose(0.7, 0.62, 75, center_y=0.60))
        self.assertGreaterEqual(fall.fall_event_score, 0.8)
        self.detector.update(_pose(1.7, 0.62, 75, motion=0.01))
        static = self.detector.update(_pose(2.8, 0.62, 75, motion=0.01))
        self.assertGreaterEqual(static.long_static_score, 0.8)

    def test_low_quality_and_reset_do_not_reuse_old_event(self) -> None:
        self.detector.update(_pose(0.0, 0.35, 10))
        low = self.detector.update(_pose(0.7, 0.65, 80, quality=0.2))
        self.assertEqual(low.fall_event_score, 0.0)
        self.detector.reset()
        after_reset = self.detector.update(_pose(3.0, 0.65, 80, motion=0.0))
        self.assertEqual(after_reset.long_static_score, 0.0)


if __name__ == "__main__":
    unittest.main()
