import unittest
from types import SimpleNamespace

from elderly_monitoring.runtime.streaming_pose import StreamingPoseTracker, adapt_yolo_pose_result


class _Tensor:
    def __init__(self, value):
        self.value = value

    def cpu(self):
        return self

    def tolist(self):
        return self.value


def _result(track_ids=(7,), bboxes=((10, 20, 50, 100),)):
    count = len(bboxes)
    points = [[[20 + index, 30 + index] for index in range(17)] for _ in range(count)]
    return SimpleNamespace(
        boxes=SimpleNamespace(
            xyxy=_Tensor(list(bboxes)),
            conf=_Tensor([0.9] * count),
            id=None if track_ids is None else _Tensor(list(track_ids)),
        ),
        keypoints=SimpleNamespace(xy=_Tensor(points), conf=_Tensor([[0.8] * 17 for _ in range(count)])),
    )


class StreamingPoseTest(unittest.TestCase):
    def test_one_result_produces_matching_track_and_pose_ids(self) -> None:
        tracks, poses = adapt_yolo_pose_result(
            _result(), frame_id=3, timestamp_sec=0.25, frame_size=(100, 200), scene_region="home"
        )
        self.assertEqual(tracks[0].track_id, 7)
        self.assertEqual(poses[0].track_id, 7)
        self.assertEqual(poses[0].keypoints[0].x, 0.2)

    def test_primary_track_is_kept_while_present_then_reselected(self) -> None:
        model = SimpleNamespace(track=lambda **kwargs: [_result()])
        tracker = StreamingPoseTracker(model=model, person_id="elder-1", scene_region="home", lost_timeout_sec=1.0)
        first = tracker.process_frame(object(), frame_id=1, timestamp_sec=0.0, frame_size=(100, 200))
        self.assertEqual(first.primary_pose.track_id, 7)

        model.track = lambda **kwargs: [_result(track_ids=(8, 7), bboxes=((0, 0, 90, 190), (0, 0, 20, 40)))]
        second = tracker.process_frame(object(), frame_id=2, timestamp_sec=0.5, frame_size=(100, 200))
        self.assertEqual(second.primary_pose.track_id, 7)

        model.track = lambda **kwargs: [_result(track_ids=(8,), bboxes=((0, 0, 90, 190),))]
        missing = tracker.process_frame(object(), frame_id=3, timestamp_sec=0.8, frame_size=(100, 200))
        self.assertIsNone(missing.primary_pose)
        reselected = tracker.process_frame(object(), frame_id=4, timestamp_sec=1.9, frame_size=(100, 200))
        self.assertTrue(reselected.window_reset)
        self.assertEqual(reselected.primary_pose.track_id, 8)
        self.assertEqual(reselected.primary_pose.person_id, "elder-1")

    def test_missing_detection_data_does_not_crash(self) -> None:
        empty = SimpleNamespace(boxes=None, keypoints=None)
        tracks, poses = adapt_yolo_pose_result(
            empty, frame_id=1, timestamp_sec=0.0, frame_size=(100, 100), scene_region="home"
        )
        self.assertEqual((tracks, poses), ([], []))


if __name__ == "__main__":
    unittest.main()
