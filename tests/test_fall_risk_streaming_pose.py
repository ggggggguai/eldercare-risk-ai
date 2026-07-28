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
        self.assertEqual(poses[0].bbox, [0.1, 0.1, 0.5, 0.5])
        self.assertEqual(poses[0].bbox_pixels, [10.0, 20.0, 50.0, 100.0])
        self.assertEqual(poses[0].coordinate_system, "image_normalized_0_1")
        self.assertEqual(tracks[0].bbox, [10.0, 20.0, 50.0, 100.0])

    def test_absolute_coordinates_keep_pose_bbox_in_pixels(self) -> None:
        _, poses = adapt_yolo_pose_result(
            _result(),
            frame_id=3,
            timestamp_sec=0.25,
            frame_size=(100, 200),
            scene_region="home",
            normalize_coordinates=False,
        )

        self.assertEqual(poses[0].bbox, [10.0, 20.0, 50.0, 100.0])
        self.assertEqual(poses[0].bbox_pixels, [10.0, 20.0, 50.0, 100.0])
        self.assertEqual(poses[0].coordinate_system, "image_pixels")

    def test_missing_frame_size_does_not_mislabel_pixel_coordinates(self) -> None:
        _, poses = adapt_yolo_pose_result(
            _result(),
            frame_id=3,
            timestamp_sec=0.25,
            frame_size=(0, 0),
            scene_region="home",
        )

        self.assertEqual(poses[0].bbox, [10.0, 20.0, 50.0, 100.0])
        self.assertEqual(poses[0].keypoints[0].x, 20.0)
        self.assertEqual(poses[0].coordinate_system, "image_pixels")

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
        self.assertTrue(reselected.target_changed)
        self.assertEqual(reselected.target_reason, "target_replaced_after_ttl")

    def test_multiple_initial_candidates_are_ambiguous_instead_of_largest_box_binding(self) -> None:
        model = SimpleNamespace(
            track=lambda **kwargs: [
                _result(
                    track_ids=(7, 8),
                    bboxes=((0, 0, 90, 190), (10, 20, 50, 100)),
                )
            ]
        )
        tracker = StreamingPoseTracker(
            model=model,
            person_id="elder-1",
            scene_region="home",
        )

        result = tracker.process_frame(
            object(), frame_id=1, timestamp_sec=0.0, frame_size=(100, 200)
        )

        self.assertIsNone(result.primary_pose)
        self.assertEqual(result.target_state, "ambiguous")
        self.assertEqual(result.target_reason, "multiple_unbound_candidates")
        self.assertIsNone(tracker.primary_track_id)

    def test_bound_target_stays_selected_when_other_people_enter(self) -> None:
        model = SimpleNamespace(track=lambda **kwargs: [_result()])
        tracker = StreamingPoseTracker(
            model=model,
            person_id="elder-1",
            scene_region="home",
        )
        tracker.process_frame(
            object(), frame_id=1, timestamp_sec=0.0, frame_size=(100, 200)
        )
        model.track = lambda **kwargs: [
            _result(
                track_ids=(8, 7),
                bboxes=((0, 0, 90, 190), (0, 0, 20, 40)),
            )
        ]

        result = tracker.process_frame(
            object(), frame_id=2, timestamp_sec=0.5, frame_size=(100, 200)
        )

        self.assertEqual(result.primary_pose.track_id, 7)
        self.assertEqual(result.target_state, "bound")
        self.assertEqual(result.target_reason, "bound_track_observed")
        self.assertFalse(result.binding_verified)

    def test_missing_detection_data_does_not_crash(self) -> None:
        empty = SimpleNamespace(boxes=None, keypoints=None)
        tracks, poses = adapt_yolo_pose_result(
            empty, frame_id=1, timestamp_sec=0.0, frame_size=(100, 100), scene_region="home"
        )
        self.assertEqual((tracks, poses), ([], []))

    def test_reset_clears_target_binding_and_backend_tracker_state(self) -> None:
        backend_tracker = SimpleNamespace(reset_calls=0)
        backend_tracker.reset = lambda: setattr(backend_tracker, "reset_calls", backend_tracker.reset_calls + 1)
        model = SimpleNamespace(
            track=lambda **kwargs: [_result()],
            predictor=SimpleNamespace(trackers=[backend_tracker]),
        )
        tracker = StreamingPoseTracker(model=model, person_id="elder-1", scene_region="home")
        tracker.process_frame(object(), frame_id=1, timestamp_sec=0.0, frame_size=(100, 200))

        tracker.reset()

        self.assertIsNone(tracker.primary_track_id)
        self.assertIsNone(tracker.primary_missing_since)
        self.assertEqual(backend_tracker.reset_calls, 1)
        self.assertEqual(tracker.target_state, "unbound")


if __name__ == "__main__":
    unittest.main()
