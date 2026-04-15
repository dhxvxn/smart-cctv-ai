import unittest
import sqlite3

import numpy as np

from event import clear_event_logs, get_playback_segments, init_db
from reid import GlobalIdentityManager


def make_frame(bbox, color=(0, 0, 255), shape=(240, 320, 3)):
    frame = np.zeros(shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    frame[y1:y2, x1:x2] = color
    return frame


class GlobalIdentityManagerTests(unittest.TestCase):
    def test_same_track_keeps_same_global_id(self):
        manager = GlobalIdentityManager(similarity_threshold=0.8, spatial_threshold=80.0)
        bbox = (20, 40, 80, 100)

        first_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=10,
            frame=make_frame(bbox),
            bbox=bbox,
            object_type="car",
            current_time=1.0,
        )
        second_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=10,
            frame=make_frame((25, 42, 85, 102)),
            bbox=(25, 42, 85, 102),
            object_type="car",
            current_time=2.0,
        )

        self.assertEqual(first_global_id, second_global_id)

    def test_nearby_same_car_reuses_existing_global_id_after_track_change(self):
        manager = GlobalIdentityManager(similarity_threshold=0.8, spatial_threshold=80.0)
        first_bbox = (20, 40, 80, 100)
        second_bbox = (35, 42, 95, 102)

        original_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=10,
            frame=make_frame(first_bbox),
            bbox=first_bbox,
            object_type="car",
            current_time=1.0,
        )

        manager.clear_camera_track_mappings(camera_id=1)

        reacquired_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=11,
            frame=make_frame(second_bbox),
            bbox=second_bbox,
            object_type="car",
            current_time=2.0,
        )

        self.assertEqual(original_global_id, reacquired_global_id)

    def test_same_looking_car_after_time_window_gets_new_global_id(self):
        manager = GlobalIdentityManager(similarity_threshold=0.8, spatial_threshold=80.0)
        first_bbox = (20, 40, 80, 100)
        far_bbox = (210, 40, 270, 100)

        first_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=10,
            frame=make_frame(first_bbox),
            bbox=first_bbox,
            object_type="car",
            current_time=1.0,
        )

        manager.clear_camera_track_mappings(camera_id=1)

        second_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=11,
            frame=make_frame(far_bbox),
            bbox=far_bbox,
            object_type="car",
            current_time=7.0,
        )

        self.assertNotEqual(first_global_id, second_global_id)

    def test_same_frame_simultaneous_tracks_do_not_share_global_id(self):
        manager = GlobalIdentityManager(similarity_threshold=0.8, spatial_threshold=120.0)
        left_bbox = (20, 40, 80, 100)
        right_bbox = (90, 40, 150, 100)

        first_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=10,
            frame=make_frame(left_bbox),
            bbox=left_bbox,
            object_type="car",
            current_time=1.0,
        )
        second_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=11,
            frame=make_frame(right_bbox),
            bbox=right_bbox,
            object_type="car",
            current_time=1.0,
        )

        self.assertNotEqual(first_global_id, second_global_id)

    def test_same_object_across_cameras_reuses_global_id(self):
        manager = GlobalIdentityManager(
            similarity_threshold=0.8,
            spatial_threshold=80.0,
            cross_camera_similarity_threshold=0.85,
        )
        bbox = (30, 50, 110, 170)
        frame = make_frame(bbox, color=(0, 255, 0), shape=(360, 480, 3))

        first_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=10,
            frame=frame,
            bbox=bbox,
            object_type="person",
            current_time=1.0,
        )

        manager.clear_camera_track_mappings(camera_id=2)

        second_global_id = manager.assign_global_id(
            camera_id=2,
            track_id=3,
            frame=frame,
            bbox=bbox,
            object_type="person",
            current_time=3.5,
        )

        self.assertEqual(first_global_id, second_global_id)

    def test_cross_camera_match_expires_after_three_seconds(self):
        manager = GlobalIdentityManager(similarity_threshold=0.8)
        bbox = (30, 50, 110, 170)
        frame = make_frame(bbox, color=(0, 255, 0), shape=(360, 480, 3))

        first_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=10,
            frame=frame,
            bbox=bbox,
            object_type="person",
            current_time=1.0,
        )

        manager.clear_camera_track_mappings(camera_id=2)

        second_global_id = manager.assign_global_id(
            camera_id=2,
            track_id=3,
            frame=frame,
            bbox=bbox,
            object_type="person",
            current_time=4.1,
        )

        self.assertNotEqual(first_global_id, second_global_id)

    def test_match_updates_last_seen_time_and_camera(self):
        manager = GlobalIdentityManager(similarity_threshold=0.8)
        bbox = (30, 50, 110, 170)
        frame = make_frame(bbox, color=(0, 255, 0), shape=(360, 480, 3))

        global_id = manager.assign_global_id(
            camera_id=1,
            track_id=10,
            frame=frame,
            bbox=bbox,
            object_type="person",
            current_time=1.0,
        )

        manager.clear_camera_track_mappings(camera_id=2)

        reused_global_id = manager.assign_global_id(
            camera_id=2,
            track_id=3,
            frame=frame,
            bbox=bbox,
            object_type="person",
            current_time=2.5,
        )

        self.assertEqual(global_id, reused_global_id)
        self.assertEqual(manager.identity_store[global_id].last_camera_id, 2)
        self.assertEqual(manager.identity_store[global_id].last_seen_time, 2.5)

    def test_exact_three_second_gap_creates_new_global_id(self):
        manager = GlobalIdentityManager(similarity_threshold=0.8)
        bbox = (30, 50, 110, 170)
        frame = make_frame(bbox, color=(0, 255, 0), shape=(360, 480, 3))

        first_global_id = manager.assign_global_id(
            camera_id=1,
            track_id=10,
            frame=frame,
            bbox=bbox,
            object_type="person",
            current_time=1.0,
        )

        manager.clear_camera_track_mappings(camera_id=2)

        second_global_id = manager.assign_global_id(
            camera_id=2,
            track_id=3,
            frame=frame,
            bbox=bbox,
            object_type="person",
            current_time=4.0,
        )

        self.assertNotEqual(first_global_id, second_global_id)


class PlaybackSessionTests(unittest.TestCase):
    def setUp(self):
        init_db()
        clear_event_logs()

    def tearDown(self):
        clear_event_logs()

    def test_get_playback_segments_uses_session_window_for_global_id(self):
        conn = sqlite3.connect("events.db")
        cursor = conn.cursor()
        cursor.executemany(
            """
            INSERT INTO events
            (timestamp, object_type, track_id, global_id, camera_id, video_path,
             frame_number, frame_start, frame_end, video_time, zone_id,
             event_type, entry_time, exit_time, duration, stayed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    "2026-04-14T10:00:03",
                    "person",
                    11,
                    77,
                    1,
                    "cam1.mp4",
                    40,
                    10,
                    40,
                    4.0,
                    1,
                    "leaving",
                    "2026-04-14T10:00:00",
                    "2026-04-14T10:00:03",
                    3.0,
                    0,
                ),
                (
                    "2026-04-14T10:00:06",
                    "person",
                    5,
                    77,
                    2,
                    "cam2.mp4",
                    25,
                    5,
                    25,
                    2.5,
                    2,
                    "leaving",
                    "2026-04-14T10:00:04",
                    "2026-04-14T10:00:06",
                    2.0,
                    0,
                ),
                (
                    "2026-04-14T10:01:10",
                    "person",
                    21,
                    77,
                    3,
                    "cam3.mp4",
                    90,
                    70,
                    90,
                    9.0,
                    1,
                    "leaving",
                    "2026-04-14T10:01:00",
                    "2026-04-14T10:01:10",
                    10.0,
                    1,
                ),
            ],
        )
        conn.commit()
        conn.close()

        segments = get_playback_segments(
            video_path="cam1.mp4",
            frame_start=10,
            frame_end=40,
            camera_id=1,
            track_id=11,
            global_id=77,
            entry_time="2026-04-14T10:00:00",
            exit_time="2026-04-14T10:00:03",
        )

        self.assertEqual(
            [(segment["camera_id"], segment["video_path"]) for segment in segments],
            [(1, "cam1.mp4"), (2, "cam2.mp4")],
        )
        self.assertEqual(segments[0]["start_frame"], 10)
        self.assertEqual(segments[0]["end_frame"], 40)
        self.assertEqual(segments[1]["start_frame"], 5)
        self.assertEqual(segments[1]["end_frame"], 25)


if __name__ == "__main__":
    unittest.main()
