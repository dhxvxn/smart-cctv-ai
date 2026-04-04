import cv2
import numpy as np

from event import get_tracking_data
from zone_manager import build_pixel_zones, get_camera_zones

TARGET_PLAYBACK_FPS = 20
VIEW_WINDOW_SEC = 5


def _draw_zones(frame, zones):
    if not zones:
        return

    for zone in zones:
        polygon = zone.get("polygon")
        if not polygon:
            continue

        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(frame, [pts], True, (0, 128, 255), 2)

        label = zone.get("name", f"Zone {zone.get('id', '')}")
        text_pos = (pts[0][0], max(12, pts[0][1] - 8))
        cv2.putText(
            frame,
            label,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )


def _draw_tracking_box(frame, bbox, track_id, object_type):
    x1, y1, x2, y2 = bbox
    color = (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"{object_type} Track {track_id}",
        (x1, max(12, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def _draw_status(frame, frame_number, target_track_id):
    status = f"Playback | Frame {frame_number} | Track {target_track_id}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        frame,
        status,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )


def play_event(video_file, frame_number, target_track_id, camera_id=None, video_time=None):
    print("Opening video:", video_file)
    print("Seeking to frame:", frame_number)

    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        print("❌ Could not open video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        cap.release()
        print("❌ FPS is 0 — video not loaded properly")
        return

    start_frame = max(0, int(frame_number))
    end_frame = start_frame + int(fps * VIEW_WINDOW_SEC)
    metadata_rows = get_tracking_data(
        video_file,
        target_track_id,
        start_frame,
        end_frame,
        camera_id=camera_id,
    )

    if not metadata_rows:
        cap.release()
        print("⚠️ No stored tracking metadata found for the selected event.")
        return

    print(
        f"🎯 Metadata playback for Track {target_track_id} from frame {start_frame} "
        f"to {metadata_rows[-1]['frame_number']}"
    )
    if video_time is not None:
        print(f"   Event time: {video_time:.2f}s")

    zone_defs = get_camera_zones(camera_id) if camera_id is not None else []
    pixel_zones = []
    window_name = "Event Playback"

    for metadata in metadata_rows:
        current_frame = metadata["frame_number"]
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret or frame is None:
            print(f"⚠️ Unable to read frame {current_frame}.")
            continue

        if zone_defs and not pixel_zones:
            pixel_zones = build_pixel_zones(zone_defs, frame.shape)

        _draw_zones(frame, pixel_zones)

        if metadata["track_id"] == target_track_id:
            _draw_tracking_box(
                frame,
                metadata["bbox"],
                metadata["track_id"],
                metadata["object_type"],
            )

        _draw_status(frame, current_frame, target_track_id)

        cv2.imshow(window_name, frame)
        if cv2.waitKey(int(1000 / TARGET_PLAYBACK_FPS)) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
