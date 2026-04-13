import cv2
import numpy as np

from event import get_tracking_data
from zone_manager import build_pixel_zones, get_camera_zones

TARGET_PLAYBACK_FPS = 20
VIEW_WINDOW_SEC = 5
PLAYBACK_JUMP_SECONDS = 2
LEFT_ARROW_KEYS = {81, 2424832, 65361}
RIGHT_ARROW_KEYS = {83, 2555904, 65363}


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


def _draw_status(frame, frame_number, target_track_id, paused):
    status = f"Playback | Frame {frame_number} | Track {target_track_id}"
    if paused:
        status += " | PAUSED"

    controls = "SPACE pause/play | LEFT -2s | RIGHT +2s | Q quit"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 52), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        frame,
        status,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        controls,
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


def _frame_delta_for_seconds(fps, seconds):
    return max(1, int(round(fps * seconds)))


def _is_left_arrow(key):
    return key in LEFT_ARROW_KEYS


def _is_right_arrow(key):
    return key in RIGHT_ARROW_KEYS


def _seek_frame(cap, target_frame, start_frame, end_frame):
    if end_frame <= start_frame:
        target_frame = start_frame
    elif target_frame < start_frame:
        target_frame = end_frame
    elif target_frame > end_frame:
        target_frame = start_frame

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    return target_frame


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
    metadata_by_frame = {row["frame_number"]: row for row in metadata_rows}
    end_frame = max(end_frame, metadata_rows[-1]["frame_number"])
    frame_step = max(1, int(round(fps / TARGET_PLAYBACK_FPS)))
    jump_frames = _frame_delta_for_seconds(fps, PLAYBACK_JUMP_SECONDS)
    pixel_zones = None
    window_name = "Event Playback"
    paused = False
    current_frame = start_frame
    display_frame = None

    while True:
        if not paused or display_frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()

            if not ret or frame is None:
                print(f"⚠️ Unable to read frame {current_frame}.")
                current_frame = _seek_frame(cap, start_frame, start_frame, end_frame)
                display_frame = None
                continue

            if zone_defs and pixel_zones is None:
                pixel_zones = build_pixel_zones(zone_defs, frame.shape)

            _draw_zones(frame, pixel_zones)

            metadata = metadata_by_frame.get(current_frame)
            if metadata and metadata["track_id"] == target_track_id:
                _draw_tracking_box(
                    frame,
                    metadata["bbox"],
                    metadata["track_id"],
                    metadata["object_type"],
                )

            display_frame = frame

        frame_to_show = display_frame.copy()
        _draw_status(frame_to_show, current_frame, target_track_id, paused)
        cv2.imshow(window_name, frame_to_show)

        key = cv2.waitKeyEx(30 if paused else max(1, int(1000 / TARGET_PLAYBACK_FPS)))
        if key in (ord("q"), ord("Q"), 27):
            break

        if key == ord(" "):
            paused = not paused
            continue

        if _is_left_arrow(key) or _is_right_arrow(key):
            frame_offset = -jump_frames if _is_left_arrow(key) else jump_frames
            current_frame = _seek_frame(cap, current_frame + frame_offset, start_frame, end_frame)
            display_frame = None
            continue

        if not paused:
            current_frame = _seek_frame(cap, current_frame + frame_step, start_frame, end_frame)
            display_frame = None

    cap.release()
    cv2.destroyWindow(window_name)
