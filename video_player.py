from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

from event import (
    get_playback_segments,
    get_tracking_data,
)
from multi_view import compose_multiview
from zone_manager import build_pixel_zones, get_camera_zones

TARGET_PLAYBACK_FPS = 20
PLAYBACK_JUMP_SECONDS = 2
LEFT_ARROW_KEYS = {81, 2424832, 65361}
RIGHT_ARROW_KEYS = {83, 2555904, 65363}


@dataclass
class PlaybackFeed:
    camera_id: Optional[int]
    video_path: str
    cap: cv2.VideoCapture
    fps: float
    total_frames: int
    start_frame: int
    end_frame: int
    zone_defs: List[Dict]
    metadata_by_frame: Dict[int, Dict[str, object]]
    current_frame: int
    title: str
    target_global_id: Optional[int]

    def close(self) -> None:
        self.cap.release()


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


def _draw_tracking_box(frame, bbox, track_id, global_id, object_type):
    x1, y1, x2, y2 = bbox
    color = (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    cv2.putText(
        frame,
        f"{object_type} GID {global_id} | Track {track_id}",
        (x1, max(16, y1 - 12)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def _draw_status(
    frame,
    playback_seconds: float,
    total_duration_seconds: float,
    target_label: str,
    paused: bool,
    view_label: str,
):
    status = f"Playback | {target_label} | {playback_seconds:.1f}s / {total_duration_seconds:.1f}s | {view_label}"
    if paused:
        status += " | PAUSED"

    controls = "1-9 fullscreen | M multi-view | SPACE pause/play | LEFT/RIGHT seek | Q quit"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 58), (0, 0, 0), cv2.FILLED)
    cv2.putText(
        frame,
        status,
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        controls,
        (12, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )


def _is_left_arrow(key):
    return key in LEFT_ARROW_KEYS


def _is_right_arrow(key):
    return key in RIGHT_ARROW_KEYS


def _clamp_frame(frame_number: int, total_frames: int) -> int:
    if total_frames > 0:
        return max(0, min(frame_number, total_frames - 1))
    return max(0, frame_number)


def _event_payload(
    event_or_video_file,
    frame_number=None,
    target_track_id=None,
    camera_id=None,
    video_time=None,
    target_global_id=None,
):
    if isinstance(event_or_video_file, dict):
        return dict(event_or_video_file)

    return {
        "video_path": event_or_video_file,
        "frame_number": frame_number,
        "frame_start": frame_number,
        "frame_end": frame_number,
        "track_id": target_track_id,
        "camera_id": camera_id,
        "video_time": video_time,
        "global_id": target_global_id,
        "entry_time": None,
        "exit_time": None,
    }


def _build_feed(
    source: Dict[str, object],
    event_entry: Dict[str, object],
) -> Optional[PlaybackFeed]:
    video_path = str(source["video_path"])
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Could not open video for camera {source.get('camera_id')}: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    if fps <= 0:
        cap.release()
        print(f"⚠️ Invalid FPS for camera {source.get('camera_id')}: {video_path}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = _clamp_frame(int(source.get("start_frame") or event_entry.get("frame_start") or 0), total_frames)
    end_frame = _clamp_frame(
        int(source.get("end_frame") or event_entry.get("frame_end") or event_entry.get("frame_number") or start_frame),
        total_frames,
    )
    if end_frame < start_frame:
        end_frame = start_frame

    feed_camera_id = source.get("camera_id")
    event_camera_id = event_entry.get("camera_id")
    feed_track_id = event_entry.get("track_id") if feed_camera_id == event_camera_id else None
    feed_global_id = event_entry.get("global_id")
    metadata_rows: List[Dict[str, object]] = []
    if feed_global_id is not None or feed_track_id is not None:
        metadata_rows = get_tracking_data(
            video_path,
            feed_track_id,
            start_frame,
            end_frame,
            camera_id=feed_camera_id,
            global_id=feed_global_id,
        )

    return PlaybackFeed(
        camera_id=feed_camera_id,
        video_path=video_path,
        cap=cap,
        fps=fps,
        total_frames=total_frames,
        start_frame=start_frame,
        end_frame=end_frame,
        zone_defs=get_camera_zones(feed_camera_id) if feed_camera_id is not None else [],
        metadata_by_frame={row["frame_number"]: row for row in metadata_rows},
        current_frame=start_frame,
        title=f"Camera {feed_camera_id}" if feed_camera_id is not None else "Camera",
        target_global_id=feed_global_id,
    )


def _read_feed_frame(feed: PlaybackFeed, playback_offset_sec: float) -> tuple[Optional[np.ndarray], bool]:
    absolute_target_frame = feed.start_frame + int(round(playback_offset_sec * feed.fps))
    is_within_session = absolute_target_frame <= feed.end_frame
    target_frame = _clamp_frame(min(feed.end_frame, absolute_target_frame), feed.total_frames)
    feed.current_frame = target_frame
    feed.cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ret, frame = feed.cap.read()
    if not ret or frame is None:
        return None, False

    pixel_zones = build_pixel_zones(feed.zone_defs, frame.shape) if feed.zone_defs else []
    _draw_zones(frame, pixel_zones)

    metadata = feed.metadata_by_frame.get(target_frame) if is_within_session else None
    if metadata is not None:
        _draw_tracking_box(
            frame,
            metadata["bbox"],
            metadata["track_id"],
            metadata["global_id"],
            metadata["object_type"],
        )

    return frame, metadata is not None


def play_event(
    event_or_video_file,
    frame_number=None,
    target_track_id=None,
    camera_id=None,
    video_time=None,
    target_global_id=None,
):
    event_entry = _event_payload(
        event_or_video_file,
        frame_number=frame_number,
        target_track_id=target_track_id,
        camera_id=camera_id,
        video_time=video_time,
        target_global_id=target_global_id,
    )
    video_file = str(event_entry["video_path"])
    frame_number = int(event_entry.get("frame_number") or event_entry.get("frame_start") or 0)
    camera_id = event_entry.get("camera_id")
    target_global_id = event_entry.get("global_id")
    target_track_id = event_entry.get("track_id")

    print("Opening video:", video_file)
    print("Seeking to frame:", frame_number)

    sources = get_playback_segments(
        video_path=video_file,
        frame_start=event_entry.get("frame_start"),
        frame_end=event_entry.get("frame_end"),
        camera_id=camera_id,
        track_id=target_track_id,
        global_id=target_global_id,
        entry_time=event_entry.get("entry_time"),
        exit_time=event_entry.get("exit_time"),
        event_mode=event_entry.get("event_mode") or event_entry.get("session_mode"),
    )
    feeds: List[PlaybackFeed] = []
    for source in sources:
        feed = _build_feed(source=source, event_entry=event_entry)
        if feed is not None:
            feeds.append(feed)

    if not feeds:
        print("❌ No playable camera feeds found for the selected event.")
        return

    target_label = f"GID {target_global_id}" if target_global_id is not None else f"Track {target_track_id}"
    session_duration_sec = 0.0
    for feed in feeds:
        feed_duration = max(0.0, (feed.end_frame - feed.start_frame) / max(1.0, float(feed.fps)))
        session_duration_sec = max(session_duration_sec, feed_duration)

    print(f"🎯 Session playback for {target_label} from frame {frame_number}")

    window_name = "Event Playback"
    paused = False
    playback_offset_sec = 0.0
    fullscreen_camera_id: Optional[int] = None

    try:
        while True:
            rendered_feeds = []
            for feed in feeds:
                frame, has_highlight = _read_feed_frame(feed, playback_offset_sec)
                rendered_feeds.append(
                    {
                        "camera_id": feed.camera_id,
                        "frame": frame,
                        "title": feed.title,
                        "subtitle": f"Frame {feed.current_frame} | Session {feed.start_frame}-{feed.end_frame}",
                        "highlight": has_highlight,
                    }
                )

            view_label = (
                f"Camera {fullscreen_camera_id}"
                if fullscreen_camera_id is not None
                else "Multi-view"
            )
            canvas = compose_multiview(rendered_feeds, fullscreen_camera_id=fullscreen_camera_id)
            _draw_status(canvas, playback_offset_sec, session_duration_sec, target_label, paused, view_label)
            cv2.imshow(window_name, canvas)

            key = cv2.waitKeyEx(30 if paused else max(1, int(1000 / TARGET_PLAYBACK_FPS)))
            if key in (ord("q"), ord("Q"), 27):
                break

            if key == ord(" "):
                paused = not paused
                continue

            if key in (ord("m"), ord("M")):
                fullscreen_camera_id = None
                continue

            if ord("1") <= key <= ord("9"):
                selected_camera_id = key - ord("0")
                if any(feed.camera_id == selected_camera_id for feed in feeds):
                    fullscreen_camera_id = selected_camera_id
                continue

            if _is_left_arrow(key) or _is_right_arrow(key):
                frame_offset_seconds = -PLAYBACK_JUMP_SECONDS if _is_left_arrow(key) else PLAYBACK_JUMP_SECONDS
                playback_offset_sec = max(0.0, min(session_duration_sec, playback_offset_sec + frame_offset_seconds))
                continue

            if not paused:
                playback_offset_sec += 1.0 / TARGET_PLAYBACK_FPS
                if playback_offset_sec > session_duration_sec:
                    playback_offset_sec = session_duration_sec
                    paused = True
    finally:
        for feed in feeds:
            feed.close()
        cv2.destroyWindow(window_name)
