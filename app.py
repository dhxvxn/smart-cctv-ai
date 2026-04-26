from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2

from detector import HumanDetector
from event import (
    clear_event_logs,
    finalize_camera_sessions,
    flush_tracking_data,
    init_db,
    log_tracking_data,
    reset_runtime_state,
    update_session_event,
)
from intent_manager import IntentManager
from multi_view import compose_multiview
from query_engine import QueryEngine
from reid import GlobalIdentityManager
from tracker import PersonTracker
from video_player import play_event
from zone_manager import build_pixel_zones, draw_camera_zones, get_camera_zones, overwrite_zones

TRACKED_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
DEFAULT_FPS = 25.0
TARGET_PROCESS_FPS = 20.0
MIN_FRAME_STRIDE = 2
MAX_FRAME_STRIDE = 3
PLAYBACK_JUMP_SECONDS = 2
LEFT_ARROW_KEYS = {81, 2424832, 65361}
RIGHT_ARROW_KEYS = {83, 2555904, 65363}


@dataclass
class CameraRuntime:
    camera_id: int
    name: str
    source: object
    cap: cv2.VideoCapture
    fps: float
    frame_skip: int
    jump_frames: int
    total_frames: int
    zone_defs: List[Dict]
    tracker: PersonTracker = field(default_factory=PersonTracker)
    track_type_locks: Dict[int, str] = field(default_factory=dict)
    pixel_zones: Optional[List[Dict]] = None
    display_frame: Optional[object] = None
    current_frame_number: int = 0
    finished: bool = False

    def close(self) -> None:
        self.cap.release()

    def reset_tracking(self) -> None:
        self.tracker = PersonTracker()
        self.track_type_locks = {}
        self.display_frame = None


def _prompt_non_empty(prompt_text):
    while True:
        value = input(prompt_text).strip()
        if value:
            return value
        print("Input cannot be empty.")


def prompt_camera_configs():
    print("\nSelect Input Flow:")
    print("1. Single Camera")
    print("2. Multi Camera")

    while True:
        mode_choice = input("Enter choice (1/2): ").strip()
        if mode_choice in {"1", "2"}:
            break
        print("Invalid choice. Please enter 1 or 2.")

    if mode_choice == "1":
        source = _prompt_non_empty("Enter video path for camera 1: ")
        return (
            [
                {
                    "camera_id": 1,
                    "name": "Camera 1",
                    "source": source,
                }
            ],
            "single",
        )

    while True:
        total_cameras = input("Enter number of cameras: ").strip()
        if total_cameras.isdigit() and int(total_cameras) > 0:
            total_cameras = int(total_cameras)
            break
        print("Please enter a positive number.")

    camera_configs = []
    for camera_idx in range(1, total_cameras + 1):
        source = _prompt_non_empty(f"Enter video path for camera {camera_idx}: ")
        camera_configs.append(
            {
                "camera_id": camera_idx,
                "name": f"Camera {camera_idx}",
                "source": source,
            }
        )

    return camera_configs, "multi"


def resolve_capture_source(source):
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def draw_zone_overlays(frame, zones_list):
    for idx, zone in enumerate(zones_list, start=1):
        cv2.rectangle(
            frame,
            (zone["x1"], zone["y1"]),
            (zone["x2"], zone["y2"]),
            (0, 128, 255),
            2,
        )

        label = zone.get("name", f"Zone {idx}")
        cv2.putText(
            frame,
            label,
            (zone["x1"], zone["y1"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def _draw_global_status(frame, paused: bool, view_label: str, active_cameras: int):
    status = f"Surveillance | {view_label} | Active cameras: {active_cameras}"
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


def capture_static_frame(camera_config):
    source = resolve_capture_source(camera_config["source"])
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def _frame_delta_for_seconds(fps, seconds):
    return max(1, int(round(fps * seconds)))


def _frame_stride_for_fps(fps: float) -> int:
    if fps <= 20:
        return MIN_FRAME_STRIDE
    return MAX_FRAME_STRIDE


def _seek_frame(cap, target_frame, total_frames):
    if total_frames > 0:
        target_frame = max(0, min(target_frame, total_frames - 1))
    else:
        target_frame = max(0, target_frame)

    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    return target_frame


def _is_left_arrow(key):
    return key in LEFT_ARROW_KEYS


def _is_right_arrow(key):
    return key in RIGHT_ARROW_KEYS


def configure_zones_at_startup(camera_configs):
    if not camera_configs:
        return

    print("\n📦 Zone setup starts now. Existing zones will be overwritten in zones.json.")
    overwrite_zones([])
    all_zones = []
    next_zone_id = 1

    for camera in camera_configs:
        print(f"\n🎯 Draw zones for {camera['name']} ({camera['source']})")
        print("   Drag with the mouse, press 's' to save each zone, 'z' to remove the last one, Enter to finish.")

        frame = capture_static_frame(camera)
        if frame is None:
            print(f"⚠️ Unable to capture frame for {camera['name']}; saving no zones for this camera.")
            continue

        drawn_zones = draw_camera_zones(camera, frame, next_zone_id)
        all_zones.extend(drawn_zones)
        next_zone_id += len(drawn_zones)

    overwrite_zones(all_zones)
    print(f"💾 Saved {len(all_zones)} zone(s) to zones.json")


def _print_event_summary(results: list):
    label = results[0]["display_label"] if results else "session"
    print("\nFound events:")
    print(f"Index | {label.title():19} | Camera | Zone | Object | Global")
    for idx, event in enumerate(results):
        cameras = event.get("cameras") or [event["camera_id"]]
        camera_label = ",".join(str(camera) for camera in cameras if camera is not None)
        print(
            f"{idx:5} | {str(event['display_value'])[:19]:19} | "
            f"{camera_label[:6]:6} | {str(event['zone_id']):4} | "
            f"{event['object_type']:6} | {str(event.get('global_id', '-')):6}"
        )


def _select_event(results: list):
    while True:
        choice = input("\nSelect event number to play (blank to new query): ").strip().lower()
        if choice == "" or choice in {"b", "back"}:
            return

        if not choice.isdigit():
            print("  ➜ Please enter a valid index or press Enter to go back.")
            continue

        idx = int(choice)
        if idx < 0 or idx >= len(results):
            print("  ➜ Index out of range.")
            continue

        entry = results[idx]
        flush_tracking_data()
        play_event(entry)
        return


def run_query_mode(query_engine: QueryEngine, intent_manager: IntentManager, session_mode: str):
    while True:
        query = input(f"\nSearch query ({session_mode} mode, Enter to return): ").strip()
        if not query:
            return

        intent_manager.set_intent(query)
        filters = intent_manager.get_filters()
        results = query_engine.run_query(filters=filters, session_mode=session_mode)

        if not results:
            print("No matching events found.")
            continue

        _print_event_summary(results)
        _select_event(results)


def _create_camera_runtime(camera_config: Dict[str, object]) -> Optional[CameraRuntime]:
    source = resolve_capture_source(camera_config["source"])
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"⚠️ Unable to open {camera_config['name']} ({camera_config['source']})")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    zone_defs = get_camera_zones(camera_config["camera_id"])
    if not zone_defs:
        print(f"⚠️ No zones configured for {camera_config['name']}; camera will be skipped.")
        cap.release()
        return None

    return CameraRuntime(
        camera_id=int(camera_config["camera_id"]),
        name=str(camera_config["name"]),
        source=camera_config["source"],
        cap=cap,
        fps=fps,
        frame_skip=_frame_stride_for_fps(fps),
        jump_frames=_frame_delta_for_seconds(fps, PLAYBACK_JUMP_SECONDS),
        total_frames=total_frames,
        zone_defs=zone_defs,
    )


def _process_camera_frame(
    camera_state: CameraRuntime,
    detector: HumanDetector,
    identity_manager: GlobalIdentityManager,
    session_mode: str,
) -> None:
    if camera_state.finished:
        return

    ret, frame = camera_state.cap.read()
    if not ret or frame is None:
        camera_state.finished = True
        finalize_camera_sessions(camera_state.camera_id, camera_state.source)
        flush_tracking_data()
        print(f"✅ Finished processing {camera_state.name}.")
        return

    camera_state.current_frame_number = max(0, int(camera_state.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
    if camera_state.pixel_zones is None:
        camera_state.pixel_zones = build_pixel_zones(camera_state.zone_defs, frame.shape)
        if not camera_state.pixel_zones:
            print(f"⚠️ Saved zones for {camera_state.name} could not be rendered.")
            camera_state.finished = True
            return

    video_time = camera_state.current_frame_number / camera_state.fps if camera_state.fps else camera_state.current_frame_number / DEFAULT_FPS

    detections = detector.detect(frame)
    tracked_objects = camera_state.tracker.update(frame, detections)

    for (x1, y1, x2, y2, track_id, cls_id) in tracked_objects:
        object_type = detector.model.names[cls_id]
        normalized_type = object_type.lower()

        if normalized_type not in TRACKED_CLASSES:
            continue

        locked_type = camera_state.track_type_locks.setdefault(track_id, normalized_type)
        if locked_type != normalized_type:
            print(
                f"⚠️ Discarding class mismatch for camera {camera_state.camera_id} track {track_id}: "
                f"locked={locked_type}, detected={normalized_type}"
            )
            continue

        global_id = identity_manager.assign_global_id(
            camera_id=camera_state.camera_id,
            track_id=track_id,
            frame=frame,
            bbox=(x1, y1, x2, y2),
            object_type=locked_type,
            current_time=video_time,
        )

        log_tracking_data(
            track_id,
            global_id,
            locked_type,
            (x1, y1, x2, y2),
            camera_state.current_frame_number,
            camera_state.camera_id,
            camera_state.source,
        )

        update_session_event(
            track_id=track_id,
            global_id=global_id,
            object_type=locked_type,
            bbox=(x1, y1, x2 - x1, y2 - y1),
            zones=camera_state.pixel_zones,
            video_time=video_time,
            camera_id=camera_state.camera_id,
            video_path=camera_state.source,
            frame_number=camera_state.current_frame_number,
            event_mode=session_mode,
        )

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{locked_type} GID {global_id}",
            (x1, max(16, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    draw_zone_overlays(frame, camera_state.pixel_zones)
    camera_state.display_frame = frame


def _seek_all_cameras(camera_states: List[CameraRuntime], identity_manager: GlobalIdentityManager, direction: int) -> None:
    reset_runtime_state()
    for camera_state in camera_states:
        if camera_state.finished:
            continue

        target_frame = camera_state.current_frame_number + (direction * camera_state.jump_frames)
        _seek_frame(camera_state.cap, target_frame, camera_state.total_frames)
        camera_state.reset_tracking()
        identity_manager.clear_camera_track_mappings(camera_state.camera_id)


def _advance_cameras(camera_states: List[CameraRuntime]) -> None:
    for camera_state in camera_states:
        if camera_state.finished:
            continue

        next_frame = camera_state.current_frame_number + camera_state.frame_skip
        if camera_state.total_frames > 0 and next_frame >= camera_state.total_frames:
            camera_state.finished = True
            continue

        _seek_frame(camera_state.cap, next_frame, camera_state.total_frames)
        camera_state.display_frame = None


def run_surveillance_mode(camera_configs, detector, identity_manager, session_mode: str):
    camera_states = []
    for camera_config in camera_configs:
        camera_state = _create_camera_runtime(camera_config)
        if camera_state is not None:
            identity_manager.clear_camera_track_mappings(camera_state.camera_id)
            camera_states.append(camera_state)

    if not camera_states:
        print("❌ No camera streams are available for surveillance mode.")
        return True

    print(f"\n▶ Monitoring all configured cameras ({session_mode} event mode)")
    print("   Controls: 1-9 fullscreen, M multi-view, SPACE pause/play, LEFT/RIGHT seek, Q/Esc exit.")

    reset_runtime_state()
    paused = False
    fullscreen_camera_id: Optional[int] = None
    window_name = "CCTV - Surveillance"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            active_cameras = sum(1 for state in camera_states if not state.finished)
            if active_cameras == 0:
                break

            if not paused or any(state.display_frame is None for state in camera_states if not state.finished):
                for camera_state in camera_states:
                    if not paused or camera_state.display_frame is None:
                        _process_camera_frame(camera_state, detector, identity_manager, session_mode)

            feed_views = []
            for camera_state in camera_states:
                subtitle = f"Frame {camera_state.current_frame_number}"
                if camera_state.finished:
                    subtitle += " | Ended"
                feed_views.append(
                    {
                        "camera_id": camera_state.camera_id,
                        "frame": camera_state.display_frame,
                        "title": camera_state.name,
                        "subtitle": subtitle,
                    }
                )

            view_label = (
                f"Camera {fullscreen_camera_id}"
                if fullscreen_camera_id is not None
                else "Multi-view"
            )
            canvas = compose_multiview(feed_views, fullscreen_camera_id=fullscreen_camera_id)
            _draw_global_status(canvas, paused, view_label, active_cameras)
            cv2.imshow(window_name, canvas)

            key = cv2.waitKeyEx(30 if paused else max(1, int(1000 / TARGET_PROCESS_FPS)))

            if key in (ord("q"), ord("Q"), 27):
                return False

            if key == ord(" "):
                paused = not paused
                continue

            if key in (ord("m"), ord("M")):
                fullscreen_camera_id = None
                continue

            if ord("1") <= key <= ord("9"):
                selected_camera_id = key - ord("0")
                if any(state.camera_id == selected_camera_id for state in camera_states):
                    fullscreen_camera_id = selected_camera_id
                continue

            if _is_left_arrow(key):
                _seek_all_cameras(camera_states, identity_manager, -1)
                continue

            if _is_right_arrow(key):
                _seek_all_cameras(camera_states, identity_manager, 1)
                continue

            if not paused:
                _advance_cameras(camera_states)
    finally:
        for camera_state in camera_states:
            finalize_camera_sessions(camera_state.camera_id, camera_state.source)
        flush_tracking_data()
        for camera_state in camera_states:
            camera_state.close()
        cv2.destroyWindow(window_name)

    return True


def main():
    camera_configs, session_mode = prompt_camera_configs()
    init_db()
    clear_event_logs()
    detector = HumanDetector()
    identity_manager = GlobalIdentityManager()
    intent_manager = IntentManager()
    query_engine = QueryEngine()
    configure_zones_at_startup(camera_configs)

    while True:
        print("\nSelect Mode:")
        print("1. Full Surveillance Mode")
        print("2. Query-Based Mode")
        print("q. Quit")

        choice = input("Enter choice (1/2/q): ").strip().lower()

        if choice == "1":
            should_continue = run_surveillance_mode(camera_configs, detector, identity_manager, session_mode)
            if not should_continue:
                return
        elif choice == "2":
            run_query_mode(query_engine, intent_manager, session_mode)
        elif choice == "q":
            break
        else:
            print("Invalid choice. Please enter 1, 2 or q.")


if __name__ == "__main__":
    try:
        main()
    finally:
        cv2.destroyAllWindows()
