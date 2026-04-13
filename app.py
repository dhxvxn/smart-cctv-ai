import cv2

from detector import HumanDetector
from tracker import PersonTracker
from event import init_db, log_tracking_data, reset_runtime_state, update_session_event
from intent_manager import IntentManager
from query_engine import QueryEngine
from video_player import play_event
from zone_manager import build_pixel_zones, draw_camera_zones, get_camera_zones, overwrite_zones

CAMERA_CONFIGS = [
    {
        "camera_id": 1,
        "name": "Front Entrance",
        "source": "test.mp4",
        "zone_templates": [
            {"name": "Entry Lane", "x1": 0.15, "y1": 0.6, "x2": 0.85, "y2": 0.9},
            {"name": "Drop-Off", "x1": 0.05, "y1": 0.7, "x2": 0.35, "y2": 0.95},
        ],
    },
    {
        "camera_id": 2,
        "name": "Parking Approach",
        "source": "test.mp4",
        "zone_templates": [
            {"name": "Parking Row", "x1": 0.25, "y1": 0.5, "x2": 0.9, "y2": 0.92},
        ],
    },
]

TRACKED_CLASSES = {"person", "bicycle", "car", "motorcycle", "bus", "truck"}
ZONE_COLOR = (0, 128, 255)
ZONE_TEXT_COLOR = (255, 255, 255)
STATUS_COLOR = (255, 255, 255)
STATUS_BG = (0, 0, 0)
DEFAULT_FPS = 25.0
TARGET_PROCESS_FPS = 20.0
PLAYBACK_JUMP_SECONDS = 2
LEFT_ARROW_KEYS = {81, 2424832, 65361}
RIGHT_ARROW_KEYS = {83, 2555904, 65363}


def resolve_capture_source(source):
    if isinstance(source, str) and source.isdigit():
        return int(source)
    return source


def draw_zone_overlays(frame, zones_list):
    for idx, zone in enumerate(zones_list, start=1):
        cv2.rectangle(frame,
                      (zone["x1"], zone["y1"]),
                      (zone["x2"], zone["y2"]),
                      ZONE_COLOR, 2)

        label = zone.get("name", f"Zone {idx}")
        cv2.putText(frame, label,
                    (zone["x1"], zone["y1"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    ZONE_TEXT_COLOR, 2)


def render_status(frame, camera_name, camera_id, fps, frame_count, paused):
    status = f"{camera_name} #{camera_id} | FPS:{fps:.1f} | Frame:{frame_count}"
    if paused:
        status += " | PAUSED"

    controls = "SPACE pause/play | LEFT -2s | RIGHT +2s | Q quit | N next camera"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 52), STATUS_BG, cv2.FILLED)
    cv2.putText(
        frame,
        status,
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        STATUS_COLOR,
        2,
    )
    cv2.putText(
        frame,
        controls,
        (10, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        STATUS_COLOR,
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


def configure_zones_at_startup():
    if not CAMERA_CONFIGS:
        return

    print("\n📦 Zone setup starts now. Existing zones will be overwritten in zones.json.")
    all_zones = []
    next_zone_id = 1

    for camera in CAMERA_CONFIGS:
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
    print(f"Index | {label.title():19} | Camera | Zone | Object | Track")
    for idx, event in enumerate(results):
        print(
            f"{idx:5} | {str(event['display_value'])[:19]:19} | "
            f"{event['camera_id']:6} | {str(event['zone_id']):4} | "
            f"{event['object_type']:6} | track {event['track_id']}"
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
        play_event(
            entry["video_path"],
            entry["frame_number"],
            entry["track_id"],
            entry.get("camera_id"),
            entry.get("video_time"),
        )
        return


def run_query_mode(query_engine: QueryEngine, intent_manager: IntentManager):
    while True:
        query = input("\nSearch query (Enter to return): ").strip()
        if not query:
            return

        intent_manager.set_intent(query)
        filters = intent_manager.get_filters()
        results = query_engine.run_query(filters=filters)

        if not results:
            print("No matching events found.")
            continue

        _print_event_summary(results)
        _select_event(results)


def run_surveillance_mode(camera_config, detector):
    source = resolve_capture_source(camera_config["source"])
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"❌ Unable to open {camera_config['name']} ({camera_config['source']})")
        return True

    print(f"\n▶ Monitoring {camera_config['name']} ({camera_config['source']})")
    print("   Controls: SPACE pause/play, LEFT -2s, RIGHT +2s, N next camera, Q/Esc exit.")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_skip = max(1, int(round(fps / TARGET_PROCESS_FPS)))
    jump_frames = _frame_delta_for_seconds(fps, PLAYBACK_JUMP_SECONDS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    zone_defs = get_camera_zones(camera_config["camera_id"])
    if not zone_defs:
        print(f"⚠️ No zones configured for {camera_config['name']}.")
        cap.release()
        return True

    tracker = PersonTracker()
    track_type_locks = {}
    reset_runtime_state()
    pixel_zones = None
    paused = False
    display_frame = None
    current_frame_number = 0
    should_continue = True
    window_name = f"CCTV - {camera_config['name']}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            if not paused or display_frame is None:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print(f"❌ {camera_config['name']} stream ended.")
                    break

                current_frame_number = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
                if pixel_zones is None:
                    pixel_zones = build_pixel_zones(zone_defs, frame.shape)
                    if not pixel_zones:
                        print(f"⚠️ Saved zones for {camera_config['name']} could not be rendered.")
                        should_continue = False
                        break

                video_time = current_frame_number / fps if fps else current_frame_number / DEFAULT_FPS

                detections = detector.detect(frame)
                tracked_objects = tracker.update(frame, detections)

                for (x1, y1, x2, y2, track_id, cls_id) in tracked_objects:
                    object_type = detector.model.names[cls_id]
                    normalized_type = object_type.lower()

                    if normalized_type not in TRACKED_CLASSES:
                        continue

                    locked_type = track_type_locks.setdefault(track_id, normalized_type)
                    if locked_type != normalized_type:
                        print(
                            f"⚠️ Discarding class mismatch for track {track_id}: "
                            f"locked={locked_type}, detected={normalized_type}"
                        )
                        continue

                    log_tracking_data(
                        track_id,
                        locked_type,
                        (x1, y1, x2, y2),
                        current_frame_number,
                        camera_config["camera_id"],
                        camera_config["source"],
                    )

                    bbox = (x1, y1, x2 - x1, y2 - y1)

                    update_session_event(
                        track_id=track_id,
                        object_type=locked_type,
                        bbox=bbox,
                        zones=pixel_zones,
                        video_time=video_time,
                        camera_id=camera_config["camera_id"],
                        video_path=camera_config["source"],
                        frame_number=current_frame_number,
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{locked_type} ID {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )

                draw_zone_overlays(frame, pixel_zones)
                display_frame = frame

            frame_to_show = display_frame.copy()
            render_status(
                frame_to_show,
                camera_config["name"],
                camera_config["camera_id"],
                fps,
                current_frame_number,
                paused,
            )

            cv2.imshow(window_name, frame_to_show)

            key = cv2.waitKeyEx(30 if paused else max(1, int(1000 / TARGET_PROCESS_FPS)))

            if key == ord("n"):
                break

            if key in (ord("q"), ord("Q"), 27):
                should_continue = False
                break

            if key == ord(" "):
                paused = not paused
                continue

            if _is_left_arrow(key) or _is_right_arrow(key):
                frame_offset = -jump_frames if _is_left_arrow(key) else jump_frames
                _seek_frame(cap, current_frame_number + frame_offset, total_frames)
                tracker = PersonTracker()
                track_type_locks = {}
                reset_runtime_state()
                display_frame = None
                continue

            if not paused:
                next_frame = current_frame_number + frame_skip
                if total_frames > 0 and next_frame >= total_frames:
                    print(f"✅ Finished processing {camera_config['name']}.")
                    break
                else:
                    _seek_frame(cap, next_frame, total_frames)
                display_frame = None

    finally:
        cap.release()
        cv2.destroyWindow(window_name)

    return should_continue


def select_camera():
    if not CAMERA_CONFIGS:
        print("No cameras configured.")
        return None

    print("\nAvailable cameras:")
    for idx, config in enumerate(CAMERA_CONFIGS, start=1):
        print(f"{idx}. {config['name']} → {config['source']}")

    while True:
        choice = input("Enter camera number (0 to return): ").strip()
        if not choice or choice == "0":
            return None
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(CAMERA_CONFIGS):
                return CAMERA_CONFIGS[idx - 1]
        print("Invalid selection. Try again.")


def main():
    init_db()
    detector = HumanDetector()
    intent_manager = IntentManager()
    query_engine = QueryEngine()
    configure_zones_at_startup()

    while True:
        print("\nSelect Mode:")
        print("1. Full Surveillance Mode")
        print("2. Query-Based Mode")
        print("q. Quit")

        choice = input("Enter choice (1/2/q): ").strip().lower()

        if choice == "1":
            while True:
                camera = select_camera()
                if camera is None:
                    break
                should_continue = run_surveillance_mode(camera, detector)
                if not should_continue:
                    return

        elif choice == "2":
            run_query_mode(query_engine, intent_manager)

        elif choice == "q":
            break

        else:
            print("Invalid choice. Please enter 1, 2 or q.")


if __name__ == "__main__":
    try:
        main()
    finally:
        cv2.destroyAllWindows()
