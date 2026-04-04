import cv2

from detector import HumanDetector
from tracker import PersonTracker
from event import detect_event, init_db, log_event, log_tracking_data, reset_runtime_state
from intent_manager import IntentManager
from query_engine import QueryEngine
from video_player import play_event
from zone_manager import ensure_camera_zones, has_any_zones

VIDEO_PATH = "test.mp4"
CAMERA_ID = 1

ZONES = []
drawing = False
start_point = None
current_rect = None

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


def draw_rectangle(event, x, y, flags, param):
    global drawing, start_point, current_rect, ZONES

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_rect = (start_point[0], start_point[1], x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        x1, y1 = start_point
        x2, y2 = x, y

        zone_id = len(ZONES) + 1

        zone = {
            "x1": min(x1, x2),
            "y1": min(y1, y2),
            "x2": max(x1, x2),
            "y2": max(y1, y2),
            "id": zone_id,
            "name": f"Zone {zone_id}"
        }

        ZONES.append(zone)
        current_rect = None

        print(f"✅ Zone {len(ZONES)} added:", zone)


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


def render_status(frame, camera_name, camera_id, fps, frame_count):
    status = f"{camera_name} #{camera_id} | FPS:{fps:.1f} | Frame:{frame_count}"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), STATUS_BG, cv2.FILLED)
    cv2.putText(frame, status,
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                STATUS_COLOR, 2)


def capture_static_frame(camera_config):
    source = resolve_capture_source(camera_config["source"])
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def ensure_zones_at_startup():
    if has_any_zones():
        return

    if not CAMERA_CONFIGS:
        return

    print("📦 No zones configured yet; capturing frame for zone definition.")
    camera = CAMERA_CONFIGS[0]
    frame = capture_static_frame(camera)

    if frame is None:
        print("⚠️ Unable to capture frame for zone setup; zones will be requested when needed.")
        return

    ensure_camera_zones(camera, frame)


def _print_event_summary(results: list):
    print("\nFound events:")
    print("Index | Timestamp           | Camera | Zone | Object (Event)")
    for idx, event in enumerate(results):
        print(
            f"{idx:5} | {event['timestamp'][:19]:19} | "
            f"{event['camera_id']:6} | {str(event['zone_id']):4} | "
            f"{event['object_type']} ({event['event_type']}) | track {event['track_id']}"
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
        results = query_engine.run_query(filters=filters, limit=30)

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
    print("   Press 'n' to switch camera, 'q' or Esc to exit surveillance.")

    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frame_skip = max(1, int(round(fps / TARGET_PROCESS_FPS)))
    tracker = PersonTracker()
    track_type_locks = {}
    reset_runtime_state()
    pixel_zones = []
    should_continue = True
    window_name = f"CCTV - {camera_config['name']}"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # frame skipping to target ~20 FPS processing
            for _ in range(frame_skip):
                ret = cap.grab()
                if not ret:
                    break

            if not ret:
                print(f"❌ {camera_config['name']} stream ended.")
                break

            ret, frame = cap.retrieve()

            if frame is None:
                print(f"❌ {camera_config['name']} frame is None.")
                break

            if not pixel_zones:
                pixel_zones = ensure_camera_zones(camera_config, frame)

                if not pixel_zones:
                    print(f"⚠️ No zones configured for {camera_config['name']}.")
                    should_continue = False
                    break

            current_frame_number = max(0, int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1)
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

                for idx, zone in enumerate(pixel_zones, start=1):
                    zone_id = zone.get("id", idx)
                    event = detect_event(track_id, bbox, zone, zone_id, locked_type)

                    if event:
                        log_event(
                            track_id,
                            locked_type,
                            event,
                            zone_id,
                            camera_config["camera_id"],
                            video_time,
                            camera_config["source"],
                            current_frame_number,
                        )

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{locked_type} ID {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if pixel_zones:
                draw_zone_overlays(frame, pixel_zones)

            render_status(
                frame,
                camera_config["name"],
                camera_config["camera_id"],
                fps,
                current_frame_number,
            )

            cv2.imshow(window_name, frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("n"):
                break

            if key in (ord("q"), 27):
                should_continue = False
                break

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
    ensure_zones_at_startup()

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
