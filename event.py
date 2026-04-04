import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

DB_PATH = "events.db"

# keeps history per track + zone
track_history = {}

# avoid spam events
last_trigger_time = {}
COOLDOWN = 3
MIN_ZONE_CONFIRM_FRAMES = 2


def reset_runtime_state() -> None:
    track_history.clear()
    last_trigger_time.clear()


def _calculate_centroid(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2


def _point_in_polygon(point, polygon):
    if not polygon:
        return False

    x, y = point
    inside = False
    num_points = len(polygon)

    for idx in range(num_points):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % num_points]

        if (y1 > y) != (y2 > y):
            denom = y2 - y1
            if denom == 0:
                continue

            x_intersect = x1 + (y - y1) * (x2 - x1) / denom
            if x < x_intersect:
                inside = not inside

    return inside


def _point_inside_polygon_bounding_box(point, zone):
    if not zone:
        return False

    cx, cy = point
    x1 = zone.get("x1")
    x2 = zone.get("x2")
    y1 = zone.get("y1")
    y2 = zone.get("y2")

    if None in (x1, x2, y1, y2):
        return False

    return x1 <= cx <= x2 and y1 <= cy <= y2


def _point_inside_zone(centroid, zone):
    if zone is None:
        return False

    polygon = zone.get("polygon")
    if polygon:
        return _point_in_polygon(centroid, polygon)

    return _point_inside_polygon_bounding_box(centroid, zone)


def _log_detection(track_id, zone_id, centroid, zone, inside, object_type=None):
    cls_label = object_type or "unknown"
    cx, cy = centroid
    zone_name = zone.get("name", f"Zone {zone_id}")
    print(
        f"[EventDetection] {cls_label} (ID {track_id}) centroid=({cx:.1f},{cy:.1f}) "
        f"zone={zone_name}({zone_id}) inside={inside}"
    )


def _connect():
    return sqlite3.connect(DB_PATH)


def _get_columns(cursor, table_name: str) -> set:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def _ensure_column(cursor, table_name: str, column_name: str, definition: str) -> None:
    columns = _get_columns(cursor, table_name)
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def init_db():
    conn = _connect()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            object_type TEXT,
            track_id INTEGER,
            camera_id INTEGER,
            video_path TEXT,
            frame_number INTEGER,
            frame_start INTEGER,
            frame_end INTEGER,
            video_time REAL,
            zone_id INTEGER,
            event_type TEXT
        )
        """
    )

    _ensure_column(cursor, "events", "frame_number", "INTEGER")
    _ensure_column(cursor, "events", "frame_start", "INTEGER")
    _ensure_column(cursor, "events", "frame_end", "INTEGER")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tracking_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            video_path TEXT,
            frame_number INTEGER NOT NULL,
            track_id INTEGER NOT NULL,
            object_type TEXT NOT NULL,
            bbox_x1 INTEGER NOT NULL,
            bbox_y1 INTEGER NOT NULL,
            bbox_x2 INTEGER NOT NULL,
            bbox_y2 INTEGER NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_tracking_data_unique
        ON tracking_data (camera_id, video_path, frame_number, track_id)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_tracking_lookup
        ON tracking_data (camera_id, video_path, track_id, frame_number)
        """
    )

    conn.commit()
    conn.close()


def detect_event(track_id, bbox, zone, zone_id, object_type=None):
    centroid = _calculate_centroid(bbox)
    current_in_zone = _point_inside_zone(centroid, zone)

    _log_detection(track_id, zone_id, centroid, zone, current_in_zone, object_type)

    track_state = track_history.setdefault(track_id, {})
    zone_state = track_state.setdefault(zone_id, {"was_in_zone": False, "inside_frames": 0})

    if current_in_zone:
        zone_state["inside_frames"] += 1
    else:
        zone_state["inside_frames"] = 0

    event = None

    if not zone_state["was_in_zone"] and zone_state["inside_frames"] >= MIN_ZONE_CONFIRM_FRAMES:
        event = "entering"
        zone_state["was_in_zone"] = True

    elif zone_state["was_in_zone"] and not current_in_zone:
        event = "leaving"
        zone_state["was_in_zone"] = False

    return event


def log_tracking_data(
        track_id: int,
        object_type: str,
        bbox: Tuple[int, int, int, int],
        frame_number: int,
        camera_id: Optional[int],
        video_path: str,
    ) -> None:
    x1, y1, x2, y2 = bbox
    conn = _connect()
    cursor = conn.cursor()
    timestamp = datetime.utcnow().isoformat()

    cursor.execute(
        """
        INSERT OR REPLACE INTO tracking_data
        (camera_id, video_path, frame_number, track_id, object_type,
         bbox_x1, bbox_y1, bbox_x2, bbox_y2, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            camera_id,
            video_path,
            frame_number,
            track_id,
            object_type,
            x1,
            y1,
            x2,
            y2,
            timestamp,
        ),
    )

    conn.commit()
    conn.close()


def get_tracking_data(
        video_path: str,
        track_id: int,
        start_frame: int,
        end_frame: int,
        camera_id: Optional[int] = None,
    ) -> List[Dict]:
    conn = _connect()
    cursor = conn.cursor()

    sql = """
        SELECT frame_number, track_id, object_type,
               bbox_x1, bbox_y1, bbox_x2, bbox_y2
        FROM tracking_data
        WHERE video_path = ?
          AND track_id = ?
          AND frame_number BETWEEN ? AND ?
    """
    params: List[object] = [video_path, track_id, start_frame, end_frame]

    if camera_id is None:
        sql += " AND camera_id IS NULL"
    else:
        sql += " AND camera_id = ?"
        params.append(camera_id)

    sql += " ORDER BY frame_number ASC"

    cursor.execute(sql, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "frame_number": row[0],
            "track_id": row[1],
            "object_type": row[2],
            "bbox": (row[3], row[4], row[5], row[6]),
        }
        for row in rows
    ]


def log_event(
        track_id,
        object_type,
        event_type,
        zone_id,
        camera_id,
        video_time,
        video_path,
        frame_number,
    ):
    cooldown_key = (track_id, zone_id, event_type)
    now = time.time()

    if cooldown_key in last_trigger_time:
        if now - last_trigger_time[cooldown_key] < COOLDOWN:
            return

    last_trigger_time[cooldown_key] = now

    conn = _connect()
    cursor = conn.cursor()

    timestamp = datetime.utcnow().isoformat()

    cursor.execute(
        """
        INSERT INTO events
        (timestamp, object_type, track_id,
         camera_id, video_path, frame_number,
         frame_start, frame_end, video_time, zone_id, event_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            timestamp,
            object_type,
            track_id,
            camera_id,
            video_path,
            frame_number,
            frame_number,
            frame_number,
            video_time,
            zone_id,
            event_type,
        ),
    )

    conn.commit()
    conn.close()

    print(f"🚨 {event_type.upper()} [{object_type}] ID {track_id} in zone {zone_id}")
