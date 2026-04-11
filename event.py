import sqlite3
from datetime import datetime
from typing import Dict, List, Optional, Tuple

DB_PATH = "events.db"

# keeps runtime session state per track + zone
track_history: Dict[int, Dict[int, Dict[str, object]]] = {}
STAYING_THRESHOLD_SECONDS = 10.0
OBJECT_TYPE_ALIASES = {
    "people": "person",
    "man": "person",
    "woman": "person",
    "bike": "motorcycle",
    "bicycle": "motorcycle",
}


def reset_runtime_state() -> None:
    track_history.clear()


def normalize_object_type(object_type: Optional[str]) -> str:
    normalized = (object_type or "").strip().lower()
    return OBJECT_TYPE_ALIASES.get(normalized, normalized)


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
    _ensure_column(cursor, "events", "entry_time", "TEXT")
    _ensure_column(cursor, "events", "exit_time", "TEXT")
    _ensure_column(cursor, "events", "duration", "REAL DEFAULT 0")
    _ensure_column(cursor, "events", "stayed", "INTEGER DEFAULT 0")

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
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_events_session_lookup
        ON events (camera_id, video_path, zone_id, track_id, entry_time, exit_time)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_events_object_type
        ON events (object_type)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_events_track_id
        ON events (track_id)
        """
    )
    cursor.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_events_stayed
        ON events (stayed)
        """
    )

    conn.commit()
    conn.close()


def _utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat()


def _default_zone_state() -> Dict[str, object]:
    return {
        "state": "outside",
        "entry_video_time": None,
        "event_id": None,
    }


def _calculate_duration(entry_video_time: Optional[float], current_video_time: float) -> float:
    if entry_video_time is None:
        return 0.0
    return max(0.0, float(current_video_time) - float(entry_video_time))


def _resolve_open_event_id(
    cursor: sqlite3.Cursor,
    zone_state: Dict[str, object],
    track_id: int,
    zone_id: int,
    camera_id: Optional[int],
    video_path: str,
) -> Optional[int]:
    event_id = zone_state.get("event_id")
    if event_id:
        return int(event_id)

    sql = """
        SELECT id
        FROM events
        WHERE track_id = ?
          AND zone_id = ?
          AND video_path = ?
          AND entry_time IS NOT NULL
          AND exit_time IS NULL
    """
    params: List[object] = [track_id, zone_id, video_path]

    if camera_id is None:
        sql += " AND camera_id IS NULL"
    else:
        sql += " AND camera_id = ?"
        params.append(camera_id)

    sql += " ORDER BY id DESC LIMIT 1"
    cursor.execute(sql, params)
    row = cursor.fetchone()
    if row:
        zone_state["event_id"] = row[0]
        return int(row[0])

    return None


def _create_session_row(
    cursor: sqlite3.Cursor,
    track_id: int,
    object_type: str,
    zone_id: int,
    camera_id: Optional[int],
    video_time: float,
    video_path: str,
    frame_number: int,
) -> int:
    entry_time = _utc_now_iso()
    cursor.execute(
        """
        INSERT INTO events
        (timestamp, object_type, track_id, camera_id, video_path,
         frame_number, frame_start, frame_end, video_time, zone_id,
         event_type, entry_time, exit_time, duration, stayed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry_time,
            object_type,
            track_id,
            camera_id,
            video_path,
            frame_number,
            frame_number,
            frame_number,
            video_time,
            zone_id,
            "entering",
            entry_time,
            None,
            0.0,
            0,
        ),
    )
    return int(cursor.lastrowid)


def _update_session_row(
    cursor: sqlite3.Cursor,
    event_id: int,
    object_type: str,
    frame_number: int,
    video_time: float,
    duration: float,
    stayed: bool,
    event_type: str,
    exit_time: Optional[str] = None,
) -> None:
    cursor.execute(
        """
        UPDATE events
        SET object_type = ?,
            frame_end = ?,
            video_time = ?,
            duration = ?,
            stayed = ?,
            event_type = ?,
            exit_time = COALESCE(?, exit_time)
        WHERE id = ?
        """,
        (
            object_type,
            frame_number,
            video_time,
            duration,
            int(stayed),
            event_type,
            exit_time,
            event_id,
        ),
    )


def update_session_event(
    track_id: int,
    object_type: str,
    bbox: Tuple[int, int, int, int],
    zone: Dict,
    zone_id: int,
    video_time: float,
    camera_id: Optional[int],
    video_path: str,
    frame_number: int,
) -> Optional[str]:
    centroid = _calculate_centroid(bbox)
    current_in_zone = _point_inside_zone(centroid, zone)
    object_type = normalize_object_type(object_type)

    track_state = track_history.setdefault(track_id, {})
    zone_state = track_state.setdefault(zone_id, _default_zone_state())
    current_state = zone_state["state"]

    if current_state == "outside" and current_in_zone:
        zone_state["state"] = "inside"
        zone_state["entry_video_time"] = video_time

        conn = _connect()
        cursor = conn.cursor()
        event_id = _create_session_row(
            cursor,
            track_id,
            object_type,
            zone_id,
            camera_id,
            video_time,
            video_path,
            frame_number,
        )
        conn.commit()
        conn.close()

        zone_state["event_id"] = event_id
        print(f"🚨 ENTERING [{object_type}] ID {track_id} in zone {zone_id}")
        return "entering"

    if current_state == "inside" and current_in_zone:
        duration = _calculate_duration(zone_state.get("entry_video_time"), video_time)
        stayed = duration >= STAYING_THRESHOLD_SECONDS

        conn = _connect()
        cursor = conn.cursor()
        event_id = _resolve_open_event_id(
            cursor,
            zone_state,
            track_id,
            zone_id,
            camera_id,
            video_path,
        )
        if event_id is not None:
            _update_session_row(
                cursor,
                event_id,
                object_type,
                frame_number,
                video_time,
                duration,
                stayed,
                "staying" if stayed else "entering",
            )
            conn.commit()
        conn.close()
        return "staying" if stayed else None

    if current_state == "inside" and not current_in_zone:
        duration = _calculate_duration(zone_state.get("entry_video_time"), video_time)
        stayed = duration >= STAYING_THRESHOLD_SECONDS
        exit_time = _utc_now_iso()

        conn = _connect()
        cursor = conn.cursor()
        event_id = _resolve_open_event_id(
            cursor,
            zone_state,
            track_id,
            zone_id,
            camera_id,
            video_path,
        )
        if event_id is not None:
            _update_session_row(
                cursor,
                event_id,
                object_type,
                frame_number,
                video_time,
                duration,
                stayed,
                "leaving",
                exit_time=exit_time,
            )
            conn.commit()
        conn.close()

        zone_state["state"] = "outside"
        zone_state["entry_video_time"] = None
        zone_state["event_id"] = None
        print(
            f"🚨 LEAVING [{object_type}] ID {track_id} in zone {zone_id} "
            f"| duration={duration:.2f}s | stayed={stayed}"
        )
        return "leaving"

    return None


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
    timestamp = _utc_now_iso()

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
    raise RuntimeError(
        "log_event() is deprecated in session mode; use update_session_event() instead."
    )
