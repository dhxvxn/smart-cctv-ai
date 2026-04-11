import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().with_name("cctv_logs.db")


def get_db_path() -> str:
    return str(DB_PATH)


def _connect_absolute_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(get_db_path())


def _table_exists(cursor: sqlite3.Cursor, table_name: str) -> bool:
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    )
    return cursor.fetchone() is not None


def _get_columns(cursor: sqlite3.Cursor, table_name: str) -> set:
    cursor.execute(f"PRAGMA table_info({table_name})")
    return {row[1] for row in cursor.fetchall()}


def _ensure_column(
    cursor: sqlite3.Cursor,
    table_name: str,
    column_name: str,
    definition: str,
) -> None:
    columns = _get_columns(cursor, table_name)
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {definition}")


def ensure_valid_schema() -> str:
    try:
        conn = _connect_absolute_db()
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                object_type TEXT,
                track_id INTEGER,
                global_id INTEGER,
                camera_id INTEGER,
                video_path TEXT,
                frame_number INTEGER,
                frame_start INTEGER,
                frame_end INTEGER,
                video_time REAL,
                zone_id INTEGER,
                event_type TEXT,
                mode_type TEXT
            )
            """
        )

        _ensure_column(cursor, "events", "frame_number", "INTEGER")
        _ensure_column(cursor, "events", "frame_start", "INTEGER")
        _ensure_column(cursor, "events", "frame_end", "INTEGER")
        _ensure_column(cursor, "events", "global_id", "INTEGER")
        _ensure_column(cursor, "events", "mode_type", "TEXT")

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS tracking_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id INTEGER,
                video_path TEXT,
                frame_number INTEGER NOT NULL,
                track_id INTEGER NOT NULL,
                global_id INTEGER,
                object_type TEXT NOT NULL,
                bbox_x1 INTEGER NOT NULL,
                bbox_y1 INTEGER NOT NULL,
                bbox_x2 INTEGER NOT NULL,
                bbox_y2 INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        _ensure_column(cursor, "tracking_data", "global_id", "INTEGER")

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
            CREATE INDEX IF NOT EXISTS idx_tracking_global_lookup
            ON tracking_data (global_id, frame_number)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_global_lookup
            ON events (global_id, timestamp)
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_events_mode_timestamp
            ON events (mode_type, timestamp)
            """
        )

        conn.commit()

        required_tables = ("events", "tracking_data")
        missing_tables = [table for table in required_tables if not _table_exists(cursor, table)]
        if missing_tables:
            raise RuntimeError(
                f"Database schema validation failed for {get_db_path()}: missing tables {missing_tables}"
            )

        return get_db_path()
    except sqlite3.Error as exc:
        raise RuntimeError(
            f"Database schema validation failed for {get_db_path()}: {exc}"
        ) from exc
    finally:
        if "conn" in locals():
            conn.close()


def connect_db(validate_schema: bool = True) -> sqlite3.Connection:
    if validate_schema:
        ensure_valid_schema()
    return _connect_absolute_db()
