# Event Table Schema (UPDATED)

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    object_type TEXT,
    zone_id INTEGER,
    entry_time TEXT,
    exit_time TEXT,
    duration REAL,
    stayed BOOLEAN,
    video_path TEXT,
    camera_id INTEGER,
    mode_type TEXT
);