import sqlite3
from datetime import datetime

DB_NAME = "cctv.db"


def init_db():
    print("INIT_DB CALLED")

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    # Person logs
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS person_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER,
            entry_time TEXT,
            exit_time TEXT,
            duration REAL
        )
    """)

    # Cameras
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY,
            name TEXT,
            location TEXT
        )
    """)

    # Zones
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS zones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER,
            name TEXT,
            x1 INTEGER,
            y1 INTEGER,
            x2 INTEGER,
            y2 INTEGER,
            zone_type TEXT
        )
    """)

    # Intrusion logs with video info
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS intrusion_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        track_id INTEGER,
        camera_id INTEGER,
        zone_id INTEGER,
        timestamp TEXT,
        video_file TEXT,
        video_time REAL,
        object_type TEXT
    )
""")

    conn.commit()
    conn.close()


def log_entry(track_id):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    entry_time = datetime.now().isoformat()

    cursor.execute("""
        INSERT INTO person_logs (track_id, entry_time)
        VALUES (?, ?)
    """, (track_id, entry_time))

    conn.commit()
    conn.close()


def log_exit(track_id, entry_time):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    exit_time = datetime.now()
    duration = (exit_time - entry_time).total_seconds()

    cursor.execute("""
        UPDATE person_logs
        SET exit_time = ?, duration = ?
        WHERE track_id = ? AND exit_time IS NULL
    """, (exit_time.isoformat(), duration, track_id))

def log_intrusion(track_id, camera_id, zone_id, video_file, video_time):

    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO intrusion_logs
        (track_id, camera_id, zone_id, timestamp, video_file, video_time)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        track_id,
        camera_id,
        zone_id,
        datetime.now().isoformat(),
        video_file,
        video_time
    ))

    conn.commit()
    conn.close()