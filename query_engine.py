import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


class QueryEngine:
    OBJECT_KEYWORDS = {
        "person": "person",
        "people": "person",
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "bike": "motorcycle",
        "bicycle": "motorcycle",
        "motorcycle": "motorcycle",
    }

    def __init__(self, db_path="events.db"):
        self.db_path = db_path

    def run_query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        user_query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}

        if not filters and user_query:
            filters = self._build_keyword_filters(user_query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        display_mode = self._resolve_display_mode(filters)

        sql = """
            SELECT track_id, camera_id, zone_id,
                   entry_time, exit_time, duration, stayed,
                   object_type, event_type,
                   frame_start, frame_end, video_time, video_path, timestamp
            FROM events
            WHERE entry_time IS NOT NULL
        """

        params: List[Any] = []

        object_type = filters.get("object_type") or filters.get("object")
        if object_type:
            sql += " AND LOWER(object_type) = ?"
            params.append(object_type.lower())

        event_type = filters.get("event")
        if event_type == "staying":
            sql += " AND stayed = 1"
        elif event_type == "leaving":
            sql += " AND exit_time IS NOT NULL"

        zone_id = filters.get("zone_id")
        if isinstance(zone_id, int):
            sql += " AND zone_id = ?"
            params.append(zone_id)

        track_id = filters.get("track_id")
        if isinstance(track_id, int):
            sql += " AND track_id = ?"
            params.append(track_id)

        time_range = filters.get("time_range")
        if time_range:
            start_hour, end_hour = self._normalize_hour_range(time_range)
            clause, range_params = self._time_range_clause(start_hour, end_hour)
            sql += clause
            params.extend(range_params)

        sql += " ORDER BY COALESCE(exit_time, entry_time, timestamp) DESC"

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        results: List[Dict[str, Any]] = []
        for row in rows:
            frame_start = row[9]
            frame_end = row[10]
            frame_number = frame_end if display_mode == "leaving" and frame_end is not None else frame_start
            display_label, display_value = self._display_value(display_mode, row[3], row[4], row[5])

            results.append(
                {
                    "track_id": row[0],
                    "camera_id": row[1],
                    "zone_id": row[2],
                    "entry_time": row[3],
                    "exit_time": row[4],
                    "duration": row[5],
                    "stayed": bool(row[6]),
                    "object_type": row[7],
                    "event_type": row[8],
                    "frame_number": frame_number if frame_number is not None else frame_start,
                    "frame_start": frame_start,
                    "frame_end": frame_end,
                    "video_time": row[11],
                    "video_path": row[12],
                    "timestamp": row[13] or row[3],
                    "display_mode": display_mode,
                    "display_label": display_label,
                    "display_value": display_value,
                }
            )

        return results

    def _build_keyword_filters(self, user_query: str) -> Dict[str, Any]:
        query_lower = user_query.lower()
        filters: Dict[str, Any] = {}

        for keyword, normalized in self.OBJECT_KEYWORDS.items():
            if keyword in query_lower:
                filters["object"] = normalized

        track_match = re.search(r"track(?:_?id)?\s*[:=#]?\s*(\d+)", query_lower)
        if track_match:
            filters["track_id"] = int(track_match.group(1))

        if "enter" in query_lower:
            filters["event"] = "entering"

        if "stay" in query_lower:
            filters["event"] = "staying"

        if "leave" in query_lower or "exit" in query_lower:
            filters["event"] = "leaving"

        return filters

    def _resolve_display_mode(self, filters: Dict[str, Any]) -> str:
        event_type = filters.get("event")
        if event_type in {"entering", "leaving", "staying"}:
            return event_type
        return "entering"

    def _display_value(
        self,
        display_mode: str,
        entry_time: Optional[str],
        exit_time: Optional[str],
        duration: Optional[float],
    ) -> Tuple[str, str]:
        if display_mode == "leaving":
            return "exit_time", exit_time or "-"

        if display_mode == "staying":
            return "duration", self._format_duration(duration)

        return "entry_time", entry_time or "-"

    def _format_duration(self, duration: Optional[float]) -> str:
        if duration is None:
            return "0.0s"
        return f"{float(duration):.1f}s"

    def _normalize_hour_range(self, time_range: Tuple[int, int]) -> Tuple[int, int]:
        start, end = time_range
        start = max(0, min(23, start))
        end = max(0, min(23, end))
        return start, end

    def _time_range_clause(
        self, start_hour: int, end_hour: int
    ) -> Tuple[str, List[int]]:
        if start_hour <= end_hour:
            clause = (
                " AND CAST(strftime('%H', COALESCE(entry_time, timestamp)) AS INTEGER)"
                " BETWEEN ? AND ?"
            )
            return clause, [start_hour, end_hour]

        clause = (
            " AND (CAST(strftime('%H', COALESCE(entry_time, timestamp)) AS INTEGER) >= ? "
            "OR CAST(strftime('%H', COALESCE(entry_time, timestamp)) AS INTEGER) <= ?)"
        )
        return clause, [start_hour, end_hour]
