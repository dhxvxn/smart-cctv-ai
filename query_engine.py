import sqlite3
from typing import Any, Dict, List, Optional, Tuple


class QueryEngine:
    def __init__(self, db_path="events.db"):
        self.db_path = db_path

    def run_query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        user_query: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        filters = filters or {}

        if not filters and user_query:
            filters = self._build_keyword_filters(user_query)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        sql = """
            SELECT track_id, camera_id, zone_id,
                   timestamp, video_time, object_type, event_type,
                   COALESCE(frame_number, frame_start), frame_end, video_path
            FROM events
            WHERE 1=1
        """

        params: List[Any] = []

        object_type = filters.get("object")
        if object_type:
            sql += " AND LOWER(object_type) = ?"
            params.append(object_type.lower())

        event_type = filters.get("event")
        if event_type:
            sql += " AND LOWER(event_type) = ?"
            params.append(event_type.lower())

        zone_id = filters.get("zone_id")
        if isinstance(zone_id, int):
            sql += " AND zone_id = ?"
            params.append(zone_id)

        time_range = filters.get("time_range")
        if time_range:
            start_hour, end_hour = self._normalize_hour_range(time_range)
            clause, range_params = self._time_range_clause(start_hour, end_hour)
            sql += clause
            params.extend(range_params)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()

        return [
            {
                "track_id": r[0],
                "camera_id": r[1],
                "zone_id": r[2],
                "timestamp": r[3],
                "video_time": r[4],
                "object_type": r[5],
                "event_type": r[6],
                "frame_number": r[7],
                "frame_start": r[7],
                "frame_end": r[8],
                "video_path": r[9],
            }
            for r in rows
        ]

    def _build_keyword_filters(self, user_query: str) -> Dict[str, Any]:
        query_lower = user_query.lower()
        filters: Dict[str, Any] = {}

        if "car" in query_lower:
            filters["object"] = "car"

        if "person" in query_lower:
            filters["object"] = "person"

        if "enter" in query_lower:
            filters["event"] = "entering"

        if "leave" in query_lower or "exit" in query_lower:
            filters["event"] = "leaving"

        return filters

    def _normalize_hour_range(self, time_range: Tuple[int, int]) -> Tuple[int, int]:
        start, end = time_range
        start = max(0, min(23, start))
        end = max(0, min(23, end))
        return start, end

    def _time_range_clause(
        self, start_hour: int, end_hour: int
    ) -> Tuple[str, List[int]]:
        if start_hour <= end_hour:
            clause = " AND CAST(strftime('%H', timestamp) AS INTEGER) BETWEEN ? AND ?"
            return clause, [start_hour, end_hour]

        clause = (
            " AND (CAST(strftime('%H', timestamp) AS INTEGER) >= ? "
            "OR CAST(strftime('%H', timestamp) AS INTEGER) <= ?)"
        )
        return clause, [start_hour, end_hour]
