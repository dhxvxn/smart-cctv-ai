import re
from typing import Any, Dict, Optional, Tuple

from zone_manager import get_all_zones


class IntentManager:
    TIME_KEYWORDS = {
        "morning": (6, 11),
        "afternoon": (12, 17),
        "evening": (18, 21),
        "night": (22, 5),
        "day": (6, 18),
        "midnight": (0, 2),
    }

    OBJECT_ALIAS = {
        "person": "person",
        "people": "person",
        "man": "person",
        "woman": "person",
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "bike": "motorcycle",
        "bicycle": "motorcycle",
        "motorcycle": "motorcycle",
    }

    EVENT_ALIAS = {
        "enter": "entering",
        "entered": "entering",
        "entering": "entering",
        "exit": "leaving",
        "leave": "leaving",
        "leaving": "leaving",
    }

    def __init__(self):
        self.intent: Dict[str, Any] = {}
        self.last_query = ""

    def set_intent(self, query: str) -> None:
        self.last_query = query
        self.intent = self._rule_based_parse(query)
        print("🧠 Parsed Intent:", self.intent)

    def get_filters(self) -> Dict[str, Any]:
        filters: Dict[str, Any] = {}

        obj = self.intent.get("object")
        if obj:
            normalized = self._normalize_object(obj)
            if normalized:
                filters["object"] = normalized
        else:
            fallback_obj = self._fallback_object(self.last_query)
            if fallback_obj:
                filters["object"] = fallback_obj

        event = self.intent.get("event")
        if event:
            normalized = self._normalize_event(event)
            if normalized:
                filters["event"] = normalized

        zone = self.intent.get("zone")
        zone_candidate = zone or self._zone_from_query(self.last_query.lower())
        if zone_candidate is not None:
            filters["zone_id"] = zone_candidate

        time_descriptor = self.intent.get("time")
        time_range = self._extract_time_frame(
            time_descriptor or self.last_query
        )
        if time_range:
            filters["time_range"] = time_range

        return filters

    def _rule_based_parse(self, query: str) -> Dict[str, Any]:
        parsed: Dict[str, Any] = {}
        lower_query = query.lower()
        tokens = re.split(r"\W+", lower_query)

        for token in tokens:
            if not token:
                continue

            obj = self.OBJECT_ALIAS.get(token)
            if obj:
                parsed["object"] = obj

            event = self.EVENT_ALIAS.get(token)
            if event:
                parsed["event"] = event

        zone_candidate = self._zone_from_query(lower_query)
        if zone_candidate is not None:
            parsed["zone"] = zone_candidate

        time_range = self._extract_time_frame(lower_query)
        if time_range:
            parsed["time"] = time_range

        if not parsed:
            parsed["query"] = query

        return parsed

    def _zone_from_query(self, query: str) -> Optional[int]:
        direct = self._fallback_zone(query)
        if direct is not None:
            return direct

        keyword_map = {
            "restricted": ["entry", "restricted"],
            "parking": ["parking"],
            "drop": ["drop-off", "dropoff", "drop"],
        }

        zones = get_all_zones()
        if not zones:
            return None

        for keyword, patterns in keyword_map.items():
            if keyword not in query:
                continue

            for zone in zones:
                name = zone.get("name", "").lower()
                if any(pattern in name for pattern in patterns):
                    return zone.get("id")

        return None

    def _normalize_object(self, value: str) -> Optional[str]:
        return self.OBJECT_ALIAS.get(value.lower())

    def _fallback_object(self, query: str) -> Optional[str]:
        tokens = re.split(r"\W+", query.lower())
        for token in tokens:
            normalized = self.OBJECT_ALIAS.get(token)
            if normalized:
                return normalized
        return None

    def _normalize_event(self, value: str) -> Optional[str]:
        return self.EVENT_ALIAS.get(value.lower())

    def _fallback_zone(self, query: str) -> Optional[int]:
        match = re.search(r"zone\s*(\d+)", query)
        if match:
            return int(match.group(1))
        return None

    def _extract_time_frame(self, text: str) -> Optional[Tuple[int, int]]:
        if not text:
            return None

        text_lower = text.lower()
        for keyword, hours in self.TIME_KEYWORDS.items():
            if keyword in text_lower:
                return hours

        range_match = re.search(
            r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\s*(?:to|-)\s*(\d{1,2})(?::(\d{2}))?\s*(am|pm)?",
            text_lower,
        )

        if range_match:
            start_hour = self._to_24h(
                int(range_match.group(1)), range_match.group(3)
            )
            end_hour = self._to_24h(
                int(range_match.group(4)), range_match.group(6)
            )
            return start_hour, end_hour

        single_match = re.search(r"at\s+(\d{1,2})(?:[:](\d{2}))?\s*(am|pm)?", text_lower)
        if single_match:
            hour = self._to_24h(int(single_match.group(1)), single_match.group(3))
            return hour, hour

        return None

    def _to_24h(self, hour: int, suffix: Optional[str]) -> int:
        hour = max(0, min(12, hour)) if hour <= 12 else max(0, min(23, hour))
        if suffix:
            suffix = suffix.lower()
            if suffix == "pm" and hour != 12:
                hour = (hour % 12) + 12
            elif suffix == "am" and hour == 12:
                hour = 0
        return max(0, min(23, hour))
