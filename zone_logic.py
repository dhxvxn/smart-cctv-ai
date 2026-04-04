# zone_logic.py

track_history = {}

def is_inside_zone(bbox, zone):
    x, y, w, h = bbox
    cx = x + w // 2
    cy = y + h // 2

    return zone["x1"] < cx < zone["x2"] and zone["y1"] < cy < zone["y2"]


def check_entering(track_id, bbox, zone):
    current_in_zone = is_inside_zone(bbox, zone)

    if track_id not in track_history:
        track_history[track_id] = {"was_in_zone": False}

    was_in_zone = track_history[track_id]["was_in_zone"]

    event = None

    if not was_in_zone and current_in_zone:
        event = "entering"

    elif was_in_zone and not current_in_zone:
        event = "leaving"

    track_history[track_id]["was_in_zone"] = current_in_zone

    return event