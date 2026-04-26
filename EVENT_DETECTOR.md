# Event Manager (DUAL MODE SUPPORT)

## INPUT

- global_id
- camera_id
- frame
- mode

---

## LOGIC

IF mode == "single":

    create event per camera_id

ELSE IF mode == "multi":

    IF global_id exists:
        merge event
    ELSE:
        create new event

---

## ACTIVE EVENTS

active_events = {
    gid: {
        start_time,
        last_seen,
        cameras,
        frames
    }
}

---

## FINALIZE

IF now - last_seen > TIMEOUT:

    save event