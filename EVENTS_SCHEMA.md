# Event Schema (DUAL MODE)

## COMMON FIELDS

- global_id
- start_time
- end_time
- duration

---

## SINGLE CAMERA EVENT

- camera_id
- frames

---

## MULTI CAMERA EVENT

- cameras = list of camera_ids
- frames
- transitions

---

## MODE FLAG

event_mode = "single" | "multi"

---

## RULE

IF mode == single:
    log per camera

IF mode == multi:
    merge by global_id

---

## RESULT

- Flexible logging
- Supports both use cases