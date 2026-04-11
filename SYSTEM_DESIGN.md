# Event System (SESSION-BASED)

## ⚠️ NEW DESIGN

Each object visit = ONE event

---

## EVENT STRUCTURE

Store:

- track_id
- object_type
- zone_id
- entry_time
- exit_time
- duration
- stayed (true/false)

---

## EVENT FLOW

1. Object enters zone → start event
2. Object stays → update duration
3. Object leaves → close event

---

## RESULT

- No duplicate rows
- Clean event representation