# Event Database Schema

Table: events

- id
- timestamp
- object_type  ⚠️ LOCKED
- track_id
- frame_number  ⚠️ REQUIRED
- video_path
- zone_id
- event_type

---

## NEW TABLE: tracking_data

- frame_number
- track_id
- object_type
- bbox (x1,y1,x2,y2)

---

## ⚠️ RULE

- track_id must always map to same object_type