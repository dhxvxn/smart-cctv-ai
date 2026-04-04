# Application Flow

## Mode 1: Surveillance

- Detect + track
- Store metadata:
  - frame_number
  - track_id
  - bbox
  - class

- Log events

---

## Mode 2: Query

- Parse query
- Fetch events
- Select event

---

## Playback Flow (NEW)

1. Retrieve:
   - video_path
   - frame_number
   - track_id

2. Load tracking metadata

3. Seek directly to frame

4. For each frame:
   - Get stored bbox for track_id
   - Draw RED bounding box
   - Draw zones

5. Play at ~20 FPS

---

## ⚠️ IMPORTANT

- NO tracker execution in playback
- NO frame warming