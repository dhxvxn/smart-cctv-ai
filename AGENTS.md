# Agents Specification - Smart CCTV System

## 1. Detection Agent
- YOLOv8 detection

---

## 2. Tracking Agent

- Assign track_id per object

## ⚠️ CRITICAL CHANGE: TRACK STORAGE

During surveillance, system MUST store:

- track_id
- object_type (LOCKED)
- bounding boxes per frame
- frame number

---

## 3. Event Detection Agent

- Use centroid for zone detection

## ⚠️ EVENT RULE

Each event MUST store:
- track_id
- object_type (LOCKED, never change)
- timestamp
- frame_number

---

## 4. Logging Agent

Store:
- track_id
- object_type
- start_frame
- end_frame
- video_path

---

## 5. Query Agent
- Rule-based parsing

---

## 6. Retrieval Agent
- Return:
  - track_id
  - object_type
  - frame_number

---

## 7. Video Processing Agent

## ⚠️ MAJOR ARCHITECTURE CHANGE

DO NOT re-run tracker during playback

---

## NEW APPROACH: TRACK METADATA PLAYBACK

### During surveillance:
Store per-frame tracking data:

Example:
{
  "frame": 1488,
  "track_id": 78,
  "bbox": [x1,y1,x2,y2],
  "class": "person"
}

---

### During playback:

1. Load stored tracking data
2. Jump directly to frame
3. Draw bounding box using stored data

---

## ⚠️ RULES

- NEVER re-run tracking during playback
- NEVER rely on tracker IDs during playback
- Use ONLY stored metadata

---

## ⚠️ OBJECT CLASS LOCK

- track_id must always map to SAME object_type
- If mismatch → discard detection

---

## ⚠️ HIGHLIGHT RULE

Highlight ONLY:
IF metadata.track_id == selected_track_id

---

## ⚠️ ZONE DISPLAY

- Load zones.json
- Draw zones on every frame

---

## ⚠️ FPS CONTROL

- Playback FPS ≈ 20
- No frame warming
- Direct frame seek