# Critical Bug Fixes

## 1. Wrong Object Highlighted

Cause:
- Re-tracking changes ID/class

Fix:
- Use stored tracking metadata
- Lock object_type per track_id

---

## 2. Target track_id Not Found

Cause:
- Tracker mismatch

Fix:
- Remove tracker from playback
- Use stored metadata

---

## 3. Slow Playback

Cause:
- Frame warming (0 → N)

Fix:
- Direct frame seek
- No reprocessing

---

## 4. ID Class Mismatch

Cause:
- Tracker reassignment

Fix:
- Lock class per track_id