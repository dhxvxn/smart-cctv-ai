# System Design - Smart CCTV

## ⚠️ CORE CHANGE: METADATA-DRIVEN PLAYBACK

### OLD (WRONG)
- Re-run tracker during playback ❌

### NEW (CORRECT)
- Use stored tracking metadata ✅

---

## Event Lifecycle

1. Detection
2. Tracking → assign track_id
3. Store tracking metadata per frame
4. Event created
5. Store event with track_id + frame
6. Playback uses stored metadata

---

## ⚠️ ID CONSISTENCY GUARANTEE

track_id must:
- Map to ONE object_type
- Never change class

---

## ⚠️ PERFORMANCE FIX

### Problem:
Playback slow due to tracker warmup

### Solution:
- Directly seek to frame
- Load metadata
- No reprocessing

---

## ⚠️ RESULT

- Instant playback ⚡
- Correct highlighting ✅
- No ID mismatch ✅