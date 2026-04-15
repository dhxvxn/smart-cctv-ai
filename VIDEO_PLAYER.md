## FIX: OBJECT-ONLY HIGHLIGHT

Do NOT draw border around full frame.

---

## RULE

Highlight ONLY bounding box where:

global_id == target_gid

---

## REMOVE

- any rectangle covering entire frame

---

## RESULT

- clean UI
- only target object highlighted