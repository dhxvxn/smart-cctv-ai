# Cross-Camera ReID (FINAL)

## FEATURES USED

1. FastReID embedding
2. Timestamp proximity
3. Motion continuity

---

## MATCH RULE

Match if:

1. cosine_similarity > 0.75
2. abs(time difference) < 3 sec

---

## OPTIONAL BOOST

If object direction matches → increase confidence

---

## RESULT

Same object across cameras → SAME global_id
# Performance Optimization

## FIXES

1. Reduce FPS:
   process every 2–3 frames

2. Batch DB writes:
   store in memory → write once

3. Cache embeddings:
   avoid recomputing per frame

---

## RESULT

Faster logging WITHOUT losing accuracy