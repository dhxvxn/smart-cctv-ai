# Global ID Agent (FINAL PRODUCTION)

## OVERVIEW

This module assigns GLOBAL IDs across cameras using:
- ReID embeddings
- Color features
- Temporal constraints
- Track memory

---

## PIPELINE

Detection → Tracking → Feature Extraction → Memory → Matching → GID

---

## FEATURE STRUCTURE

feature = {
    embedding,
    color,
    timestamp,
    camera_id
}

---

## TRACK MEMORY

track_memory = {
    gid: {
        embeddings: last 10,
        colors: last 10,
        last_seen,
        camera_history
    }
}

---

## MATCHING STRATEGY (TWO-STAGE)

### Stage 1: Strong ReID Match

IF cosine_similarity > 0.80:

    assign same global_id
    RETURN

---

### Stage 2: Soft Matching

score =
    0.7 * reid_similarity
  + 0.05 * color_similarity
  - 0.2 * time_penalty

---

## TIME HANDLING

MAX_TIME_DIFF = 10 sec

time_penalty = min(time_diff / 10, 1.0)

---

## CANDIDATE FILTER

Only consider GIDs where:

(now - last_seen) < 10 sec

---

## DECISION

IF score > 0.45:

    assign same global_id

ELSE:

    create new global_id

---

## MEMORY UPDATE

Append embedding + color  
Keep last 10 entries  
Use average for matching

---

## RESULT

- Stable cross-camera identity
- Reduced ID switching
- No over-fragmentation