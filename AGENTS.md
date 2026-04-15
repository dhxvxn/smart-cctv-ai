# Global ID Agent (ReID Upgrade)

## PROCESS

For each detection:

1. extract FastReID embedding

2. compare with global pool

3. compute:
   - cosine similarity
   - time difference

---

## MATCH CONDITION

IF similarity > 0.75
AND time_diff < 3 sec:

    assign same global_id

ELSE:
    create new global_id