# Critical Fixes

## 43. Same object different IDs across cameras

### Cause:
No time-based matching

---

## Fix:
Add time proximity condition

---

## RESULT:
Cross-camera identity works

## 44. All detections getting new global_id

### Cause:
Matching too strict due to:
- color penalty
- strict threshold
- no strong ReID fallback

---

### Fix:

1. Add Stage-1 ReID match:
   if similarity > 0.80 → assign same ID

2. Reduce color impact:
   use as bonus, not penalty

3. Increase time window:
   from 3 sec → 10 sec

---

### Result:

- Cross-camera identity restored
- No over-fragmentation