# Debugging Global ID Matching

Print for each match:

- candidate gid
- reid similarity
- color similarity
- time difference
- final score

---

## PURPOSE

- Tune thresholds
- Identify mismatches
- Improve accuracy

---

## EXPECTATION

Good match:
- reid > 0.75
- score > 0.45

Bad match:
- low reid OR high time penalty