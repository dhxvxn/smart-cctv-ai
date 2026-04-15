# Zone System Fix

## ⚠️ PROBLEM

Zones increasing beyond drawn zones

---

## FIX

1. On app start:
   - clear existing zones

2. When saving:
   - overwrite zones.json

3. Assign zone_id sequentially:
   zone_id = 1,2,3,...

---

## RESULT

- no ghost zones
- correct indexing