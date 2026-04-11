# Query System (FINAL FILTERING)

## FILTER RULES

### If query = "car entering"
→ return events where object_type='car'

### If query = "track_id=5"
→ return only that object

### If query = "staying"
→ return events where stayed = TRUE

### If query = "entering"
→ show entry_time only

---

## DISPLAY LOGIC

- entering → show entry_time
- leaving → show exit_time
- staying → show duration