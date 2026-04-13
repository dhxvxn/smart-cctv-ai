# Event System (FINAL OPTIMIZATION)

## PROBLEM

Duplicate events + slow logging

---

## SOLUTION

### SESSION-BASED MEMORY

Store events in RAM first

---

## DATABASE WRITE POLICY

Only write when:

→ object leaves zone

---

## BENEFITS

- No duplicate rows
- Massive speed improvement