# Surveillance Flow (FINAL FIX)

## ⚠️ CRITICAL RULE

Surveillance must run ONLY ONCE

---

## CURRENT ISSUE

Video restarts after ending → causes duplicate logging

---

## FIX

When video ends:

IF ret == False:
    break

---

## DO NOT

- Reset frame position to 0
- Reopen video
- Loop video in surveillance mode

---

## RESULT

- Each video processed only once
- No duplicate logs