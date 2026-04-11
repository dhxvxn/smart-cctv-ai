# Event Agent (SESSION MODE)

## ⚠️ RULE

Do NOT log separate events

---

## LOGIC

For each track_id:

### ENTER
IF outside → inside:
    create new event
    store entry_time

---

### STAY
IF inside:
    update duration

---

### EXIT
IF inside → outside:
    update exit_time
    compute duration

    IF duration ≥ 10 sec:
        stayed = TRUE
    ELSE:
        stayed = FALSE

    save event