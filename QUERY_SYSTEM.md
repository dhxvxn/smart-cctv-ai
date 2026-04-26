# Query System (SESSION MODE)

## START BEHAVIOR

On running search_console.py:

Ask:

"Select mode:
1. Single Camera Events
2. Multi Camera Events"

---

## SESSION LOCK

- Store selection as session_mode
- Do NOT ask again during session

---

## QUERY FLOW

WHILE True:

    user_input = query

    filter events based on session_mode

---

## EXIT BEHAVIOR

IF user types:

"exit"

→ terminate program

---

## MODE SWITCH

To change mode:
- restart program

---

## RESULT

- Clean user experience
- No repeated prompts
- Clear event separation