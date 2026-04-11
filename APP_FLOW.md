# Logging Flow (FINAL)

For each track_id:

1. Check zone (inside/outside)

2. Compare previous state

3. Apply logic:

- outside → inside → ENTERING
- inside ≥ 10 sec → STAYING
- inside → outside → LEAVING

4. Log event