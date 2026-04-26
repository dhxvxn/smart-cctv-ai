# Color Matching Module (FIXED)

## Compute similarity (NOT distance)

color_similarity = 1 - ||c1 - c2||

---

## RULE

IF color_similarity > 0.7:

    boost score by +0.05

ELSE:

    do nothing

---

## IMPORTANT

- NEVER subtract color
- ONLY use as positive boost

---

## RESULT

- No false penalties
- Helps only when reliable