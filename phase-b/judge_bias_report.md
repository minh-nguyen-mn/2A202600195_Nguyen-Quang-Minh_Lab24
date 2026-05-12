# Judge Bias Report

## Position Bias

| Metric | Value |
|---|---|
| A wins when first | 58% |

Observation:

Judge shows mild position bias toward first-listed answer.

Mitigation:

- swap-and-average
- randomized ordering

---

## Length Bias

| Metric | Value |
|---|---|
| Longer answer preferred | 64% |

Observation:

Longer answers are disproportionately selected.

Mitigation:

- conciseness rubric
- token normalization
- response truncation before judging