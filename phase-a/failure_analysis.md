# Failure Cluster Analysis

## Bottom 10 Questions

| # | Type | Cluster | Root Cause |
|---|---|---|---|
| 1 | reasoning | C1 | insufficient retrieval depth |
| 2 | multi_context | C1 | missing cross-document evidence |
| 3 | reasoning | C1 | chunk fragmentation |
| 4 | multi_context | C1 | low recall |
| 5 | reasoning | C2 | semantic mismatch |
| 6 | simple | C2 | noisy retrieval |
| 7 | multi_context | C2 | reranker absent |
| 8 | reasoning | C2 | embedding similarity failure |
| 9 | simple | C3 | hallucinated answer |
| 10 | multi_context | C3 | incomplete retrieved contexts |

---

## Cluster C1 — Multi-hop Retrieval Failure

### Pattern

Questions requiring evidence aggregation across multiple chunks perform poorly.

### Root Cause

Retriever top-k too small and no reranking stage.

### Proposed Fix

- Increase retriever top_k from 3 to 5
- Add reranker layer
- Hybrid BM25 + vector retrieval

---

## Cluster C2 — Semantic Drift

### Pattern

Retrieved chunks partially related but not sufficient.

### Root Cause

Embedding-only retrieval introduces semantic drift.

### Proposed Fix

- Add metadata filters
- Add query expansion
- Use reciprocal rank fusion

---

## Cluster C3 — Hallucination

### Pattern

Generated answers unsupported by contexts.

### Root Cause

Weak grounding constraints.

### Proposed Fix

- Add citation prompting
- Add answer verification stage
- Add hallucination classifier