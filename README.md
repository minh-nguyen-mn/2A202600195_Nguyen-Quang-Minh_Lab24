# Lab 24 — Full Evaluation & Guardrail System

## Overview

This project implements a production-style evaluation and guardrail stack for a Retrieval-Augmented Generation (RAG) system. The repository includes automated RAGAS evaluation, LLM-as-Judge pipelines with calibration, multi-layer guardrails for input/output safety, and operational blueprint documentation.

The evaluation system supports synthetic test set generation, 4-metric RAGAS scoring, failure clustering, pairwise and absolute LLM judging, Cohen’s kappa calibration, and quantified judge bias analysis. The guardrail stack includes Vietnamese-aware PII redaction, topic validation, adversarial attack filtering, and Llama Guard output moderation.

The final integrated pipeline benchmarks latency across all guardrail layers and documents deployment architecture, SLOs, incident response workflows, and projected operational costs.

---

## Setup

```bash
pip install -r requirements.txt
```

Environment variables:

```bash
export OPENAI_API_KEY=your_key
export GROQ_API_KEY=your_key
```

---

## Repo Structure

```text
phase-a/
phase-b/
phase-c/
phase-d/
.github/workflows/
```

---

## Results Summary

### Phase A — RAGAS

- Synthetic test set generation
- 4-metric evaluation:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall
- Failure cluster analysis
- CI/CD eval gate

### Phase B — LLM-as-Judge

- Pairwise judge with swap-and-average mitigation
- Absolute rubric scoring
- Human calibration with Cohen’s kappa
- Judge bias quantification

### Phase C — Guardrails

- Vietnamese + English PII redaction
- Topic scope validator
- Adversarial attack testing
- Llama Guard moderation
- End-to-end latency benchmark

### Phase D — Blueprint

- SLO definitions
- Defense-in-depth architecture
- Incident playbooks
- Monthly cost analysis

---

## Academic Integrity

All AI-assisted prompts are documented in `prompts.md`.