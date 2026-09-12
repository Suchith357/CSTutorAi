# OSTutorAI

> Retrieval-Augmented Large Language Model for Interactive Operating Systems Education.

## Overview

OSTutorAI (formerly CSTutorAI, scope intentionally narrowed) is a source-grounded AI tutor for **Operating Systems**. It combines a local instruction-tuned LLM with Retrieval-Augmented Generation (RAG) over curated OS educational material, with an explicit **grounding / relevance-control layer**: when retrieval cannot support an answer, the LLM is not called and the system returns a safe insufficient-context response.

Out of scope (by design, for now): DSA/LeetCode coaching, DBMS/Networks tutoring, career preparation, frontend, PostgreSQL, the Tutor Engine, and model fine-tuning.

## Architecture

```
Student Question
      ↓
FastAPI POST /query
      ↓
RAG Retriever (FAISS top-k)
      ↓
Grounding check: retrieved chunks present AND top score >= RETRIEVAL_MIN_SCORE?
      |-- NO  -> safe insufficient-context response (LLM NOT called), grounded=false
      `-- YES -> grounded LLM prompt -> local LLM -> answer + sources, grounded=true
```

## Project Status

✅ **RAG retrieval foundation** — ingestion, chunking, embeddings, persistent FAISS index, metadata-preserving retrieval, FastAPI endpoints; verified end-to-end.

✅ **Local LLM layer** — `Qwen/Qwen2.5-0.5B-Instruct` via HuggingFace transformers (local, free, lazily loaded), grounded prompt with numbered sources and citation rules, replaceable provider interface (`transformers` / `mock`).

✅ **Grounding / relevance control** — configurable retrieval threshold (`RETRIEVAL_MIN_SCORE`); weak or empty retrieval returns a safe fallback and **provably skips LLM generation** (enforced by control-flow tests using a spy LLM).

### Important notes on retrieval scores

- Scores are **cosine similarities** (normalized embeddings + FAISS inner product) in `[-1, 1]`. They are **not** percentages or probabilities, and not a measure of answer correctness.
- `RETRIEVAL_MIN_SCORE=0.42` is an **engineering starting point** observed on the current toy corpus (OS topics ≈ 0.53–0.69, off-topic ≈ 0.29 and below). It is **not** scientifically calibrated; calibration against a labeled evaluation dataset is future work.

### Current corpus

The knowledge base is a small **toy corpus** (`data/raw/toy_corpus.md`) used only for engineering validation of the pipeline. The final curated OS knowledge base (licensed sources with provenance metadata) is the next phase.

## Testing

53 tests pass (`python -m pytest` from the repository root). The suite covers:

- embeddings, vector store, persistence, schemas, retriever
- ingestion pipeline and index round-trips
- grounded prompt construction
- grounding decisions: empty retrieval, below/at/above threshold, settings-driven threshold
- **weak retrieval → LLM generate() is never called** (spy-verified)
- FastAPI endpoints (`/`, `/health`, `/query`), including the grounded/insufficient paths

Tests use the `mock` LLM provider and synthetic chunks — no model downloads, no network dependency in CI.

## Running the Project

```bash
# Install dependencies (Python 3.12)
pip install -r requirements.txt

# Ingest documents (currently the toy corpus)
python -m backend.app.rag.ingestion

# Run tests
python -m pytest

# Start server
uvicorn backend.app.main:app --reload

# Query (strong OS question -> grounded answer)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is a deadlock in operating systems?"}'

# Query (off-topic question -> safe fallback, grounded=false, no LLM call)
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the difference between TCP and UDP."}'
```

`scripts/demo_query.py "<question>"` prints retrieval scores, the grounding decision, and whether the LLM was called. `scripts/e2e_test.py` verifies the full pipeline including persistence.

## Configuration

All settings are configurable via environment variables (see `.env.example`):

- Application: `APP_NAME`, `APP_VERSION`, `ENVIRONMENT`, `DEBUG`
- RAG: `EMBEDDING_MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`
- Grounding: `RETRIEVAL_MIN_SCORE`
- Data: `RAW_DATA_DIR`, `INDEX_DIR`
- LLM: `LLM_PROVIDER`, `LLM_MODEL_NAME`, `LLM_MAX_NEW_TOKENS`

## Honest limitations

- Grounding reduces unsupported generation; it does **not** guarantee zero hallucinations or 100% accuracy.
- The threshold is a heuristic baseline, not a calibrated constant; scores depend on the embedding model, corpus, chunking, and metric.
- The toy corpus is not representative of a real OS curriculum.

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for the current phase detail and next milestones.

## Retrieval Evaluation Framework

An offline evaluation measures how well the grounding gate separates real Operating Systems
questions from everything else, using a version-controlled labeled dataset:
`data/evaluation/os_retrieval_eval.json` (35 questions across 10 OS topics + non-OS controls).

```bash
.venv/Scripts/python.exe scripts/evaluate_retrieval.py            # terminal report
.venv/Scripts/python.exe scripts/evaluate_retrieval.py --json out # machine-readable
```

- The script reuses the existing persisted index and the existing `Retriever` — retrieval is
  evaluated, never duplicated, and no LLM is called.
- Each question has two labels: `expected_relevance` (relevant / borderline / irrelevant) and
  `corpus_support` (covered / partial / absent). Only relevant+covered questions and irrelevant
  questions enter precision/recall/F1; borderline and corpus-uncovered questions are reported
  separately instead of being silently counted — no fake ground truth.
- Outputs: confusion matrix (TP/FN/FP/TN), precision, recall, F1, and an 11-point threshold
  sweep with an evidence-based recommendation.
- The recommendation policy prefers **perfect precision** (for a tutoring gate, a false accept —
  a confident answer without source support — is the worst outcome), and never recommends
  config churn without a strictly better confusion matrix.

**Current result (toy corpus):** at `RETRIEVAL_MIN_SCORE=0.42` → Precision 1.000, Recall 0.889,
F1 0.941 (TP 8, FN 1, FP 0, TN 10). The sweep's best-F1 value (0.30) accepts one irrelevant
question, so **0.42 is kept**. These numbers are an engineering baseline on the toy corpus and
must be re-established after the curated OS knowledge base replaces it. Scores are cosine
similarities, not probabilities or percentages.
