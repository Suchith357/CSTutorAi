# PROJECT_STATUS.md

Last updated: September 11, 2026

## Project

**OSTutorAI** — Retrieval-Augmented Large Language Model for Interactive Operating Systems Education.
(Strictly Operating Systems: no DSA/LeetCode, DBMS, networking-as-a-domain, career prep,
frontend, PostgreSQL, Tutor Engine, or fine-tuning.)

## Current Phase

**Curated OS Knowledge Base — COMPLETE (project-authored corpus, index rebuilt, evaluation rerun)**

The toy corpus is no longer the knowledge source. OSTutorAI now runs on a 15-document,
project-authored Operating Systems knowledge base covering the standard B.Tech OS syllabus,
with the existing ingestion → embedding → FAISS → retrieval → grounding pipeline fully reused.

## What Works

### 1. Knowledge Base (NEW this phase)
```
data/raw/*.md  (15 topic documents, front-matter metadata)
  os_fundamentals, processes, threads, cpu_scheduling, synchronization, deadlocks,
  memory_management, virtual_memory, file_systems, io_systems, disk_management,
  protection_security, system_calls, ipc, virtualization
```
- **Source policy (academic honesty)**: all content is *project-authored educational
  material* written for OSTutorAI at B.Tech OS level — no scraped textbooks, no copied
  copyrighted sections, no invented URLs/authors/licenses. `source` metadata reads
  "OSTutorAI project-authored notes"; `license` is intentionally empty (nothing was
  copied from a licensed work). If openly licensed sources are added later, their real
  metadata must be recorded per document.
- The old toy corpus is preserved as a deterministic test fixture at
  `data/raw/fixtures/toy_corpus.md` and is EXCLUDED from ingestion (as is README.md).

### 2. Ingestion & Persistence (reused, small additions)
- `loader.py` now excludes `README*` and `fixtures/*` from knowledge ingestion.
- `ingestion.py` parses simple `--- key: value ---` front matter (title/source/topic/url/
  license) into `DocumentMetadata`; front matter is stripped before chunking so it is
  metadata, never searchable text. No external YAML dependency added.
- Result: 15 documents → 42 chunks → 42 indexed vectors, 384-dim, FAISS IndexFlatIP.

### 3. Retrieval + Grounding + LLM (preserved, unchanged)
- Retriever, grounding gate, safe fallback, grounded prompt, mock/transformers LLM
  providers: untouched this phase; all prior behavior verified intact.

### 4. Knowledge-Base Statistics (NEW)
`scripts/kb_stats.py` reports documents/topics/chunks/vectors/dimension/index type
and per-topic chunk counts from the persisted index (no fabricated numbers).

## Evaluation Before vs After (same 35-question dataset, honest re-labeling)

- `expected_relevance` labels were NOT changed. Only `corpus_support` was re-derived:
  with the real syllabus corpus, the 15 previously uncovered/borderline relevant
  questions (threads, sync, memory, VM, file systems, I/O, round-robin, context
  switch, deadlock conditions...) genuinely became covered. Dataset is schema_version 2
  with the change history recorded in-file.
- Coverage: 9 → **25 reliable positives**; excluded questions: 16 → **0**; negatives 10.

| Metric (threshold 0.42) | Toy corpus (v1) | OS knowledge base (v2) |
|---|---|---|
| reliable positives | 9 | 25 |
| TP / FN / FP / TN | 8 / 1 / 0 / 10 | 24 / 1 / 0 / 10 |
| Precision | 1.000 | 1.000 |
| Recall | 0.889 | 0.960 |
| F1 | 0.941 | 0.980 |

Observed top-1 scores (new KB): deadlock 0.6408, virtual memory 0.6512, IPC/pipe
0.7729, Round Robin 0.6069, paging 0.5289, LRU 0.5318, disk SCAN 0.5121, DMA 0.2753.
Topic-correct top-1: 14/15 probe questions (only "Explain the Banker's algorithm."
retrieves the wrong chunk — see limitations).

## Threshold Re-evaluation (kept at 0.42, with evidence)

Sweep on the new KB (25 positives / 10 negatives):
```
threshold | precision | recall | f1
    0.20  |   0.806   | 1.000  | 0.893
    0.25  |   0.862   | 1.000  | 0.926
    0.30  |   0.893   | 1.000  | 0.943   <- accepts 2-3 irrelevant questions
    0.40  |   0.960   | 0.960  | 0.960   <- accepts 1 irrelevant question
    0.42  |   1.000   | 0.960  | 0.980   <- configured value
    0.45  |   1.000   | 0.960  | 0.980   (identical decisions to 0.42)
    0.50  |   1.000   | 0.880  | 0.936
    0.60+ |   1.000   | <=0.6  | falling
```
RETRIEVAL_MIN_SCORE stays **0.42**: it is the highest-recall threshold with precision
1.000 (zero irrelevant questions accepted). 0.40 would let one irrelevant question
through; 0.45 is decision-identical to 0.42 (changing would be churn). The single FN
is os_io_001 (I/O question scoring 0.3978, just under the gate). 35 questions remain
too few to fine-tune further; recalibrate again after corpus growth.

## Chunking Observation (measured, no tuning)

At chunk_size=512 each document produces 2–3 coherent chunks (avg ~1600 chars) and
14/15 probe queries retrieve the correct topic document. The one observed weakness is
an *embedding* effect, not a chunk-boundary defect: bare named-entity queries
("Explain the Banker's algorithm.") under-match because generic "algorithm" vocabulary
appears in other topics. No CHUNK_SIZE/CHUNK_OVERLAP change was made — the current
settings demonstrably separate topics; per-topic notes (e.g., "Banker's algorithm" as
a heading lead-in) are a corpus-authoring fix, listed as future work.

## Tests

- 71 prior tests remain (grounding, evaluation, RAG core, LLM, API) — all passing.
- NEW `backend/tests/test_knowledge_base.py`: ingests the real `data/raw` corpus into a
  temp index and asserts 15+ documents, metadata integrity (source line, no front-matter
  leakage), correct chunk-count range, top-1 topic correctness for all 15 major OS
  topics, persistence round-trip, and grounding behavior (trivia rejected, core OS
  questions accepted) on the real corpus.
- Test corpus isolation: deterministic unit tests still use `data/raw/fixtures/toy_corpus.md`;
  ingestion excludes fixtures from the production KB.

## Verification Performed (this phase)

- Ingestion: 15 documents / 42 chunks / 42 vectors; manifest rebuilt; reload verified.
- Coverage probes: 14/15 topic-correct retrieval (scores recorded above).
- Evaluation: rerun with schema_version 2 dataset — P 1.000 / R 0.960 / F1 0.980.
- Threshold sweep rerun; 0.42 retained on evidence (see above).
- Full pytest and API verification: see session report (environment note: a Windows
  Application Control policy began blocking `torch` DLL loads mid-session; several
  verification steps that require model inference were completed before the block and
  remaining ones are listed in the report).

## Honest Limitations

- 35 evaluation questions is small; the sweep is indicative, not statistically robust.
- Retrieval is exact cosine over MiniLM embeddings; no reranker. Named-entity-heavy
  queries ("Banker's algorithm") can under-retrieve; the gate then safely refuses
  rather than answering wrongly.
- Content is project-authored, single-reviewer educational material — useful for study,
  but not a replacement for textbooks; no claim of 100% factual accuracy or zero
  hallucinations. Grounding reduces unsupported generation; it cannot eliminate errors.
- The dataset's corpus_support labels are relative to THIS corpus version and must be
  re-validated on any corpus change (documented in the dataset file).

## Not Yet Started (explicitly out of scope)

Reranker/cross-encoder retrieval, larger curated eval set with expected_chunk refs,
answer-groundedness evaluation, Tutor Engine, streaming, frontend, PostgreSQL,
authentication, personalization, fine-tuning.

## Next Milestones

1. Add expected_chunk references + retrieval@k to the evaluation; grow the dataset.
2. Corpus authoring pass: add named-entity lead-ins (e.g., a "Banker's Algorithm"
   heading sentence) where probing showed weak matching; re-run evaluation.
3. Consider an OSS cross-encoder reranker behind the existing retriever interface.
4. Answer-groundedness evaluation harness on top of the dataset.

## Running the Project

```bash
pip install -r requirements.txt
python -m backend.app.rag.ingestion        # rebuilds index from data/raw
python -m pytest
uvicorn backend.app.main:app --reload
.venv/Scripts/python.exe scripts/kb_stats.py
.venv/Scripts/python.exe scripts/evaluate_retrieval.py
python scripts/demo_query.py "What is a deadlock in operating systems?"
```
