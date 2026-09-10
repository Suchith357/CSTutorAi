# PROJECT_STATUS.md

Last updated: September 10, 2026

## Current Phase

**RAG Retrieval Foundation — COMPLETE**

The retrieval subsystem is now reliable, tested, and end-to-end verified. This is the foundation upon which the LLM layer, tutoring logic, and frontend will be built.

## What Works

### Ingestion Pipeline
```
data/raw → Loader → Chunker → Metadata → Embeddings → FAISS → Persisted Index
```

- **Loader**: `SimpleDirectoryReader` reads documents from `data/raw/`
- **Chunker**: `SentenceSplitter` splits documents into configurable chunks
- **Metadata**: Each chunk carries provenance (title, source, topic, url, license, page)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` produces normalized 384-dim vectors
- **Vector Store**: FAISS `IndexFlatIP` (exact inner-product search)
- **Persistence**: FAISS index + chunk metadata (`metadata.jsonl`) + manifest (`manifest.json`) saved to disk

### Retrieval
```
Query → Embedding → FAISS Search → RetrievedChunks with metadata + scores
```

- Retriever accepts `DocumentChunk` list or loads from persisted index
- Returns `RetrievedChunk` objects with text, similarity score, and full metadata
- Supports `top_k` parameter (default from config)
- Validates embedding model consistency between index and current config

### FastAPI API
- `GET /` — project info
- `GET /health` — health check with index status
- `POST /query` — retrieve source-grounded knowledge for a student's question

### Configuration
All settings are configurable via environment variables (`.env` file):
- `APP_NAME`, `APP_VERSION`, `ENVIRONMENT`, `DEBUG`
- `EMBEDDING_MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`, `RETRIEVAL_TOP_K`
- `RAW_DATA_DIR`, `INDEX_DIR`

### Tests
- 25 tests passing (pytest)
- Covers: embeddings, vector store, schemas, retrieval, ingestion, API endpoints
- Tests run from repository root: `python -m pytest`

### End-to-End Verification
Full pipeline verified:
```
data/raw → Ingestion → Chunks + metadata → Embeddings → FAISS
→ Persistence → Reload → Retriever → FastAPI /query → Source chunks + metadata
```

## Architecture

```
backend/
├── app/
│   ├── main.py              FastAPI app with /query endpoint
│   ├── core/
│   │   └── config.py        Pydantic Settings (env-file backed)
│   ├── rag/
│   │   ├── schemas.py       DocumentMetadata, DocumentChunk, RetrievedChunk
│   │   ├── loader.py        SimpleDirectoryReader wrapper
│   │   ├── chunker.py       SentenceSplitter wrapper
│   │   ├── embeddings.py    SentenceTransformer wrapper with caching
│   │   ├── vector_store.py  FAISS index with persistence
│   │   ├── retriever.py     Retriever (in-memory or from persisted index)
│   │   └── ingestion.py     Full ingestion pipeline + CLI entry point
│   ├── llm/                 (empty — next phase)
│   ├── tutoring/            (empty — future phase)
│   └── models/              (empty — future phase)
├── tests/
│   ├── conftest.py          Shared fixtures (toy chunks, temp dirs)
│   ├── test_rag_core.py     Embeddings, vector store, schemas, retriever
│   └── test_api_and_ingestion.py  Ingestion pipeline, API endpoints
scripts/
├── demo_retrieval.py        Manual retrieval demo
├── demo_retriever.py        Manual retriever demo
└── e2e_test.py              End-to-end pipeline verification
data/
├── raw/                     Source documents (toy corpus + README)
└── index/                   Persisted FAISS index + metadata
requirements.txt             Pinned dependencies
pytest.ini                   Test configuration
```

## Technical Debt Addressed

1. ✅ Reproducible dependencies (`requirements.txt`)
2. ✅ Metadata-preserving retrieval (`RetrievedChunk` with full provenance)
3. ✅ Connected ingestion pipeline (`ingestion.py`)
4. ✅ Persistent FAISS index (`save`/`load` with metadata.jsonl)
5. ✅ Proper pytest test suite (25 tests with assertions)
6. ✅ Ingestion CLI entry point (`python -m backend.app.rag.ingestion`)
7. ✅ FastAPI `/query` endpoint connected to RAG
8. ✅ Configuration expanded (model, chunking, paths)
9. ✅ Deprecated API fixed (`get_embedding_dimension`)

## Not Yet Started

- LLM integration (no models, no prompt templates, no response generation)
- Tutoring behavior (no session management, no teaching logic)
- Personalization (no user accounts, no progress tracking)
- Database (no PostgreSQL, no user models)
- Frontend (no React, no UI)
- Evaluation (no automated metrics, no benchmarks)

## Next Milestone

**LLM Layer Integration** (after which tutoring behavior can begin):
1. Choose LLM provider/model (configurable, not hardcoded)
2. Build prompt templates for CS tutoring
3. Ground LLM responses in retrieved context + citations
4. Add streaming support for real-time tutoring

## Running the Project

```bash
# Install dependencies
pip install -r requirements.txt

# Ingest documents
python -m backend.app.rag.ingestion

# Run tests
python -m pytest

# Start server
uvicorn backend.app.main:app --reload

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is deadlock?"}'
```
