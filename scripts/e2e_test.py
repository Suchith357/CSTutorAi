"""End-to-end verification using FastAPI TestClient.

Demonstrates the full pipeline without needing a separate server process:
    Documents -> Loader -> Chunker -> Metadata -> Embeddings -> FAISS
    -> Persistence -> Reload -> Retriever -> FastAPI /query -> Source + metadata

Run from the repository root:
    python scripts/e2e_test.py
"""

import json
import sys
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    from backend.app.rag import ingestion
    from backend.app.core.config import settings

    print("=" * 60)
    print("CSTutorAI End-to-End Verification")
    print("=" * 60)

    # --- Step 1: Ingestion ---
    print("\n[1] Ingesting toy corpus...")
    report = ingestion.ingest(
        raw_dir=settings.raw_data_path,
        index_dir=settings.index_path,
    )
    print(f"    Documents: {report['documents']}")
    print(f"    Chunks:    {report['chunks']}")
    print(f"    Model:     {report['embedding_model']}")
    print(f"    Dimension: {report['dimension']}")

    # Verify persisted files exist
    index_path = settings.index_path
    for fname in ["index.faiss", "metadata.jsonl", "manifest.json"]:
        assert (index_path / fname).exists(), f"Missing {fname}"
    print(f"    Persisted: index.faiss, metadata.jsonl, manifest.json")

    # --- Step 2: Load persisted chunks ---
    print("\n[2] Loading persisted chunk records...")
    chunks = ingestion.load_chunk_records(index_path)
    print(f"    Loaded {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"      [{i}] {chunk.metadata.source}: {chunk.text[:80]}...")

    # --- Step 3: Build retriever from persisted index ---
    print("\n[3] Building retriever from persisted index...")
    from backend.app.rag.retriever import Retriever
    retriever = Retriever.from_prebuilt(index_dir=index_path)
    print(f"    Index size: {retriever.vector_store.size} vectors")

    # --- Step 4: Retrieve with metadata ---
    print("\n[4] Retrieval test: 'What is deadlock?'")
    results = retriever.retrieve("What is deadlock?")
    print(f"    Retrieved {len(results)} chunks:")
    for i, r in enumerate(results):
        citation = f"{r.metadata.title} (source: {r.metadata.source})"
        print(f"      [{i+1}] score={r.score:.4f}  {citation}")
        print(f"          text={r.text[:100]}...")

    assert len(results) >= 1
    top = results[0]
    assert top.metadata.source, "Source metadata must be present"
    assert top.metadata.title, "Title metadata must be present"

    # --- Step 5: Verify persistence survives reload ---
    print("\n[5] Verifying persistence (reloading index)...")
    reloaded_chunks = ingestion.load_chunk_records(index_path)
    reloaded_retriever = Retriever.from_prebuilt(index_dir=index_path)
    results2 = reloaded_retriever.retrieve("What is deadlock?")
    assert len(results2) >= 1
    print(f"    After reload: {len(results2)} chunks retrieved")
    print(f"    Top source: {results2[0].metadata.source}")

    # --- Step 6: FastAPI /query endpoint ---
    print("\n[6] FastAPI /query endpoint test...")
    from fastapi.testclient import TestClient
    from backend.app.main import app, state
    from backend.app.core import config

    # Reset state to simulate a fresh start
    state.retriever = None
    config.settings.index_dir = str(index_path)

    with TestClient(app) as client:
        # Health check
        health = client.get("/health").json()
        print(f"    Health: {health['status']}, index_loaded={health['index_loaded']}")
        assert health["status"] == "healthy"
        assert health["index_loaded"] is True

        # Query
        response = client.post(
            "/query",
            json={"question": "What is deadlock?"},
        )
        assert response.status_code == 200
        body = response.json()
        print(f"    Question: {body['question']}")
        print(f"    Retrieved {len(body['retrieved'])} chunks:")
        for i, chunk in enumerate(body["retrieved"]):
            citation = f"{chunk['metadata']['title']} (source: {chunk['metadata']['source']})"
            print(f"      [{i+1}] score={chunk['score']:.4f}  {citation}")

        top_api = body["retrieved"][0]
        assert top_api["metadata"]["source"], "API must return source metadata"
        assert top_api["metadata"]["title"], "API must return title metadata"
        assert isinstance(top_api["score"], float), "API must return score"

        # Blank question rejected
        bad = client.post("/query", json={"question": "   "})
        assert bad.status_code == 422
        print("    Blank question correctly rejected (422)")

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)
    print("\nFull pipeline verified:")
    print("  data/raw -> Loader -> Chunker -> Metadata -> Embeddings")
    print("  -> FAISS -> Persistence -> Reload -> Retriever")
    print("  -> FastAPI /query -> Source chunks + metadata")
    return 0


if __name__ == "__main__":
    sys.exit(main())
