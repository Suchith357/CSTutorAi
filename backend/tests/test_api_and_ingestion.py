"""End-to-end tests: ingestion pipeline, chunk reload, and the FastAPI /query endpoint."""
import pytest

from backend.app.rag import ingestion
from backend.app.rag.schemas import DocumentChunk


def test_document_to_chunks_attaches_metadata():
    from llama_index.core import Document

    from backend.app.rag.ingestion import document_to_chunks

    document = Document(
        text="Deadlock is a situation where processes wait on each other's resources.",
        metadata={
            "file_path": "toy_corpus.md",
            "title": "Toy Corpus",
            "topic": "operating-systems",
            "source": "toy_corpus.md",
            "license": "CC0",
            "page": 3,
        },
    )

    chunks = document_to_chunks(document, chunk_size=512, chunk_overlap=50)

    assert len(chunks) >= 1
    for chunk in chunks:
        assert isinstance(chunk, DocumentChunk)
        assert chunk.metadata.title == "Toy Corpus"
        assert chunk.metadata.topic == "operating-systems"
        assert chunk.metadata.license == "CC0"
        assert chunk.metadata.page == 3
        assert chunk.text.strip()


def test_document_to_chunks_empty_nodes_produce_no_chunks():
    from llama_index.core import Document

    from backend.app.rag.ingestion import document_to_chunks

    document = Document(text="", metadata={"file_path": "empty.md", "topic": "t"})
    assert document_to_chunks(document, chunk_size=512, chunk_overlap=50) == []


def test_ingest_persists_index_and_records(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "toy.md").write_text(
        "Deadlock is a situation where a set of processes are permanently "
        "waiting for resources held by each other.\n\n"
        "TCP is a reliable connection-oriented transport layer protocol.",
        encoding="utf-8",
    )
    index_dir = tmp_path / "index"

    report = ingestion.ingest(
        raw_dir=raw_dir,
        index_dir=index_dir,
        chunk_size=64,
        chunk_overlap=10,
    )

    assert report["chunks"] >= 2
    assert report["documents"] == 1
    assert (index_dir / "index.faiss").exists()
    assert (index_dir / "metadata.jsonl").exists()
    assert (index_dir / "manifest.json").exists()

    # Chunk records must round-trip with metadata intact
    records = ingestion.load_chunk_records(index_dir)
    assert len(records) == report["chunks"]
    assert all(r.metadata.source == "toy.md" for r in records)
    assert all(r.metadata.title == "toy" for r in records)

    manifest = ingestion.load_manifest(index_dir)
    assert manifest["num_chunks"] == report["chunks"]
    assert manifest["embedding_model"].startswith("sentence-transformers/")
    assert manifest["dimension"] > 0


def test_ingest_empty_raw_dir_raises(tmp_path):
    raw_dir = tmp_path / "empty_raw"
    raw_dir.mkdir()
    with pytest.raises(Exception):
        ingestion.ingest(raw_dir=raw_dir, index_dir=tmp_path / "idx")


def test_load_chunk_records_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ingestion.load_chunk_records(tmp_path)


def test_query_endpoint_returns_metadata(ingested_index, client):
    response = client.post(
        "/query",
        json={"question": "Why are processes unable to continue when they wait for each other's resources?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["question"].startswith("Why are processes")
    assert len(body["retrieved"]) >= 1

    top = body["retrieved"][0]
    # The top result must be relevant to deadlock (retrieval correctness)
    assert "deadlock" in top["text"].lower(), (
        f"Expected deadlock-related text, got: {top['text'][:120]}"
    )
    # Source-grounded contract: every retrieved chunk carries provenance
    assert top["metadata"]["source"] == "toy_corpus.md"
    assert top["metadata"]["title"]  # non-empty title
    assert isinstance(top["score"], float)
    # A citation must be derivable from the API response
    citation = f"{top['metadata']['title']} (source: {top['metadata']['source']})"
    assert "toy_corpus.md" in citation


def test_query_endpoint_rejects_blank_question(client):
    response = client.post("/query", json={"question": "   "})
    assert response.status_code == 422


def test_health_reports_index_state(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["index_loaded"] is True


def test_query_without_index_returns_503(client_no_index):
    response = client_no_index.post("/query", json={"question": "What is deadlock?"})
    assert response.status_code == 503
    assert "ingestion" in response.json()["detail"]
