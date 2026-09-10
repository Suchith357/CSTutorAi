import shutil
from pathlib import Path

import pytest

from backend.app.rag.schemas import DocumentChunk, DocumentMetadata

REPO_ROOT = Path(__file__).resolve().parents[2]
TOY_CORPUS = REPO_ROOT / "data" / "raw" / "toy_corpus.md"

TOY_CHUNK_TEXTS = [
    "A process is a program in execution. A process has its own state and resources.",
    "Deadlock is a situation where a set of processes are permanently waiting for resources held by each other.",
    "CPU scheduling determines which process should be allocated the CPU next.",
    "A database management system is software used to create, manage, and access databases.",
    "TCP is a reliable connection-modified transport layer protocol.",
]


@pytest.fixture
def toy_chunks() -> list[DocumentChunk]:
    """Five chunks with distinct provenance metadata (clearly labeled toy data)."""
    return [
        DocumentChunk(
            text=text,
            metadata=DocumentMetadata(
                title=f"Toy Chunk {i}",
                source="toy_corpus.md",
                topic="toy",
            ),
        )
        for i, text in enumerate(TOY_CHUNK_TEXTS, start=1)
    ]


@pytest.fixture
def tmp_index_dir(tmp_path, monkeypatch) -> str:
    """Redirect the index directory to a per-test temp dir and reset the cached retriever."""
    from backend.app.core import config
    from backend.app.main import state

    index_dir = tmp_path / "index"
    monkeypatch.setattr(config.settings, "index_dir", str(index_dir))
    state.retriever = None  # reset process-wide state between tests
    return str(index_dir)


@pytest.fixture
def ingested_index(tmp_index_dir) -> str:
    """Ingest the toy corpus into the per-test index dir; returns the index path."""
    from backend.app.rag import ingestion

    raw_dir = Path(tmp_index_dir).parent / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(TOY_CORPUS, raw_dir / "toy_corpus.md")

    # Use a small chunk_size so the toy corpus splits into multiple chunks,
    # making the test exercise real multi-chunk retrieval and ranking.
    ingestion.ingest(
        raw_dir=raw_dir,
        index_dir=tmp_index_dir,
        chunk_size=64,
        chunk_overlap=10,
    )
    return tmp_index_dir


@pytest.fixture
def client(ingested_index, monkeypatch):
    """TestClient with lifespan run after ingesting the toy corpus.

    The mock LLM provider is forced so /query never downloads a real model
    during the test suite (the real model is exercised by scripts/demo_query.py).
    """
    from fastapi.testclient import TestClient

    from backend.app import main
    from backend.app.core import config

    monkeypatch.setattr(config.settings, "llm_provider", "mock")
    with TestClient(main.app) as test_client:
        yield test_client


@pytest.fixture
def client_no_index(tmp_index_dir, monkeypatch):
    """TestClient with lifespan run with no index present (retriever must stay unloaded)."""
    from fastapi.testclient import TestClient

    from backend.app import main
    from backend.app.core import config

    monkeypatch.setattr(config.settings, "llm_provider", "mock")
    with TestClient(main.app) as test_client:
        yield test_client
