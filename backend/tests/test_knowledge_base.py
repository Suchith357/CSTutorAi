"""Knowledge-base coverage tests for the real OSTutorAI corpus.

These tests ingest the actual project-authored documents from data/raw into a
temporary index and verify that every major OS topic is retrievable through
the EXISTING pipeline (loader -> chunker -> embeddings -> FAISS -> retriever).

The toy-corpus fixtures in conftest.py remain for deterministic unit tests;
this module validates the production knowledge base.
"""

import pytest

from backend.app.rag.retriever import Retriever

# Every major OS topic that must be retrievable, with a question that a
# student would realistically ask and the source document expected to win.
TOPIC_COVERAGE = [
    ("What is an operating system and what does the kernel do?", "os-fundamentals"),
    ("What is a process and what is a PCB?", "processes"),
    ("What is a thread and how is it different from a process?", "threads"),
    ("How does Round Robin CPU scheduling work?", "cpu-scheduling"),
    ("What is a semaphore and what is a critical section?", "synchronization"),
    ("What are the four necessary conditions for deadlock?", "deadlocks"),
    ("What is paging and what is a page table?", "memory-management"),
    ("What is virtual memory and what is a page fault?", "virtual-memory"),
    ("How are file blocks allocated on disk?", "file-systems"),
    ("What is DMA and how does interrupt-driven I/O work?", "io"),
    ("How does SCAN disk scheduling work?", "disk-management"),
    ("What is the difference between authentication and authorization?", "protection-security"),
    ("What is a system call and how does fork work?", "system-calls"),
    ("What is a pipe and how does shared-memory IPC work?", "ipc"),
    ("What is a hypervisor and how do containers differ from VMs?", "virtualization"),
]


@pytest.fixture(scope="module")
def kb_retriever(tmp_path_factory):
    """Retriever over the real knowledge base, ingested into a temp index."""
    import shutil

    from backend.app.rag import ingestion

    tmp = tmp_path_factory.mktemp("kb_index")
    ingestion.ingest(
        raw_dir="data/raw",
        index_dir=str(tmp),
    )
    return Retriever.from_prebuilt(index_dir=str(tmp))


class TestKnowledgeBaseCoverage:
    def test_all_fifteen_documents_ingested(self, kb_retriever):
        from backend.app.rag.ingestion import load_chunk_records

        chunks = load_chunk_records(kb_retriever.chunks and "data/index")
        titles = {c.metadata.title for c in chunks}
        assert len(titles) >= 15

    def test_expected_chunk_count(self, kb_retriever):
        # 15 documents with 2-3 chunks each at the configured chunk size
        assert 30 <= kb_retriever.vector_store.size <= 60

    def test_metadata_survives_ingestion(self, kb_retriever):
        for chunk in kb_retriever.chunks:
            assert chunk.metadata.title
            assert chunk.metadata.source == "OSTutorAI project-authored notes"
            assert chunk.metadata.topic
            # front matter must not leak into chunk text
            assert not chunk.text.lstrip().startswith("---")

    @pytest.mark.parametrize("question,expected_topic", TOPIC_COVERAGE)
    def test_major_os_topics_are_retrievable(self, kb_retriever, question, expected_topic):
        results = kb_retriever.retrieve(question)
        assert results, f"No retrieval at all for: {question}"
        top = results[0]
        assert top.metadata.topic == expected_topic, (
            f"Expected topic '{expected_topic}' for {question!r}, "
            f"got '{top.metadata.topic}' (score {top.score:.4f})"
        )

    def test_persistence_round_trip(self, kb_retriever, tmp_path):
        from backend.app.rag.ingestion import save_index
        from backend.app.rag.embeddings import EmbeddingModel

        save_index(
            index_dir=tmp_path,
            chunks=kb_retriever.chunks,
            embedding_model=EmbeddingModel(),
            num_documents=15,
            chunk_size=512,
            chunk_overlap=50,
        )
        reloaded = Retriever.from_prebuilt(index_dir=str(tmp_path))
        r1 = kb_retriever.retrieve("What is deadlock?")
        r2 = reloaded.retrieve("What is deadlock?")
        assert [x.text for x in r1] == [x.text for x in r2]
        assert [x.score for x in r1] == [x.score for x in r2]


class TestGroundingOnRealKnowledgeBase:
    """Grounding gate behavior against the real KB (synthetic scores not used)."""

    def test_clearly_unrelated_question_stays_weak(self, kb_retriever):
        from backend.app.rag.grounding import evaluate_retrieval

        results = kb_retriever.retrieve("What is the capital of France?")
        # The KB has no trivia content; the score must stay low and the gate
        # must reject at the configured threshold.
        assert results
        with pytest.raises(Exception):
            evaluate_retrieval(results, min_score=0.42)

    def test_core_os_question_scores_strongly(self, kb_retriever):
        from backend.app.rag.grounding import evaluate_retrieval

        results = kb_retriever.retrieve("What is a deadlock in operating systems?")
        evaluate_retrieval(results, min_score=0.42)  # must NOT raise
        assert results[0].metadata.topic == "deadlocks"
