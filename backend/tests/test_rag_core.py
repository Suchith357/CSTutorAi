"""Unit tests: embeddings, vector store (incl. persistence), schemas, retrieval."""
import json

import pytest

from backend.app.rag.embeddings import EmbeddingModel
from backend.app.rag.retriever import Retriever
from backend.app.rag.schemas import DocumentChunk, DocumentMetadata, RetrievedChunk
from backend.app.rag.vector_store import VectorStore

DEADLOCK_TEXT = (
    "Deadlock is a situation where a set of processes are permanently "
    "waiting for resources held by each other."
)


class TestEmbeddingModel:
    def test_encode_documents_returns_normalized_matrix(self):
        model = EmbeddingModel()
        embeddings = model.encode_documents(["hello world", "deadlock in operating systems"])

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == model.dimension
        for row in embeddings:
            assert abs(float(row @ row) - 1.0) < 1e-5  # unit norm

    def test_encode_query_matches_document_space(self):
        model = EmbeddingModel()
        docs = model.encode_documents(["CPU scheduling picks the next process"])
        query = model.encode_query("how is the next process chosen for the CPU")
        assert docs.shape[1] == query.shape[1]

    def test_shared_model_cache(self):
        a = EmbeddingModel()
        b = EmbeddingModel()
        # Two wrappers are distinct objects, but they share the same
        # underlying SentenceTransformer model (the expensive resource).
        assert a.model is b.model
        assert a.model_name == b.model_name


class TestVectorStore:
    def test_add_and_size(self, toy_chunks):
        model = EmbeddingModel()
        embeddings = model.encode_documents([c.text for c in toy_chunks])
        store = VectorStore(embeddings.shape[1])
        store.add(embeddings)
        assert store.size == len(toy_chunks)

    def test_search_returns_top_k_ranked_descending(self, toy_chunks):
        model = EmbeddingModel()
        embeddings = model.encode_documents([c.text for c in toy_chunks])
        store = VectorStore(embeddings.shape[1])
        store.add(embeddings)

        query = model.encode_query("deadlock: processes waiting for each other's resources")
        scores, indices = store.search(query, top_k=3)

        assert len(scores[0]) == 3
        assert scores[0][0] >= scores[0][1] >= scores[0][2]
        # The deadlock sentence is at index 1 in the toy corpus and must rank first
        assert indices[0][0] == 1

    def test_search_fewer_vectors_than_top_k(self, toy_chunks):
        model = EmbeddingModel()
        embeddings = model.encode_documents([c.text for c in toy_chunks[:2]])
        store = VectorStore(embeddings.shape[1])
        store.add(embeddings)

        query = model.encode_query("processes")
        scores, indices = store.search(query, top_k=5)

        valid = [(s, i) for s, i in zip(scores[0], indices[0]) if i != -1]
        assert len(valid) == 2

    def test_save_and_reload_preserves_search(self, toy_chunks, tmp_path):
        model = EmbeddingModel()
        embeddings = model.encode_documents([c.text for c in toy_chunks])

        store = VectorStore(embeddings.shape[1])
        store.add(embeddings)
        store.save(tmp_path)

        reloaded = VectorStore.load(tmp_path)
        assert reloaded.size == store.size
        assert reloaded.dimension == store.dimension

        query = model.encode_query("database management system")
        s1, i1 = store.search(query, top_k=3)
        s2, i2 = reloaded.search(query, top_k=3)
        assert i1[0].tolist() == i2[0].tolist()
        assert s1[0].tolist() == s2[0].tolist()

    def test_load_missing_index_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VectorStore.load(tmp_path)


class TestSchemas:
    def test_metadata_citation(self):
        metadata = DocumentMetadata(
            title="Deadlock", source="toy_corpus.md", topic="operating-systems", page=42
        )
        assert "Deadlock" in metadata.citation()
        assert "p. 42" in metadata.citation()
        assert "toy_corpus.md" in metadata.citation()

    def test_metadata_rejects_empty_title(self):
        with pytest.raises(Exception):
            DocumentMetadata(title="   ", source="toy_corpus.md", topic="toy")

    def test_retrieved_chunk_roundtrip(self, toy_chunks):
        chunk = toy_chunks[0]
        retrieved = RetrievedChunk(text=chunk.text, score=0.9, metadata=chunk.metadata)
        assert RetrievedChunk.model_validate_json(retrieved.model_dump_json()) == retrieved


class TestRetriever:
    def test_retrieve_returns_metadata_and_ranking(self, toy_chunks):
        retriever = Retriever(chunks=toy_chunks, top_k=3)
        results = retriever.retrieve(
            "Why are processes unable to continue when they wait for each other's resources?"
        )

        assert len(results) == 3
        assert all(isinstance(r, RetrievedChunk) for r in results)
        assert results[0].score >= results[1].score >= results[2].score
        # Metadata preserved on every result
        assert all(r.metadata.source == "toy_corpus.md" for r in results)
        assert all(r.metadata.title for r in results)
        # The deadlock chunk must be the top hit
        assert results[0].text == DEADLOCK_TEXT

    def test_retrieve_respects_top_k(self, toy_chunks):
        retriever = Retriever(chunks=toy_chunks, top_k=2)
        results = retriever.retrieve("TCP protocol")
        assert len(results) == 2

    def test_rejects_empty_chunk_list(self):
        with pytest.raises(ValueError):
            Retriever(chunks=[])

    def test_from_prebuilt_matches_in_memory_results(self, toy_chunks, tmp_path):
        from backend.app.core.config import settings as app_settings
        from backend.app.rag.ingestion import save_index

        save_index(
            index_dir=tmp_path,
            chunks=toy_chunks,
            embedding_model=EmbeddingModel(),
            num_documents=1,
            chunk_size=app_settings.chunk_size,
            chunk_overlap=app_settings.chunk_overlap,
        )

        in_memory = Retriever(chunks=toy_chunks, top_k=3)
        prebuilt = Retriever.from_prebuilt(index_dir=tmp_path, top_k=3)

        q = "deadlock between processes"
        r1 = in_memory.retrieve(q)
        r2 = prebuilt.retrieve(q)
        assert [r.text for r in r1] == [r.text for r in r2]
        assert [r.metadata.title for r in r1] == [r.metadata.title for r in r2]

    def test_from_prebuilt_rejects_model_mismatch(self, toy_chunks, tmp_path):
        from backend.app.core.config import settings as app_settings
        from backend.app.rag.ingestion import save_index

        save_index(
            index_dir=tmp_path,
            chunks=toy_chunks,
            embedding_model=EmbeddingModel(),
            num_documents=1,
            chunk_size=app_settings.chunk_size,
            chunk_overlap=app_settings.chunk_overlap,
        )
        # Tamper with the manifest to simulate a model change after ingestion
        manifest_path = tmp_path / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["embedding_model"] = "sentence-transformers/some-other-model"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(ValueError, match="embedding model"):
            Retriever.from_prebuilt(index_dir=tmp_path)
