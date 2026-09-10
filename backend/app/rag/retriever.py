from pathlib import Path

from backend.app.core.config import settings
from backend.app.rag.embeddings import EmbeddingModel
from backend.app.rag.schemas import DocumentChunk, RetrievedChunk
from backend.app.rag.vector_store import VectorStore


class Retriever:
    """
    Finds the most relevant knowledge (with provenance) for a student's question.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk],
        top_k: int | None = None,
    ):
        if not chunks:
            raise ValueError("Retriever requires at least one document chunk")

        self.chunks = chunks
        self.top_k = top_k or settings.retrieval_top_k
        self.embedding_model = EmbeddingModel()

        document_embeddings = self.embedding_model.encode_documents(
            [chunk.text for chunk in self.chunks]
        )

        self.vector_store = VectorStore(document_embeddings.shape[1])
        self.vector_store.add(document_embeddings)

    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant document chunks for a query.
        """
        query_embedding = self.embedding_model.encode_query(query)

        scores, indices = self.vector_store.search(
            query_embedding,
            top_k=self.top_k,
        )

        results: list[RetrievedChunk] = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue  # FAISS pads results with -1 when fewer than top_k exist
            results.append(
                RetrievedChunk(
                    text=self.chunks[index].text,
                    score=float(score),
                    metadata=self.chunks[index].metadata,
                )
            )

        return results

    @classmethod
    def from_prebuilt(
        cls,
        index_dir: str | Path | None = None,
        top_k: int | None = None,
    ) -> "Retriever":
        """
        Build a retriever from a previously ingested index on disk.
        """
        from backend.app.rag.ingestion import load_chunk_records, load_manifest

        directory = Path(index_dir or settings.index_path)

        records = load_chunk_records(directory)
        if not records:
            raise ValueError(f"No chunk records found in index directory: {directory}")

        retriever = cls(chunks=records, top_k=top_k)

        persisted = VectorStore.load(directory)
        if persisted.dimension != retriever.vector_store.dimension:
            raise ValueError(
                "Persisted index dimension "
                f"({persisted.dimension}) does not match the embedding model "
                f"dimension ({retriever.vector_store.dimension}). "
                "The embedding model may have changed since ingestion; "
                "re-run ingestion."
            )

        manifest = load_manifest(directory)
        if manifest and manifest.get("embedding_model") != retriever.embedding_model.model_name:
            raise ValueError(
                "Persisted index was built with embedding model "
                f"'{manifest.get('embedding_model')}' but the configured model is "
                f"'{retriever.embedding_model.model_name}'. Re-run ingestion."
            )

        retriever.vector_store = persisted
        return retriever