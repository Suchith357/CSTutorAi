"""Manual demo: embeddings + vector store (no metadata), kept from the original audit phase.

This is an interactive demo, not a test. The real test suite lives in backend/tests.

Run from the repository root:

    python scripts/demo_retrieval.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.rag.embeddings import EmbeddingModel  # noqa: E402
from backend.app.rag.vector_store import VectorStore  # noqa: E402

# Toy corpus (clearly labeled as such; not real course material).
DOCUMENTS = [
    "A process is a program in execution. A process has its own state and resources.",
    "Deadlock is a situation where a set of processes are permanently waiting for resources held by each other.",
    "CPU scheduling determines which process should be allocated the CPU next.",
    "A database management system is software used to create, manage, and access databases.",
    "TCP is a reliable connection-oriented transport layer protocol.",
]


def main():
    print("Loading embedding model...")
    embedding_model = EmbeddingModel()

    print("Creating document embeddings...")
    document_embeddings = embedding_model.encode_documents(DOCUMENTS)
    print(f"Embedding dimension: {document_embeddings.shape[1]}")

    vector_store = VectorStore(document_embeddings.shape[1])
    vector_store.add(document_embeddings)
    print(f"Documents stored in FAISS: {vector_store.size}")

    query = "What happens when processes wait for resources held by other processes?"
    print(f"\nQuery: {query}")

    query_embedding = embedding_model.encode_query(query)
    scores, indices = vector_store.search(query_embedding, top_k=3)

    print("\nTop results:\n")
    for rank, (score, index) in enumerate(zip(scores[0], indices[0]), start=1):
        print(f"{rank}. Score: {score:.4f}")
        print(f"   {DOCUMENTS[index]}\n")


if __name__ == "__main__":
    main()
