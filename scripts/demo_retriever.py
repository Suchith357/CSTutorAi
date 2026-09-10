"""Manual demo: metadata-preserving retrieval via the Retriever.

This is an interactive demo, not a test. The real test suite lives in backend/tests.

Run from the repository root:

    python scripts/demo_retriever.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.rag.retriever import Retriever  # noqa: E402
from backend.app.rag.schemas import DocumentChunk, DocumentMetadata  # noqa: E402

# Toy corpus with explicit provenance metadata (clearly labeled; not real course material).
CHUNKS = [
    DocumentChunk(
        text="A process is a program in execution. A process has its own state and resources.",
        metadata=DocumentMetadata(title="Processes", source="toy_corpus.md", topic="operating-systems"),
    ),
    DocumentChunk(
        text="Deadlock is a situation where a set of processes are permanently waiting for resources held by each other.",
        metadata=DocumentMetadata(title="Deadlock", source="toy_corpus.md", topic="operating-systems"),
    ),
    DocumentChunk(
        text="CPU scheduling determines which process should be allocated the CPU next.",
        metadata=DocumentMetadata(title="CPU Scheduling", source="toy_corpus.md", topic="operating-systems"),
    ),
    DocumentChunk(
        text="A database management system is software used to create, manage, and access databases.",
        metadata=DocumentMetadata(title="DBMS", source="toy_corpus.md", topic="databases"),
    ),
    DocumentChunk(
        text="TCP is a reliable connection-oriented transport layer protocol.",
        metadata=DocumentMetadata(title="TCP", source="toy_corpus.md", topic="networking"),
    ),
]


def main():
    print("Creating retriever...")
    retriever = Retriever(chunks=CHUNKS, top_k=3)

    query = "Why are processes unable to continue when they are waiting for each other's resources?"
    print(f"\nQuestion:\n{query}")

    results = retriever.retrieve(query)

    print("\nRetrieved knowledge:\n")
    for rank, result in enumerate(results, start=1):
        print(f"{rank}. Similarity: {result.score:.4f}")
        print(f"   Text: {result.text}")
        print(f"   Citation: {result.metadata.citation()}\n")


if __name__ == "__main__":
    main()
