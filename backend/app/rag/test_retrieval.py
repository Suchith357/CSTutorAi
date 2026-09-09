from embeddings import EmbeddingModel
from vector_store import VectorStore


documents = [
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

    document_embeddings = embedding_model.encode_documents(documents)

    dimension = document_embeddings.shape[1]

    print(f"Embedding dimension: {dimension}")

    vector_store = VectorStore(dimension)

    vector_store.add(document_embeddings)

    print(f"Documents stored in FAISS: {vector_store.size}")

    query = "What happens when processes wait for resources held by other processes?"

    print(f"\nQuery: {query}")

    query_embedding = embedding_model.encode_query(query)

    scores, indices = vector_store.search(
        query_embedding,
        top_k=3,
    )

    print("\nTop results:\n")

    for rank, (score, index) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        print(f"{rank}. Score: {score:.4f}")
        print(f"   {documents[index]}")
        print()


if __name__ == "__main__":
    main()