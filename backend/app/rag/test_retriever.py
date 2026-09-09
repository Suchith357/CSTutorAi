from backend.app.rag.retriever import Retriever


documents = [
    "A process is a program in execution. A process has its own state and resources.",
    "Deadlock is a situation where a set of processes are permanently waiting for resources held by each other.",
    "CPU scheduling determines which process should be allocated the CPU next.",
    "A database management system is software used to create, manage, and access databases.",
    "TCP is a reliable connection-oriented transport layer protocol.",
]


def main():
    print("Creating retriever...")

    retriever = Retriever(
        documents=documents,
        top_k=3,
    )

    query = "Why are processes unable to continue when they are waiting for each other's resources?"

    print(f"\nQuestion:\n{query}")

    results = retriever.retrieve(query)

    print("\nRetrieved knowledge:\n")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. Similarity: {result['score']:.4f}")
        print(f"   {result['text']}")
        print()


if __name__ == "__main__":
    main()