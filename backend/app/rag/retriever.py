from .embeddings import EmbeddingModel
from .vector_store import VectorStore


class Retriever:
    """
    Finds the most relevant knowledge for a student's question.
    """

    def __init__(
        self,
        documents: list[str],
        top_k: int = 3,
    ):
        self.documents = documents
        self.top_k = top_k

        self.embedding_model = EmbeddingModel()

        document_embeddings = self.embedding_model.encode_documents(
            documents
        )

        dimension = document_embeddings.shape[1]

        self.vector_store = VectorStore(dimension)

        self.vector_store.add(document_embeddings)

    def retrieve(self, query: str):
        """
        Retrieve the most relevant documents for a query.
        """

        query_embedding = self.embedding_model.encode_query(query)

        scores, indices = self.vector_store.search(
            query_embedding,
            top_k=self.top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            results.append(
                {
                    "text": self.documents[index],
                    "score": float(score),
                }
            )

        return results