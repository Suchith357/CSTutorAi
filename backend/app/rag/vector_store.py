import faiss
import numpy as np


class VectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)

    def add(self, embeddings):
        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        self.index.add(vectors)

    def search(self, query_embedding, top_k: int = 3):
        query_vector = np.asarray(
            query_embedding,
            dtype="float32",
        )

        scores, indices = self.index.search(
            query_vector,
            top_k,
        )

        return scores, indices

    @property
    def size(self):
        return self.index.ntotal