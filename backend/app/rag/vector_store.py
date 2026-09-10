from pathlib import Path

import faiss
import numpy as np


class VectorStore:
    """FAISS-backed vector store with disk persistence."""

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

    def save(self, directory: str | Path):
        """Persist the FAISS index to disk."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path / "index.faiss"))

    @classmethod
    def load(cls, directory: str | Path):
        """Load a previously persisted FAISS index."""
        path = Path(directory) / "index.faiss"
        if not path.exists():
            raise FileNotFoundError(f"No persisted index found at: {path}")

        index = faiss.read_index(str(path))

        store = cls(index.d)
        store.index = index
        return store