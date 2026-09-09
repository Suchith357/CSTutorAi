from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_NAME)

    def encode_documents(self, texts: list[str]):
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def encode_query(self, query: str):
        return self.model.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )