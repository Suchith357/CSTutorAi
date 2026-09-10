from sentence_transformers import SentenceTransformer

from backend.app.core.config import settings

# Models are expensive to load; share one instance per model name per process.
_MODEL_CACHE: dict[str, SentenceTransformer] = {}


def _load_model(model_name: str) -> SentenceTransformer:
    if model_name not in _MODEL_CACHE:
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


class EmbeddingModel:
    """Wraps a sentence-transformers model for document and query encoding."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embedding_model_name
        self.model = _load_model(self.model_name)

    @property
    def dimension(self) -> int:
        return self.model.get_embedding_dimension()

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