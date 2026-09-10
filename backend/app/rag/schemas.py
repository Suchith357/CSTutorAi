from pydantic import BaseModel, field_validator


class DocumentMetadata(BaseModel):
    """Provenance information attached to every chunk of knowledge."""

    title: str
    source: str
    topic: str
    url: str | None = None
    license: str | None = None
    page: int | None = None
    created_at: str | None = None

    @field_validator("title", "source", "topic")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must be a non-empty string")
        return value

    def citation(self) -> str:
        """Human-readable citation, used later for source-grounded answers."""
        parts = [self.title]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        parts.append(f"(source: {self.source})")
        return " ".join(parts)


class DocumentChunk(BaseModel):
    """A retrievable unit of knowledge together with its provenance."""

    text: str
    metadata: DocumentMetadata


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, with its similarity score."""

    text: str
    score: float
    metadata: DocumentMetadata