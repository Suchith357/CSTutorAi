from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    title: str
    source: str
    topic: str
    url: str | None = None
    license: str | None = None
    page: int | None = None


class DocumentChunk(BaseModel):
    text: str
    metadata: DocumentMetadata