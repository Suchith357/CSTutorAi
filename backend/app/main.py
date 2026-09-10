from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from backend.app.core.config import settings
from backend.app.rag.retriever import Retriever
from backend.app.rag.schemas import RetrievedChunk


class AppState:
    """Holds process-wide services built at startup."""

    retriever: Retriever | None = None


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the persisted knowledge index at startup; serve retrieval from memory."""
    try:
        state.retriever = Retriever.from_prebuilt()
        print(
            f"Loaded knowledge index: {state.retriever.vector_store.size} chunks "
            f"from {settings.index_path}"
        )
    except FileNotFoundError as exc:
        print(f"Knowledge index not available: {exc}")
        state.retriever = None
    yield


app = FastAPI(
    title=settings.app_name,
    description="An interactive, personalized and source-grounded Computer Science AI tutor.",
    version=settings.app_version,
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def _require_non_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be empty")
        return value


class QueryResponse(BaseModel):
    question: str
    retrieved: list[RetrievedChunk]


@app.get("/")
def root():
    return {
        "project": settings.app_name,
        "status": "running",
        "version": settings.app_version,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "environment": settings.environment,
        "index_loaded": state.retriever is not None,
    }


@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    """
    Retrieve source-grounded knowledge for a student's question.

    Grounded answer generation (LLM layer) is intentionally not integrated yet;
    this endpoint proves question -> retrieval -> cited source chunks.
    """
    if state.retriever is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge index is not loaded. Build it first with: "
                "python -m backend.app.rag.ingestion"
            ),
        )

    retrieved = state.retriever.retrieve(request.question)
    return QueryResponse(question=request.question, retrieved=retrieved)