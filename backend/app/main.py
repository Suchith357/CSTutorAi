from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator

from backend.app.core.config import settings
from backend.app.llm.client import LLMClient, create_llm_client
from backend.app.llm.prompts import build_grounded_prompt
from backend.app.rag.grounding import (
    INSUFFICIENT_CONTEXT_ANSWER,
    InsufficientContextError,
    evaluate_retrieval,
)
from backend.app.rag.retriever import Retriever
from backend.app.rag.schemas import RetrievedChunk


class AppState:
    """Holds process-wide services built at startup."""

    retriever: Retriever | None = None
    llm: LLMClient | None = None


state = AppState()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Load the persisted knowledge index and the LLM client at startup."""
    try:
        state.retriever = Retriever.from_prebuilt()
        print(
            f"Loaded knowledge index: {state.retriever.vector_store.size} chunks "
            f"from {settings.index_path}"
        )
    except FileNotFoundError as exc:
        print(f"Knowledge index not available: {exc}")
        state.retriever = None

    state.llm = create_llm_client()  # model loads lazily on first query
    print(f"LLM provider: {settings.llm_provider} (model: {settings.llm_model_name})")
    yield


app = FastAPI(
    title=settings.app_name,
    description=(
        "An interactive, source-grounded AI tutor for Operating Systems education."
    ),
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
    answer: str
    retrieved: list[RetrievedChunk]
    grounded: bool


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


def _answer_with_grounding(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Generate an answer from retrieval, or raise InsufficientContextError.

    Kept out of the route body so the grounding decision can be unit tested
    directly.
    """
    evaluate_retrieval(retrieved)
    system, user = build_grounded_prompt(
        question,
        retrieved,
        retrieval_min_score=settings.retrieval_min_score,
    )
    return state.llm.generate(system, user)


@app.post("/query")
def query(request: QueryRequest) -> QueryResponse:
    """
    Grounded tutoring: question -> retrieval -> relevance check -> LLM -> answer + sources.

    When retrieval is empty or too weak (best score below
    settings.retrieval_min_score), the LLM is NOT called and a safe
    insufficient-context response is returned with grounded=False.
    """
    if state.retriever is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Knowledge index is not loaded. Build it first with: "
                "python -m backend.app.rag.ingestion"
            ),
        )
    if state.llm is None:
        raise HTTPException(status_code=503, detail="LLM client is not initialized")

    retrieved = state.retriever.retrieve(request.question)

    try:
        answer = _answer_with_grounding(request.question, retrieved)
    except InsufficientContextError as exc:
        return QueryResponse(
            question=request.question,
            answer=INSUFFICIENT_CONTEXT_ANSWER,
            retrieved=exc.retrieved,
            grounded=False,
        )

    return QueryResponse(
        question=request.question,
        answer=answer,
        retrieved=retrieved,
        grounded=True,
    )