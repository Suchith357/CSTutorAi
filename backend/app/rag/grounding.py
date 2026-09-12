"""Retrieval relevance / grounding evaluation for OSTutorAI.

A retrieval score is a cosine similarity between the normalized query embedding
and normalized chunk embeddings (FAISS inner product over unit vectors). It is
NOT a probability or a percentage of answer quality - it only expresses how
close the retrieved text is to the question in embedding space.

evaluate_retrieval() is the single decision point used by the /query endpoint:
it rejects empty retrieval or top scores below settings.retrieval_min_score
BEFORE any LLM call is made, so weakly supported questions never reach
generation.
"""

from backend.app.rag.schemas import RetrievedChunk

# Wording of the safe fallback returned when retrieval is not sufficient.
# Deliberately honest, OS-specific, and not phrased like a server error.
INSUFFICIENT_CONTEXT_ANSWER = (
    "I don't have enough information in the available Operating Systems "
    "sources to answer this question reliably. The retrieved material does "
    "not cover this topic in sufficient depth, so I won't guess. Please try "
    "rephrasing the question, or ask about a different Operating Systems "
    "topic that is covered by the available sources."
)


class InsufficientContextError(Exception):
    """Raised when retrieval cannot support a grounded answer.

    Attributes:
        retrieved: whatever chunks were retrieved (may be empty). The API
            returns them so scores and provenance stay visible for debugging.
    """

    def __init__(self, reason: str, retrieved: list[RetrievedChunk]):
        self.reason = reason
        self.retrieved = retrieved
        super().__init__(reason)


def evaluate_retrieval(
    retrieved: list[RetrievedChunk],
    min_score: float | None = None,
) -> None:
    """Raise InsufficientContextError if retrieval is not sufficient.

    Decision rules, in order:
      1. No chunks retrieved at all -> insufficient.
      2. Top-1 score < min_score   -> insufficient.

    The threshold defaults to settings.retrieval_min_score. It is an
    engineering starting point for the current toy corpus, to be calibrated
    later against a labeled evaluation dataset - it is not a scientifically
    derived constant.
    """
    if min_score is None:
        from backend.app.core.config import settings

        min_score = settings.retrieval_min_score

    if not retrieved:
        raise InsufficientContextError(
            reason="no documents retrieved for this question",
            retrieved=retrieved,
        )

    top_score = retrieved[0].score
    if top_score < min_score:
        raise InsufficientContextError(
            reason=(
                f"best retrieval score {top_score:.4f} is below the "
                f"configured minimum {min_score:.4f}"
            ),
            retrieved=retrieved,
        )
