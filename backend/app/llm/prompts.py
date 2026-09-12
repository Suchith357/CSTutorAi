"""Grounded tutor prompt construction for OSTutorAI.

Prompt contract (enforced in tests):
- The system instruction restricts the answer to the retrieved context, which
  is authoritative; model-pretrained knowledge is a fallback, never a source
  for claims that the sources do not support.
- It requires an explicit insufficiency statement when the sources are not enough.
- Every retrieved chunk appears as a numbered source with title and source file
  so the model can cite what it uses as [1], [2], ...
"""

SYSTEM_INSTRUCTION = """\
You are OSTutorAI, a patient tutor for Operating Systems students.

The retrieved sources provided in the conversation are the authoritative \
material for your answer.

Strict rules:
1. Use the retrieved sources as the basis of your answer. Explain concepts \
for an Operating Systems student and keep the important technical terminology.
2. Never invent facts, definitions, or examples that the sources do not \
support. Do not pretend information exists in the sources when it does not.
3. Prefer the retrieved educational material over unsupported model knowledge. \
If you add helpful background beyond the sources, say clearly that it is not \
from the sources.
4. If the retrieved sources do not contain enough information to answer the \
question, state explicitly that the available sources do not contain enough \
information. Never fabricate an answer.5. Teach, don't just dump text: define terms, explain step by step, and \
   structure the explanation so a student can follow it rather than simply \
   copying the retrieved text.
6. When you use information from a source, cite it as [1], [2], ... matching \
the numbered sources in the user message. Do not fabricate citations or \
source details.
"""


def format_context(chunks) -> str:
    """Render retrieved chunks as numbered sources for the prompt."""
    if not chunks:
        return ""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        citation = f"{meta.title} (source: {meta.source})"
        lines.append(f"Source [{i}]: {citation}\n{chunk.text}")
    return "\n\n".join(lines)


def build_grounded_prompt(
    question: str,
    chunks,
    retrieval_min_score: float | None = None,
) -> tuple[str, str]:
    """Return (system, user) messages for a grounded tutoring answer.

    When no chunks are retrieved, or the best score is below the configured
    threshold, the user message explicitly instructs the model to say the
    sources are insufficient instead of inventing an answer. (The /query
    endpoint normally short-circuits before the LLM in those cases; this
    branch keeps the prompt safe when the prompt builder is used directly.)
    """
    context = format_context(chunks)

    top_score = getattr(chunks[0], "score", None) if chunks else None
    insufficient = not chunks or (
        top_score is not None
        and retrieval_min_score is not None
        and top_score < retrieval_min_score
    )

    if not insufficient:
        user = (
            f"Retrieved sources:\n\n{context}\n\n"
            f"Student question: {question}\n\n"
            "Answer the student's question using only these sources. "
            "Cite sources as [1], [2], ... If they are insufficient to answer "
            "fully, say so explicitly."
        )
    else:
        user = (
            "Retrieved sources: (none)\n\n"
            f"Student question: {question}\n\n"
            "The available sources do not contain enough information to answer "
            "this question. Respond by explaining that clearly, and suggest "
            "what the student could study next. Do not invent an answer."
        )
    return SYSTEM_INSTRUCTION, user
