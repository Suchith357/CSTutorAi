"""Grounded tutor prompt construction for CSTutorAI.

Prompt contract (enforced in tests):
- The system instruction forbids inventing facts outside the retrieved context.
- It requires an explicit insufficiency statement when the sources are not enough.
- Every retrieved chunk appears as a numbered source with title and source file
  so the model can cite what it uses as [1], [2], ...
"""

SYSTEM_INSTRUCTION = """\
You are CSTutorAI, a patient tutor for Computer Science students.

You answer using ONLY the retrieved sources provided in the conversation.

Strict rules:
1. Base your answer only on the retrieved sources. Never invent facts,
   definitions, or examples that are not supported by them.
2. If the retrieved sources do not contain enough information to answer,
   state clearly that the available sources do not contain enough information.
3. Teach, don't just answer: explain step by step, define terms, and structure
   the explanation so a CS student can follow it.
4. When you use information from a source, cite it as [1], [2], ... matching
   the numbered sources in the user message.
5. Do not claim authority for anything that is not present in the sources.
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


def build_grounded_prompt(question: str, chunks) -> tuple[str, str]:
    """Return (system, user) messages for a grounded tutoring answer.

    When no chunks are retrieved, the user message explicitly instructs the
    model to say the sources are insufficient instead of inventing an answer.
    """
    context = format_context(chunks)
    if context:
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
